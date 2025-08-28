import os
import time
import wave
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import config
from config import ensure_dirs
from audio.recorder import Recorder
from transcribers.wav2vec2_transcriber import Wav2Vec2Transcriber
from transcribers.whisper_turbo_transcriber import WhisperTurboTranscriber
from transcribers.vosk_transcriber import VoskTranscriber
from utils.csv_logger import append_result_row
from utils.audio_pad import pad_wav_silence


class App:
	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Arabic Command Comparator (Wav2Vec2 + Whisper Turbo + Vosk)")

		ensure_dirs()

		self.recorder = Recorder(sample_rate=config.SAMPLE_RATE)
		self.output_wav_path = ""

		self.w2v2_model_var = tk.StringVar(value=config.W2V2_MODEL_ID)
		self.whisper_model_var = tk.StringVar(value=config.WHISPER_MODEL)
		self.vosk_model_var = tk.StringVar(value=config.VOSK_MODEL_PATH)

		self.w2v2 = None
		self.whisper = None
		self.vosk = None

		frm = ttk.Frame(root, padding=12)
		frm.grid(row=0, column=0, sticky="nsew")

		# Wav2Vec2 model id
		self.w2v2_label = ttk.Label(frm, text="Wav2Vec2 model (HF id)")
		self.w2v2_label.grid(row=0, column=0, sticky="w", pady=(0,6))
		self.w2v2_entry = ttk.Entry(frm, textvariable=self.w2v2_model_var, width=64)
		self.w2v2_entry.grid(row=0, column=1, sticky="ew", pady=(0,6))

		# Whisper model name
		self.whisper_label = ttk.Label(frm, text="Whisper model (faster-whisper)")
		self.whisper_label.grid(row=1, column=0, sticky="w", pady=(0,6))
		self.whisper_entry = ttk.Entry(frm, textvariable=self.whisper_model_var, width=64)
		self.whisper_entry.grid(row=1, column=1, sticky="ew", pady=(0,6))

		# Vosk model path
		self.vosk_label = ttk.Label(frm, text="Vosk model path")
		self.vosk_label.grid(row=2, column=0, sticky="w", pady=(0,6))
		self.vosk_entry = ttk.Entry(frm, textvariable=self.vosk_model_var, width=64)
		self.vosk_entry.grid(row=2, column=1, sticky="ew", pady=(0,6))

		# Record button and status
		self.record_btn = ttk.Button(frm, text="Hold to Record")
		self.record_btn.grid(row=3, column=0, padx=(0,8))
		self.record_btn.bind("<ButtonPress-1>", self._on_press)
		self.record_btn.bind("<ButtonRelease-1>", self._on_release)

		self.status_var = tk.StringVar(value="Ready")
		self.status_lbl = ttk.Label(frm, textvariable=self.status_var)
		self.status_lbl.grid(row=3, column=1, sticky="w")

		# Results
		self.text = tk.Text(frm, width=84, height=20)
		self.text.grid(row=4, column=0, columnspan=2, pady=(10,0))

		frm.columnconfigure(1, weight=1)

	def _on_press(self, _event) -> None:
		self.status_var.set("Recording…")
		self.output_wav_path = os.path.join(
			config.RECORDINGS_DIR,
			f"cmd_{int(time.time())}.wav",
		)
		self.recorder.start()

	def _on_release(self, _event) -> None:
		try:
			# Stop stream and leave frames in buffer; we'll VAD-trim into a temp WAV
			if self.recorder._stream is not None:
				self.recorder._stream.stop()
				self.recorder._stream.close()
				self.recorder._stream = None
		except Exception as exc:
			messagebox.showerror("Recording error", str(exc))
			self.status_var.set("Error during recording")
			return

		self.status_var.set("Transcribing…")
		self.root.after(50, self._transcribe_current)

	def _ensure_transcribers(self) -> None:
		if self.w2v2 is None or self.w2v2.model_id != self.w2v2_model_var.get().strip():
			self.w2v2 = Wav2Vec2Transcriber(self.w2v2_model_var.get().strip())
		if self.whisper is None or self.whisper.model_name != self.whisper_model_var.get().strip():
			self.whisper = WhisperTurboTranscriber(self.whisper_model_var.get().strip())
		if self.vosk is None or self.vosk.model_path != self.vosk_model_var.get().strip():
			self.vosk = VoskTranscriber(self.vosk_model_var.get().strip())

	def _get_duration_ms(self, wav_path: str) -> int:
		try:
			with wave.open(wav_path, "rb") as wf:
				frames = wf.getnframes()
				rate = wf.getframerate()
				if rate <= 0:
					return 0
				return int(frames * 1000 / rate)
		except Exception:
			return 0

	def _transcribe_current(self) -> None:
		self._ensure_transcribers()
		# Produce a VAD-trimmed temp WAV from buffered frames
		try:
			vad_path = self.recorder.record_with_vad(
				aggressiveness=config.VAD_AGGRESSIVENESS,
				frame_ms=config.VAD_FRAME_MS,
				ring_ms=config.RING_BUFFER_MS,
				min_speech_ms=config.MIN_SPEECH_MS,
			)
		except Exception as exc:
			# Fall back to raw file when VAD fails
			vad_path = self.output_wav_path

		wav_path = vad_path

		duration_ms = self._get_duration_ms(wav_path)
		# Simple SNR estimate (RMS speech vs overall, coarse)
		try:
			import numpy as np  # type: ignore
			with wave.open(wav_path, "rb") as wf:
				raw = wf.readframes(wf.getnframes())
				sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
				rms = np.sqrt(np.mean(sig**2) + 1e-6)
				# crude noise floor estimate using first 200ms
				noise_samples = int(wf.getframerate() * 0.2)
				noise = sig[: max(1, min(noise_samples, sig.size))]
				rms_n = np.sqrt(np.mean(noise**2) + 1e-6)
				snr_db = float(20.0 * np.log10(max(rms, 1e-3) / max(rms_n, 1e-3)))
		except Exception:
			snr_db = 20.0

		w2v2_text = ""
		whisper_text = ""
		vosk_text = ""
		w2v2_time_ms = 0
		whisper_time_ms = 0
		vosk_time_ms = 0
		# Initialize to avoid UnboundLocalError and guide fallback logic
		w2v2_conf = 0.0
		whisper_used = False
		errors = []

		total_start_time = time.time()

		try:
			start_time = time.time()
			w2v2_text, w2v2_conf = self.w2v2.transcribe_with_confidence(wav_path)
			end_time = time.time()
			w2v2_time_ms = int((end_time - start_time) * 1000)
		except Exception as exc:
			errors.append(f"Wav2Vec2: {exc}")

		# Decide if we need Whisper fallback (include SNR/duration heuristics)
		need_whisper = (
			config.ALWAYS_RUN_WHISPER
			or (duration_ms >= config.WHISPER_FALLBACK_LONG_MS)
			or (w2v2_conf > 0 and w2v2_conf < (config.W2V2_CONF_THRESHOLD if snr_db >= config.SNR_LOW_DB else (config.W2V2_CONF_THRESHOLD + 0.1)))
			or (not w2v2_text.strip())
		)

		if need_whisper:
			try:
				start_time = time.time()
				whisper_text = self.whisper.transcribe(wav_path)
				end_time = time.time()
				whisper_time_ms = int((end_time - start_time) * 1000)
				whisper_used = True
			except Exception as exc:
				errors.append(f"Whisper: {exc}")

		try:
			vosk_input = wav_path
			if config.PAD_SHORT_FOR_VOSK and duration_ms < config.MIN_VOSK_MS:
				with tempfile.NamedTemporaryFile(prefix="vosk_pad_", suffix=".wav", delete=False) as tf:
					tmp_path = tf.name
				try:
					pad_wav_silence(wav_path, tmp_path, config.MIN_VOSK_MS)
					vosk_input = tmp_path
				except Exception as exc:
					errors.append(f"Pad Vosk: {exc}")
			start_time = time.time()
			vosk_text = self.vosk.transcribe(vosk_input)
			end_time = time.time()
			vosk_time_ms = int((end_time - start_time) * 1000)
		except Exception as exc:
			errors.append(f"Vosk: {exc}")
		finally:
			# Clean temp file if created
			try:
				if 'tmp_path' in locals() and os.path.exists(tmp_path):
					os.unlink(tmp_path)
			except Exception:
				pass

		# Cleanup VAD temp
		try:
			if vad_path != self.output_wav_path and os.path.exists(vad_path):
				os.unlink(vad_path)
		except Exception:
			pass

		total_end_time = time.time()
		total_processing_time_ms = int((total_end_time - total_start_time) * 1000)

		row = {
			"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
			"audio_file": os.path.basename(wav_path),
			"audio_duration_ms": duration_ms,
			"wav2vec2": w2v2_text,
			"w2v2_confidence": f"{w2v2_conf:.3f}",
			"whisper_turbo": whisper_text,
			"whisper_used": str(whisper_used),
			"vosk": vosk_text,
			"wav2vec2_time_ms": w2v2_time_ms,
			"whisper_time_ms": whisper_time_ms,
			"vosk_time_ms": vosk_time_ms,
			"total_processing_time_ms": total_processing_time_ms,
		}
		append_result_row(row)

		self.text.insert("end", f"File: {row['audio_file']}  ({duration_ms} ms)\n")
		self.text.insert("end", f"Wav2Vec2 ({w2v2_time_ms}ms, conf={w2v2_conf:.2f}): {w2v2_text}\n")
		if whisper_used:
			self.text.insert("end", f"Whisper ({whisper_time_ms}ms): {whisper_text}\n")
		else:
			self.text.insert("end", f"Whisper: skipped (policy)\n")
		self.text.insert("end", f"Vosk ({vosk_time_ms}ms): {vosk_text}\n")
		self.text.insert("end", f"Total processing time: {total_processing_time_ms}ms\n")
		if errors:
			self.text.insert("end", "Errors: \n  - " + "\n  - ".join(errors) + "\n")
		self.text.insert("end", "\n")
		self.text.see("end")
		self.status_var.set("Done")


def main() -> None:
	root = tk.Tk()
	App(root)
	root.mainloop()


if __name__ == "__main__":
	main()

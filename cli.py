import os
import sys
import time
import termios
import tty
import wave
from datetime import datetime

import config
from config import ensure_dirs
from audio.recorder import Recorder
from transcribers.wav2vec2_transcriber import Wav2Vec2Transcriber
from transcribers.whisper_turbo_transcriber import WhisperTurboTranscriber
from utils.csv_logger import append_result_row
from utils.text_map import map_to_commands, normalize_ar


def _getch() -> str:
	fd = sys.stdin.fileno()
	old = termios.tcgetattr(fd)
	try:
		tty.setraw(fd)
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old)
	return ch


def _snr_db(wav_path: str) -> float:
	try:
		import numpy as np  # type: ignore
		with wave.open(wav_path, "rb") as wf:
			raw = wf.readframes(wf.getnframes())
			sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
			rms = float(np.sqrt(np.mean(sig**2) + 1e-6))
			noise_samples = int(wf.getframerate() * 0.2)
			noise = sig[: max(1, min(noise_samples, sig.size))]
			rms_n = float(np.sqrt(np.mean(noise**2) + 1e-6))
			return float(20.0 * np.log10(max(rms, 1e-3) / max(rms_n, 1e-3)))
	except Exception:
		return 20.0


def _duration_ms(wav_path: str) -> int:
	try:
		with wave.open(wav_path, "rb") as wf:
			frames = wf.getnframes()
			rate = wf.getframerate()
			if rate <= 0:
				return 0
			return int(frames * 1000 / rate)
	except Exception:
		return 0


def run_once(recorder: Recorder, w2v2: Wav2Vec2Transcriber, whisper: WhisperTurboTranscriber) -> None:
	# Start recording
	print("Press 's' to stop recording…")
	recorder.start()
	while True:
		ch = _getch()
		if ch.lower() == 's':
			break
	# Stop stream (leave frames buffered for VAD trim)
	try:
		if recorder._stream is not None:
			recorder._stream.stop()
			recorder._stream.close()
			recorder._stream = None
	except Exception as exc:
		print(f"Recording error: {exc}")
		return

	# VAD-trim buffered audio
	try:
		vad_path = recorder.record_with_vad(
			aggressiveness=config.VAD_AGGRESSIVENESS,
			frame_ms=config.VAD_FRAME_MS,
			ring_ms=config.RING_BUFFER_MS,
			min_speech_ms=config.MIN_SPEECH_MS,
		)
	except Exception as exc:
		print(f"VAD error: {exc}")
		return

	wav_path = vad_path
	snr = _snr_db(wav_path)
	duration = _duration_ms(wav_path)

	# KWS gate
	kws_passed = True
	kws_score = 1.0
	if config.KWS_ENABLED:
		try:
			from kws.temporal_gate import temporal_gate
			kws_passed, kws_score = temporal_gate(wav_path)
		except Exception as exc:
			print(f"KWS error: {exc}")
			kws_passed = False

	if config.KWS_ENABLED and not kws_passed:
		print(f"KWS rejected (score={kws_score:.2f}).")
		try:
			if os.path.exists(vad_path) and vad_path != "":
				os.unlink(vad_path)
		except Exception:
			pass
		return

	# STT pipeline
	w2v2_text = ""
	w2v2_conf = 0.0
	w2v2_time_ms = 0
	whisper_text = ""
	whisper_used = False
	whisper_time_ms = 0

	start = time.time()
	w2v2_text, w2v2_conf = w2v2.transcribe_with_confidence(wav_path)
	w2v2_time_ms = int((time.time() - start) * 1000)
	w2v2_mapped, w2v2_dist = map_to_commands(w2v2_text, config.COMMANDS, config.CMD_MAP_MAX_DISTANCE)
	w2v2_norm = (w2v2_dist / max(1, len(normalize_ar(w2v2_text)))) if w2v2_dist < 10**9 else 1.0

	need_whisper = (
		config.ALWAYS_RUN_WHISPER
		or (duration >= config.WHISPER_FALLBACK_LONG_MS)
		or (w2v2_conf > 0 and w2v2_conf < (config.W2V2_CONF_THRESHOLD if snr >= config.SNR_LOW_DB else (config.W2V2_CONF_THRESHOLD + 0.1)))
		or (not w2v2_text.strip())
		or (not w2v2_mapped and w2v2_norm > 0.35)
	)

	whisper_mapped = ""
	if need_whisper:
		start = time.time()
		whisper_text = whisper.transcribe(wav_path)
		whisper_time_ms = int((time.time() - start) * 1000)
		whisper_used = True
		whisper_mapped, _ = map_to_commands(whisper_text, config.COMMANDS, config.CMD_MAP_MAX_DISTANCE)

	# Append CSV
	row = {
		"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
		"audio_file": os.path.basename(wav_path),
		"audio_duration_ms": duration,
		"kws_passed": str(kws_passed),
		"kws_score": f"{kws_score:.3f}",
		"wav2vec2": w2v2_text,
		"w2v2_confidence": f"{w2v2_conf:.3f}",
		"wav2vec2_mapped": w2v2_mapped,
		"whisper_turbo": whisper_text,
		"whisper_used": str(whisper_used),
		"whisper_mapped": whisper_mapped,
		"wav2vec2_time_ms": w2v2_time_ms,
		"whisper_time_ms": whisper_time_ms,
		"total_processing_time_ms": (w2v2_time_ms + whisper_time_ms),
	}
	append_result_row(row)

	# Print summary
	print(f"KWS: passed={kws_passed} score={kws_score:.2f}  SNR={snr:.1f}dB  dur={duration}ms")
	print(f"W2V2 ({w2v2_time_ms}ms, conf={w2v2_conf:.2f}): {w2v2_text}")
	print(f"W2V2 mapped: {w2v2_mapped or '-'}")
	if whisper_used:
		print(f"Whisper ({whisper_time_ms}ms): {whisper_text}")
		print(f"Whisper mapped: {whisper_mapped or '-'}")

	# Cleanup temp
	try:
		if os.path.exists(vad_path):
			os.unlink(vad_path)
	except Exception:
		pass


def main() -> None:
	ensure_dirs()
	recorder = Recorder(sample_rate=config.SAMPLE_RATE)
	w2v2 = Wav2Vec2Transcriber(config.W2V2_MODEL_ID)
	whisper = WhisperTurboTranscriber(config.WHISPER_MODEL)
	print("CLI ready. Press 'r' to record, 'q' to quit.")
	while True:
		ch = _getch().lower()
		if ch == 'q':
			break
		if ch == 'r':
			run_once(recorder, w2v2, whisper)
	print("Bye.")


if __name__ == "__main__":
	main()

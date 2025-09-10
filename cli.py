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
import torch  # type: ignore
import torchaudio  # type: ignore
from models.mfcc_cnn import MFCC_CNN
from utils.text_map import map_to_commands, normalize_ar
import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore
import webrtcvad  # type: ignore
import collections


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


def _load_mfcc_cnn():
	if not config.MFCC_CNN_ENABLED:
		return None, None
	try:
		state = torch.load(config.MFCC_CNN_CKPT_PATH, map_location="cpu")
		label_map = state["label_map"]
		rev_map = {v: k for k, v in label_map.items()}
		model = MFCC_CNN(num_classes=len(label_map), dropout=0.30)
		model.load_state_dict(state["model_state"])
		model.eval()
		return model, rev_map
	except Exception as exc:
		print(f"MFCC-CNN load error: {exc}")
		return None, None


def _extract_features(wav_path: str) -> torch.Tensor:
	waveform, sr = torchaudio.load(wav_path)
	if sr != config.SAMPLE_RATE:
		waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)
		sr = config.SAMPLE_RATE
	if waveform.shape[0] > 1:
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	win_len = int(config.MFCC_WIN_MS * sr / 1000)
	hop_len = int(config.MFCC_HOP_MS * sr / 1000)
	if config.MFCC_FEATURE_TYPE == "mfcc":
		feat_t = torchaudio.transforms.MFCC(
			sample_rate=sr,
			n_mfcc=config.MFCC_NUM_MFCC,
			melkwargs={
				"n_mels": config.MFCC_NUM_MELS,
				"win_length": win_len,
				"hop_length": hop_len,
				"f_min": config.MFCC_FMIN,
				"f_max": config.MFCC_FMAX,
			},
		)
		feat = feat_t(waveform).squeeze(0)
	else:
		mel = torchaudio.transforms.MelSpectrogram(
			sample_rate=sr,
			n_mels=config.MFCC_NUM_MELS,
			win_length=win_len,
			hop_length=hop_len,
			f_min=config.MFCC_FMIN,
			f_max=config.MFCC_FMAX,
		)(waveform)
		feat = torchaudio.transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)
	if config.MFCC_CMVN:
		mean = feat.mean(dim=1, keepdim=True)
		std = feat.std(dim=1, keepdim=True) + 1e-6
		feat = (feat - mean) / std
	if config.MFCC_DELTAS:
		d1 = torchaudio.functional.compute_deltas(feat)
		d2 = torchaudio.functional.compute_deltas(d1)
		feat = torch.cat([feat, d1, d2], dim=0)
	return feat.unsqueeze(0).unsqueeze(0)


def _cnn_predict_sliding(wav_path: str, model: MFCC_CNN, rev_map: dict[int, str]) -> tuple[str, float]:
	# Slide over raw waveform using window/hop in ms, aggregate softmax by average
	waveform, sr = torchaudio.load(wav_path)
	if sr != config.SAMPLE_RATE:
		waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)
		sr = config.SAMPLE_RATE
	if waveform.shape[0] > 1:
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	win_samps = int(config.MFCC_SLIDE_WINDOW_MS * sr / 1000)
	hop_samps = int(config.MFCC_SLIDE_HOP_MS * sr / 1000)
	T = waveform.shape[1]
	if T < max(1, win_samps):
		# pad to one window
		pad = win_samps - T
		waveform = torch.nn.functional.pad(waveform, (0, pad))
		T = waveform.shape[1]
	starts = list(range(0, max(1, T - win_samps + 1), max(1, hop_samps)))
	if not starts:
		starts = [0]
	probs_accum = None
	for s in starts:
		chunk = waveform[:, s:s+win_samps]
		# Save temp chunk to feature pipeline by reusing transform
		# Build features for the chunk
		win_len = int(config.MFCC_WIN_MS * sr / 1000)
		hop_len = int(config.MFCC_HOP_MS * sr / 1000)
		if config.MFCC_FEATURE_TYPE == "mfcc":
			feat_t = torchaudio.transforms.MFCC(
				sample_rate=sr,
				n_mfcc=config.MFCC_NUM_MFCC,
				melkwargs={
					"n_mels": config.MFCC_NUM_MELS,
					"win_length": win_len,
					"hop_length": hop_len,
					"f_min": config.MFCC_FMIN,
					"f_max": config.MFCC_FMAX,
				},
			)
			feat = feat_t(chunk).squeeze(0)
		else:
			mel = torchaudio.transforms.MelSpectrogram(
				sample_rate=sr,
				n_mels=config.MFCC_NUM_MELS,
				win_length=win_len,
				hop_length=hop_len,
				f_min=config.MFCC_FMIN,
				f_max=config.MFCC_FMAX,
			)(chunk)
			feat = torchaudio.transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)
		if config.MFCC_CMVN:
			mean = feat.mean(dim=1, keepdim=True)
			std = feat.std(dim=1, keepdim=True) + 1e-6
			feat = (feat - mean) / std
		if config.MFCC_DELTAS:
			d1 = torchaudio.functional.compute_deltas(feat)
			d2 = torchaudio.functional.compute_deltas(d1)
			feat = torch.cat([feat, d1, d2], dim=0)
		x = feat.unsqueeze(0).unsqueeze(0)
		with torch.no_grad():
			logits = model(x)
			probs = torch.softmax(logits, dim=1).squeeze(0)
			probs_accum = probs if probs_accum is None else (probs_accum + probs)
	if probs_accum is None:
		return "", 0.0
	probs_mean = probs_accum / max(1, len(starts))
	conf, idx = torch.max(probs_mean, dim=0)
	label = rev_map[int(idx.item())]
	return label, float(conf.item())


def _beep(freq_hz: int = 1000, dur_ms: int = 120, volume: float = 0.2) -> None:
	try:
		sr = config.SAMPLE_RATE
		t = np.linspace(0, dur_ms / 1000.0, int(sr * (dur_ms / 1000.0)), False)
		tone = (np.sin(2 * np.pi * freq_hz * t) * volume).astype(np.float32)
		sd.play(tone, sr)
		sd.wait()
	except Exception:
		try:
			print("\a", end="", flush=True)
		except Exception:
			pass


def _write_wav_from_frames(frames: list[bytes], sample_rate: int) -> str:
	import tempfile, wave
	with tempfile.NamedTemporaryFile(prefix="stream_seg_", suffix=".wav", delete=False) as tf:
		path = tf.name
	with wave.open(path, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		for fr in frames:
			wf.writeframes(fr)
	return path


def always_on_loop(recorder: Recorder, w2v2: Wav2Vec2Transcriber, whisper: WhisperTurboTranscriber) -> None:
	print("Always-on mode: Press Ctrl+C to quit.")
	vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
	frame_ms = config.VAD_FRAME_MS
	ring_ms = config.RING_BUFFER_MS
	min_speech_ms = config.MIN_SPEECH_MS
	end_silence_ms = 600
	end_silence_frames = max(1, int(end_silence_ms / frame_ms))
	frame_bytes = int(config.SAMPLE_RATE * (frame_ms / 1000.0)) * 2
	ring_frames: collections.deque[bytes] = collections.deque(maxlen=int(ring_ms / frame_ms))  # type: ignore
	segment_frames: list[bytes] = []
	speech_active = False
	silence_streak = 0
	leftover = b""

	# CNN
	mfcc_model, mfcc_rev = _load_mfcc_cnn()
	ema_probs: torch.Tensor | None = None
	ema_alpha = 0.6
	above_count = 0
	require_consecutive = 2

	# Announce CNN availability
	if mfcc_model is not None:
		print(f"MFCC-CNN ready ({len(mfcc_rev)} classes). Threshold={config.MFCC_CNN_CONF_THRESHOLD}")
	else:
		print(f"MFCC-CNN not available. Check checkpoint at: {config.MFCC_CNN_CKPT_PATH}")

	recorder.start()
	try:
		_beep(880, 80, 0.15)
		import time as _time
		last_window_time = 0.0
		hop_sec = max(0.05, config.MFCC_SLIDE_HOP_MS / 1000.0)
		win_samps = int(config.MFCC_SLIDE_WINDOW_MS * config.SAMPLE_RATE / 1000)
		audio_bytes_buf = b""
		max_buf_bytes = int(config.SAMPLE_RATE * 2 * 3)
		while True:
			# Drain new frames from recorder
			with recorder._lock:  # type: ignore[attr-defined]
				new_chunks = list(recorder._frames)  # type: ignore[attr-defined]
				recorder._frames = []  # type: ignore[attr-defined]
			if new_chunks:
				chunk_bytes = b"".join(new_chunks)
				audio_bytes_buf += chunk_bytes
				if len(audio_bytes_buf) > max_buf_bytes:
					audio_bytes_buf = audio_bytes_buf[-max_buf_bytes:]

				data = leftover + chunk_bytes
				# Iterate in VAD frames
				for i in range(0, len(data) - (len(data) % frame_bytes), frame_bytes):
					fr = data[i:i+frame_bytes]
					is_speech = False
					try:
						is_speech = vad.is_speech(fr, config.SAMPLE_RATE)
					except Exception:
						is_speech = False
					ring_frames.append(fr)
					if not speech_active and is_speech:
						# Speech onset
						_beep(1200, 60, 0.15)
						speech_active = True
						segment_frames.extend(list(ring_frames))
						ring_frames.clear()
						segment_frames.append(fr)
						silence_streak = 0
					elif speech_active:
						segment_frames.append(fr)
						if is_speech:
							silence_streak = 0
						else:
							silence_streak += 1
							if silence_streak >= end_silence_frames:
								# Finalize segment
								silence_streak = 0
								speech_active = False
								# Enforce minimum speech length
								seg_ms = int(len(segment_frames) * frame_ms)
								if seg_ms < min_speech_ms:
									segment_frames.clear()
								else:
									wav_path = _write_wav_from_frames(segment_frames, config.SAMPLE_RATE)
									segment_frames.clear()
									# Decide via CNN first
									mfcc_label = ""
									mfcc_conf = 0.0
									mfcc_used = False
									if mfcc_model is not None:
										try:
											pred_label, pred_conf = _cnn_predict_sliding(wav_path, mfcc_model, mfcc_rev)
											mfcc_label = pred_label
											mfcc_conf = pred_conf
											mfcc_used = True
										except Exception as exc:
											print(f"MFCC-CNN inference error: {exc}")
									# Always log CNN prediction (even if we will fall back)
									if mfcc_used:
										print(f"MFCC-CNN: {mfcc_label} (conf={mfcc_conf:.2f})")
									cnn_accept = (
										config.MFCC_CNN_ENABLED and mfcc_used and mfcc_conf >= config.MFCC_CNN_CONF_THRESHOLD
									)
									if cnn_accept and config.MFCC_CNN_REQUIRE_MAPPING:
										mapped_from_cnn = config.MFCC_LABEL_TO_COMMAND.get(mfcc_label, "")
										if mapped_from_cnn:
											mfcc_label = mapped_from_cnn
										else:
											cnn_accept = False
											print("CNN label not in MFCC_LABEL_TO_COMMAND mapping; falling back to ASR")
									elif mfcc_used and mfcc_conf < config.MFCC_CNN_CONF_THRESHOLD:
										print(f"CNN below threshold ({mfcc_conf:.2f} < {config.MFCC_CNN_CONF_THRESHOLD}); falling back to ASR")
									# STT fallback if not accepted
									w2v2_text = ""
									w2v2_conf = 0.0
									w2v2_time_ms = 0
									whisper_text = ""
									whisper_used = False
									whisper_time_ms = 0
									if not cnn_accept:
										start_t = _time.time()
										w2v2_text, w2v2_conf = w2v2.transcribe_with_confidence(wav_path)
										w2v2_time_ms = int((_time.time() - start_t) * 1000)
										w2v2_mapped, w2v2_dist = map_to_commands(w2v2_text, config.COMMANDS, config.CMD_MAP_MAX_DISTANCE)
										w2v2_norm = (w2v2_dist / max(1, len(normalize_ar(w2v2_text)))) if w2v2_dist < 10**9 else 1.0
										need_whisper = (
											config.ALWAYS_RUN_WHISPER
											or (int(seg_ms) >= config.WHISPER_FALLBACK_LONG_MS)
											or (w2v2_conf > 0 and w2v2_conf < (config.W2V2_CONF_THRESHOLD))
											or (not w2v2_text.strip())
											or (not w2v2_mapped and w2v2_norm > 0.35)
										)
										whisper_mapped = ""
										if need_whisper:
											start_t = _time.time()
											whisper_text = whisper.transcribe(wav_path)
											whisper_time_ms = int((_time.time() - start_t) * 1000)
											whisper_used = True
											whisper_mapped, _ = map_to_commands(whisper_text, config.COMMANDS, config.CMD_MAP_MAX_DISTANCE)
									else:
										w2v2_text = ""
										w2v2_mapped = ""
										whisper_mapped = ""

									# Append CSV
									row = {
										"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
										"audio_file": os.path.basename(wav_path),
										"audio_duration_ms": seg_ms,
										"mfcc_label": mfcc_label,
										"mfcc_confidence": f"{mfcc_conf:.3f}",
										"mfcc_used": str(mfcc_used),
										"wav2vec2": w2v2_text,
										"w2v2_confidence": f"{w2v2_conf:.3f}",
										"wav2vec2_mapped": w2v2_mapped if not cnn_accept else "",
										"whisper_turbo": whisper_text,
										"whisper_used": str(whisper_used),
										"whisper_mapped": whisper_mapped,
										"wav2vec2_time_ms": w2v2_time_ms,
										"whisper_time_ms": whisper_time_ms,
										"total_processing_time_ms": (w2v2_time_ms + whisper_time_ms),
									}
									append_result_row(row)
									if cnn_accept:
										_beep(1500, 80, 0.2)
										print(f"CNN accepted: {mfcc_label} (conf={mfcc_conf:.2f})")
									else:
										print(f"W2V2 ({w2v2_time_ms}ms, conf={w2v2_conf:.2f}): {w2v2_text}")
										if whisper_used:
											print(f"Whisper ({whisper_time_ms}ms): {whisper_text}")
									try:
										if os.path.exists(wav_path):
											os.unlink(wav_path)
									except Exception:
										pass

				leftover_len = len(data) % frame_bytes
				leftover = data[-leftover_len:] if leftover_len > 0 else b""

			# Periodic window-level smoothing (informational)
			now = _time.time()
			if mfcc_model is not None and (now - last_window_time) >= hop_sec and len(audio_bytes_buf) >= win_samps * 2:
				last_window_time = now
				win_bytes = audio_bytes_buf[-win_samps*2:]
				wave_np = np.frombuffer(win_bytes, dtype=np.int16).astype(np.float32) / 32768.0
				wave_t = torch.from_numpy(wave_np).unsqueeze(0)
				# Build features for the window
				sr = config.SAMPLE_RATE
				win_len = int(config.MFCC_WIN_MS * sr / 1000)
				hop_len = int(config.MFCC_HOP_MS * sr / 1000)
				if config.MFCC_FEATURE_TYPE == "mfcc":
					feat_t = torchaudio.transforms.MFCC(
						sample_rate=sr,
						n_mfcc=config.MFCC_NUM_MFCC,
						melkwargs={
							"n_mels": config.MFCC_NUM_MELS,
							"win_length": win_len,
							"hop_length": hop_len,
							"f_min": config.MFCC_FMIN,
							"f_max": config.MFCC_FMAX,
						},
					)
					feat = feat_t(wave_t).squeeze(0)
				else:
					mel = torchaudio.transforms.MelSpectrogram(
						sample_rate=sr,
						n_mels=config.MFCC_NUM_MELS,
						win_length=win_len,
						hop_length=hop_len,
						f_min=config.MFCC_FMIN,
						f_max=config.MFCC_FMAX,
					)(wave_t)
					feat = torchaudio.transforms.AmplitudeToDB(stype="power")(mel).squeeze(0)
				if config.MFCC_CMVN:
					mean = feat.mean(dim=1, keepdim=True)
					std = feat.std(dim=1, keepdim=True) + 1e-6
					feat = (feat - mean) / std
				if config.MFCC_DELTAS:
					d1 = torchaudio.functional.compute_deltas(feat)
					d2 = torchaudio.functional.compute_deltas(d1)
					feat = torch.cat([feat, d1, d2], dim=0)
				x = feat.unsqueeze(0).unsqueeze(0)
				with torch.no_grad():
					logits = mfcc_model(x) if mfcc_model is not None else None
				if logits is not None:
					probs = torch.softmax(logits, dim=1).squeeze(0)
					ema_probs = probs if ema_probs is None else (ema_alpha * probs + (1.0 - ema_alpha) * ema_probs)
					conf, idx = torch.max(ema_probs, dim=0)
					if float(conf.item()) >= config.MFCC_CNN_CONF_THRESHOLD:
						above_count += 1
					else:
						above_count = 0
			_time.sleep(0.02)
	except KeyboardInterrupt:
		print("\nStopping…")
	finally:
		try:
			if recorder._stream is not None:
				recorder._stream.stop()
				recorder._stream.close()
				recorder._stream = None
		except Exception:
			pass


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

	# KWS removed

	# Optional MFCC-CNN front-end
	mfcc_label = ""
	mfcc_conf = 0.0
	mfcc_used = False
	mfcc_model, mfcc_rev = _load_mfcc_cnn()
	if mfcc_model is not None:
		try:
			# Use sliding-window inference
			pred_label, pred_conf = _cnn_predict_sliding(wav_path, mfcc_model, mfcc_rev)
			mfcc_label = pred_label
			mfcc_conf = pred_conf
			mfcc_used = True
		except Exception as exc:
			print(f"MFCC-CNN inference error: {exc}")

	# If gating is enabled and MFCC-CNN is confident, accept and skip STT
	# Optionally require mapping to Arabic command before accepting
	cnn_accept = (
		config.MFCC_CNN_ENABLED
		and config.MFCC_CNN_GATE_STT
		and mfcc_used
		and mfcc_conf >= config.MFCC_CNN_CONF_THRESHOLD
	)
	if cnn_accept:
		mapped_from_cnn = config.MFCC_LABEL_TO_COMMAND.get(mfcc_label, "")
		if mapped_from_cnn:
			mfcc_label = mapped_from_cnn
		else:
			cnn_accept = False

	if cnn_accept:
		row = {
			"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
			"audio_file": os.path.basename(wav_path),
			"audio_duration_ms": duration,
			"mfcc_label": mfcc_label,
			"mfcc_confidence": f"{mfcc_conf:.3f}",
			"mfcc_used": "True",
			"wav2vec2": "",
			"w2v2_confidence": "",
			"wav2vec2_mapped": "",
			"whisper_turbo": "",
			"whisper_used": "False",
			"whisper_mapped": "",
			"wav2vec2_time_ms": 0,
			"whisper_time_ms": 0,
			"total_processing_time_ms": 0,
		}
		append_result_row(row)
		print(f"MFCC-CNN accepted: {mfcc_label} (conf={mfcc_conf:.2f}) — skipping STT")
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
		"mfcc_label": mfcc_label,
		"mfcc_confidence": f"{mfcc_conf:.3f}",
		"mfcc_used": str(mfcc_used),
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
	print(f"SNR={snr:.1f}dB  dur={duration}ms")
	if mfcc_used:
		print(f"MFCC-CNN: {mfcc_label} (conf={mfcc_conf:.2f})")
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
	if "--always_on" in sys.argv:
		always_on_loop(recorder, w2v2, whisper)
		print("Bye.")
		return
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

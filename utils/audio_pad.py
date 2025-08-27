import wave
import os


def pad_wav_silence(src_wav: str, dst_wav: str, target_ms: int) -> None:
	with wave.open(src_wav, "rb") as rf:
		channels = rf.getnchannels()
		width = rf.getsampwidth()
		rate = rf.getframerate()
		frames = rf.getnframes()
		data = rf.readframes(frames)

	# Only handle 16-bit mono 16kHz as produced by our recorder
	if channels != 1 or width != 2 or rate <= 0:
		raise ValueError("pad_wav_silence expects mono 16-bit WAV")

	current_ms = int(frames * 1000 / rate)
	if current_ms >= target_ms:
		# No padding needed; copy file
		if os.path.abspath(src_wav) != os.path.abspath(dst_wav):
			with open(src_wav, "rb") as s, open(dst_wav, "wb") as d:
				d.write(s.read())
		return

	target_frames = int((target_ms * rate) / 1000)
	pad_frames = max(0, target_frames - frames)
	pad_bytes = pad_frames * width * channels
	pad_data = b"\x00" * pad_bytes

	with wave.open(dst_wav, "wb") as wf:
		wf.setnchannels(channels)
		wf.setsampwidth(width)
		wf.setframerate(rate)
		wf.writeframes(data)
		wf.writeframes(pad_data)

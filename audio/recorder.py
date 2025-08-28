import sounddevice as sd
import wave
import threading
import collections
import tempfile
from typing import Optional, List, Deque

import webrtcvad  # type: ignore


class Recorder:
	def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype: str = "int16") -> None:
		self.sample_rate = sample_rate
		self.channels = channels
		self.dtype = dtype
		self._stream: Optional[sd.RawInputStream] = None
		self._frames: List[bytes] = []
		self._lock = threading.Lock()

	def _callback(self, indata, frames, time, status):  # type: ignore[no-untyped-def]
		if status:
			pass
		with self._lock:
			self._frames.append(bytes(indata))

	def start(self) -> None:
		if self._stream is not None:
			return
		self._frames = []
		self._stream = sd.RawInputStream(
			samplerate=self.sample_rate,
			blocksize=8000,
			dtype=self.dtype,
			channels=self.channels,
			callback=self._callback,
		)
		self._stream.start()

	def stop_and_save(self, wav_path: str) -> None:
		if self._stream is None:
			return
		try:
			self._stream.stop()
			self._stream.close()
		finally:
			self._stream = None

		with self._lock:
			frames_copy = list(self._frames)
			self._frames = []

		with wave.open(wav_path, "wb") as wf:
			wf.setnchannels(self.channels)
			wf.setsampwidth(2)
			wf.setframerate(self.sample_rate)
			for chunk in frames_copy:
				wf.writeframes(chunk)

	def record_with_vad(self, aggressiveness: int, frame_ms: int, ring_ms: int, min_speech_ms: int) -> str:
		"""Blocks until user releases button; returns temp WAV path trimmed by VAD.
		Assumes start() was called on press and stop_and_save won't be used. We take frames
		directly from the buffer and write a temp file at the end.
		"""
		vad = webrtcvad.Vad(aggressiveness)
		frame_bytes = int(self.sample_rate * (frame_ms / 1000.0)) * 2  # int16 mono
		ring_frames: Deque[bytes] = collections.deque(maxlen=int(ring_ms / frame_ms))
		speech_chunks: List[bytes] = []
		speech_seen = False

		def consume_frames(frames: List[bytes]) -> None:
			for chunk in frames:
				# chunk is ~0.5s at current blocksize; iterate in frame_ms slices
				for i in range(0, len(chunk), frame_bytes):
					frame = chunk[i:i+frame_bytes]
					if len(frame) < frame_bytes:
						continue
					is_speech = False
					try:
						is_speech = vad.is_speech(frame, self.sample_rate)
					except Exception:
						# If VAD complains on frame alignment, treat as non-speech
						is_speech = False
					ring_frames.append(frame)
					if is_speech:
						nonlocal speech_seen
						speech_seen = True
						speech_chunks.extend(list(ring_frames))
						ring_frames.clear()
						speech_chunks.append(frame)
					elif speech_seen:
						# After speech starts, keep all frames (simple endpointing)
						speech_chunks.append(frame)

		# Drain collected frames up to stop()
		with self._lock:
			frames_copy = list(self._frames)
			self._frames = []
		consume_frames(frames_copy)

		# Write temp WAV if long enough
		duration_ms = int((len(speech_chunks) * frame_ms))
		with tempfile.NamedTemporaryFile(prefix="vad_trim_", suffix=".wav", delete=False) as tf:
			trim_path = tf.name
		if duration_ms < min_speech_ms:
			# Not enough speech; still write the ring buffer (best effort) for consistency
			payload = b"".join(list(ring_frames)) if ring_frames else b"".join(speech_chunks)
		else:
			payload = b"".join(speech_chunks)

		with wave.open(trim_path, "wb") as wf:
			wf.setnchannels(self.channels)
			wf.setsampwidth(2)
			wf.setframerate(self.sample_rate)
			wf.writeframes(payload)
		return trim_path


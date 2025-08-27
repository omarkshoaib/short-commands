import sounddevice as sd
import wave
import threading
from typing import Optional, List


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


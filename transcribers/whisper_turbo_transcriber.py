from typing import Optional

from faster_whisper import WhisperModel  # type: ignore

import config


class WhisperTurboTranscriber:
	def __init__(self, model_name: Optional[str] = None) -> None:
		self.model_name = model_name or config.WHISPER_MODEL
		device = config.STT_DEVICE if config.STT_DEVICE in ("cpu", "cuda") else "cpu"
		compute_type = config.WHISPER_COMPUTE_TYPE
		self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

	def transcribe(self, wav_path: str) -> str:
		segments, info = self._model.transcribe(
			wav_path,
			language="ar",
			beam_size=1,
		)
		text_parts = []
		for seg in segments:
			text_parts.append(seg.text)
		return " ".join(text_parts).strip()

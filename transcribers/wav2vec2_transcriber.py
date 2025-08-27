from typing import Optional

import torch  # type: ignore
import librosa  # type: ignore
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # type: ignore

import config
from utils.buckwalter import buckwalter_to_arabic


class Wav2Vec2Transcriber:
	def __init__(self, model_id: Optional[str] = None) -> None:
		self.model_id = model_id or config.W2V2_MODEL_ID
		self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
		self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
		preferred = config.STT_DEVICE
		self._device = preferred if preferred in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
		self._model.to(self._device)

	def transcribe(self, wav_path: str) -> str:
		audio, sr = librosa.load(wav_path, sr=16000)
		inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest")
		with torch.no_grad():
			logits = self._model(inputs.input_values.to(self._model.device)).logits
			pred_ids = torch.argmax(logits, dim=-1)
			text = self._processor.batch_decode(pred_ids)[0]
		return buckwalter_to_arabic(text.strip())

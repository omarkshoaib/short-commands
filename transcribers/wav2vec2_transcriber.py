from typing import Optional, Tuple

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import librosa  # type: ignore
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # type: ignore

import config
from utils.buckwalter import normalize_arabic_text


class Wav2Vec2Transcriber:
	def __init__(self, model_id: Optional[str] = None) -> None:
		self.model_id = model_id or config.W2V2_MODEL_ID
		self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
		self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
		preferred = config.STT_DEVICE
		self._device = preferred if preferred in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
		self._model.to(self._device)

	def transcribe(self, wav_path: str) -> str:
		text, _ = self.transcribe_with_confidence(wav_path)
		return text

	def transcribe_with_confidence(self, wav_path: str) -> Tuple[str, float]:
		audio, sr = librosa.load(wav_path, sr=16000)
		inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest")
		with torch.no_grad():
			logits = self._model(inputs.input_values.to(self._model.device)).logits  # [B, T, V]
			pred_ids = torch.argmax(logits, dim=-1)
			text = self._processor.batch_decode(pred_ids)[0]
			# Confidence: mean of max softmax probabilities across frames (ignoring padding frames)
			probs = F.softmax(logits, dim=-1)
			max_probs, _ = probs.max(dim=-1)  # [B, T]
			# Use attention mask if available
			if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
				mask = inputs.attention_mask.to(max_probs.device).bool()
				valid = max_probs[0][mask[0]] if max_probs.shape[0] > 0 else max_probs
			else:
				valid = max_probs[0]
			conf = float(valid.mean().item()) if valid.numel() else 0.0
		return normalize_arabic_text(text), conf

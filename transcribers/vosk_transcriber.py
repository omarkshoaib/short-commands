from typing import Optional, List
import json
import wave
import os
import vosk  # type: ignore

import config
from utils.buckwalter import normalize_arabic_text


class VoskTranscriber:
	def __init__(self, model_path: Optional[str] = None, grammar: Optional[List[str]] = None) -> None:
		self.model_path = model_path or config.VOSK_MODEL_PATH
		vosk.SetLogLevel(-1)
		self._model = vosk.Model(self.model_path)
		self._grammar = grammar
		# Static graph models contain graph/HCLG.fst and do not support runtime grammars
		self._supports_runtime_grammar = not os.path.exists(os.path.join(self.model_path, "graph", "HCLG.fst"))

	def _parse_grammar(self) -> List[str]:
		if self._grammar is not None:
			return self._grammar
		items = [s.strip() for s in (config.VOSK_GRAMMAR or "").split(",")]
		return [s for s in items if s]

	def transcribe(self, wav_path: str) -> str:
		try:
			with wave.open(wav_path, "rb") as wf:
				if wf.getnchannels() != 1:
					raise ValueError("Audio file must be mono (single channel)")
				if wf.getsampwidth() != 2:
					raise ValueError("Audio file must be 16-bit")
				rate = wf.getframerate()
				phrases = self._parse_grammar() if self._supports_runtime_grammar else []
				if phrases:
					rec = vosk.KaldiRecognizer(self._model, rate, json.dumps(phrases, ensure_ascii=False))
				else:
					rec = vosk.KaldiRecognizer(self._model, rate)
				rec.SetWords(True)
				parts = []
				while True:
					data = wf.readframes(4000)
					if len(data) == 0:
						break
					if rec.AcceptWaveform(data):
						res = json.loads(rec.Result())
						text = normalize_arabic_text((res.get("text") or "").strip())
						if text:
							parts.append(text)
				final = json.loads(rec.FinalResult())
				text = normalize_arabic_text((final.get("text") or "").strip())
				if text:
					parts.append(text)
				return " ".join(parts).strip()
		except Exception as e:
			raise RuntimeError(f"Vosk transcription failed: {e}")

import os
import glob
from typing import Tuple

import config
from .dtw_kws import enroll_templates, score_keyword


def _profile_dir() -> str:
	base = config.KWS_TEMPLATES_DIR
	pid = config.KWS_PROFILE_ID or "default"
	return os.path.join(base, pid)


def temporal_gate(wav_path: str) -> Tuple[bool, float]:
	"""Temporal integration + dynamic thresholds + optional secondary check.
	Return (passed, peak_score).
	"""
	dir_path = _profile_dir()
	if not os.path.exists(dir_path):
		try:
			os.makedirs(dir_path, exist_ok=True)
		except Exception:
			pass
	templates = enroll_templates(sorted(glob.glob(os.path.join(dir_path, "*.wav"))))
	score, _ = score_keyword(templates, wav_path, band_frac=config.KWS_BAND_FRAC)
	passed_primary = score >= config.KWS_THR_HIGH or score >= config.KWS_THR_LOW
	if not passed_primary:
		return False, float(score)
	if config.KWS_REQUIRE_SECONDARY:
		sec_score, _ = score_keyword(templates, wav_path, band_frac=config.KWS_SECONDARY_BAND_FRAC)
		passed_secondary = sec_score >= config.KWS_THR_LOW
		return (passed_secondary, float(max(score, sec_score)))
	return True, float(score)

import os
import glob
from typing import Tuple

import config
from .dtw_kws import enroll_templates, score_keyword


def temporal_gate(wav_path: str) -> Tuple[bool, float]:
	"""Window the audio with hop and integrate KWS scores; dynamic thresholds.
	Return (passed, peak_score).
	"""
	# For now, run a single-shot score; future: slice wav into windows and fuse
	templates = enroll_templates(sorted(glob.glob(os.path.join(config.KWS_TEMPLATES_DIR, "*.wav"))))
	score, raw = score_keyword(templates, wav_path, band_frac=config.KWS_BAND_FRAC)
	# Dynamic thresholding: use HIGH, else LOW if borderline
	passed = score >= config.KWS_THR_HIGH or score >= config.KWS_THR_LOW
	return passed, float(score)

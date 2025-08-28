import os
import librosa  # type: ignore
import numpy as np  # type: ignore
from typing import List, Tuple


def _mfcc(path: str, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
	y, s = librosa.load(path, sr=sr)
	mf = librosa.feature.mfcc(y=y, sr=s, n_mfcc=n_mfcc)
	return mf.T  # [frames, n_mfcc]


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
	# Simple DTW with Euclidean local cost
	na, nb = a.shape[0], b.shape[0]
	D = np.full((na + 1, nb + 1), np.inf, dtype=np.float32)
	D[0, 0] = 0.0
	for i in range(1, na + 1):
		for j in range(1, nb + 1):
			cost = np.linalg.norm(a[i - 1] - b[j - 1])
			D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
	return float(D[na, nb] / (na + nb))


def enroll_templates(wav_paths: List[str]) -> List[np.ndarray]:
	return [_mfcc(p) for p in wav_paths if os.path.exists(p)]


def score_keyword(templates: List[np.ndarray], wav_path: str) -> Tuple[float, float]:
	"""Returns (score_0_1, raw_distance). Higher score is better.
	Score = 1 / (1 + min_d), min_d is min DTW distance to templates.
	"""
	if not templates:
		return 0.0, float("inf")
	test = _mfcc(wav_path)
	dists = [_dtw_distance(t, test) for t in templates]
	min_d = float(min(dists)) if dists else float("inf")
	score = 1.0 / (1.0 + min_d)
	return score, min_d

import os
from typing import List, Tuple, Literal

import librosa  # type: ignore
import numpy as np  # type: ignore


def _mfcc(path: str, sr: int = 16000, n_mfcc: int = 13, cmvn: bool = True) -> np.ndarray:
	y, s = librosa.load(path, sr=sr)
	mf = librosa.feature.mfcc(y=y, sr=s, n_mfcc=n_mfcc)
	feat = mf.T  # [frames, n_mfcc]
	if cmvn and feat.size > 0:
		mu = feat.mean(axis=0, keepdims=True)
		sigma = feat.std(axis=0, keepdims=True) + 1e-6
		feat = (feat - mu) / sigma
	return feat


def _frame_distance(a: np.ndarray, b: np.ndarray, metric: Literal["cosine", "euclidean"]) -> float:
	if metric == "cosine":
		na = np.linalg.norm(a) + 1e-6
		nb = np.linalg.norm(b) + 1e-6
		return float(1.0 - float(np.dot(a, b) / (na * nb)))
	# euclidean
	return float(np.linalg.norm(a - b))


def _subsequence_dtw(
	test: np.ndarray,
	template: np.ndarray,
	band_frac: float = 0.15,
	metric: Literal["cosine", "euclidean"] = "cosine",
) -> float:
	"""Open-begin subsequence DTW: aligns template within test sequence.
	Returns normalized cost (lower is better).
	"""
	n = int(test.shape[0])
	m = int(template.shape[0])
	if n == 0 or m == 0:
		return float("inf")
	# Precompute band limits relative to scaled time
	band = max(1, int(band_frac * max(n, m)))
	D = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
	D[:, 0] = 0.0  # open begin in test
	for i in range(1, n + 1):
		# allowed j around scaled index
		j_center = int(i * (m / max(1, n)))
		j_lo = max(1, j_center - band)
		j_hi = min(m, j_center + band)
		for j in range(j_lo, j_hi + 1):
			c = _frame_distance(test[i - 1], template[j - 1], metric)
			D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
	# Best alignment that ends at any i with full template consumed
	best = float(np.min(D[1:, m]))
	# Normalize by path length approx (m) to keep scale stable
	return best / float(m)


def enroll_templates(wav_paths: List[str], sr: int = 16000, n_mfcc: int = 13, cmvn: bool = True) -> List[np.ndarray]:
	return [_mfcc(p, sr=sr, n_mfcc=n_mfcc, cmvn=cmvn) for p in wav_paths if os.path.exists(p)]


def score_keyword(
	templates: List[np.ndarray],
	wav_path: str,
	sr: int = 16000,
	n_mfcc: int = 13,
	cmvn: bool = True,
	band_frac: float = 0.15,
	metric: Literal["cosine", "euclidean"] = "cosine",
) -> Tuple[float, float]:
	"""Advanced KWS scoring: CMVN, subsequence DTW with band, multi-template min.
	Returns (score_0_1, raw_distance). Higher score is better.
	"""
	if not templates:
		return 0.0, float("inf")
	test = _mfcc(wav_path, sr=sr, n_mfcc=n_mfcc, cmvn=cmvn)
	if test.size == 0:
		return 0.0, float("inf")
	dists = [_subsequence_dtw(test, t, band_frac=band_frac, metric=metric) for t in templates if t.size > 0]
	if not dists:
		return 0.0, float("inf")
	min_d = float(min(dists))
	# Map distance to score -> [0,1], use 1/(1+d)
	score = 1.0 / (1.0 + min_d)
	return score, min_d

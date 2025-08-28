import re
from typing import List, Tuple

# Basic Arabic normalization
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
_TATWEEL = "\u0640"


def normalize_ar(text: str) -> str:
	if not text:
		return ""
	t = text
	t = _ARABIC_DIACRITICS.sub("", t)
	t = t.replace(_TATWEEL, "")
	t = t.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
	t = t.replace("ى", "ي").replace("ة", "ه")
	t = re.sub(r"\s+", " ", t).strip()
	return t


def levenshtein(a: str, b: str) -> int:
	a, b = a or "", b or ""
	la, lb = len(a), len(b)
	if la == 0:
		return lb
	if lb == 0:
		return la
	dp = list(range(lb + 1))
	for i in range(1, la + 1):
		prev = dp[0]
		dp[0] = i
		for j in range(1, lb + 1):
			tmp = dp[j]
			cost = 0 if a[i - 1] == b[j - 1] else 1
			dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
			prev = tmp
	return dp[lb]


def map_to_commands(text: str, commands: List[str], max_distance: int) -> Tuple[str, int]:
	norm_in = normalize_ar(text)
	best = ("", 10**9)
	for cmd in commands:
		norm_cmd = normalize_ar(cmd)
		d = levenshtein(norm_in, norm_cmd)
		if d < best[1]:
			best = (cmd, d)
	return (best[0] if best[1] <= max_distance else "", best[1])

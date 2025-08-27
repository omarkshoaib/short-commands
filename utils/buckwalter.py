from typing import Dict
import re


# Minimal Buckwalter mapping for common characters seen in Arabic CTC outputs
# Reference: Standard Buckwalter transliteration
_BUCK2AR: Dict[str, str] = {
	"A": "ا",
	"b": "ب",
	"t": "ت",
	"v": "ث",
	"j": "ج",
	"H": "ح",
	"x": "خ",
	"d": "د",
	"*": "ذ",
	"r": "ر",
	"z": "ز",
	"s": "س",
	"$": "ش",
	"S": "ص",
	"D": "ض",
	"T": "ط",
	"Z": "ظ",
	"E": "ع",
	"g": "غ",
	"f": "ف",
	"q": "ق",
	"k": "ك",
	"l": "ل",
	"m": "م",
	"n": "ن",
	"h": "ه",
	"w": "و",
	"y": "ي",
	"Y": "ى",
	"p": "ة",
	">": "أ",
	"<": "إ",
	"|": "آ",
	"&": "ؤ",
	"}": "ئ",
	"'": "ء",
	" ": " ",
}

# Diacritics and markers to drop in plain text outputs
_DIACRITICS = set(list("auiFNKo~`^"))

# Arabic diacritics (Unicode) and Tatweel
_ARABIC_DIACRITICS_PATTERN = re.compile(
	"[\u064B-\u0652\u0670\u0653\u0654\u0655\u0656\u0657\u0658\u0671\u0640]"
)

# Common BiDi control marks to remove
_BIDI_CONTROL_PATTERN = re.compile("[\u200E\u200F\u202A-\u202E\u2066-\u2069]")


def buckwalter_to_arabic(text: str) -> str:
	out = []
	for ch in text:
		if ch in _DIACRITICS:
			continue
		out.append(_BUCK2AR.get(ch, ch))
	return "".join(out)


def normalize_arabic_text(text: str) -> str:
	"""Best-effort normalization for Arabic output.
	- Map stray Buckwalter symbols to Arabic letters
	- Remove Arabic diacritics and Tatweel
	- Strip BiDi control marks
	- Collapse excess whitespace
	"""
	if not text:
		return ""
	# 1) Map common Buckwalter glyphs if present
	mapped = buckwalter_to_arabic(text)
	# 2) Remove Arabic diacritics and Tatweel
	mapped = _ARABIC_DIACRITICS_PATTERN.sub("", mapped)
	# 3) Remove bidi control characters
	mapped = _BIDI_CONTROL_PATTERN.sub("", mapped)
	# 4) Normalize spaces
	mapped = re.sub(r"\s+", " ", mapped).strip()
	return mapped

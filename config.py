import os

try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass

SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
RECORDINGS_DIR: str = os.getenv("RECORDINGS_DIR", "recordings")
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")
RESULTS_CSV: str = os.path.join(RESULTS_DIR, os.getenv("RESULTS_CSV", "transcriptions.csv"))

# Device control (cpu|cuda). Default to CPU to avoid cuDNN issues.
STT_DEVICE: str = os.getenv("STT_DEVICE", "cpu").lower()

# Wav2Vec2 Arabic model id (HF)
W2V2_MODEL_ID: str = os.getenv("W2V2_MODEL_ID", "elgeish/wav2vec2-large-xlsr-53-arabic")
# Whisper model (faster-whisper) - e.g., "large-v3-turbo" or "large-v3"
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
# Whisper compute type. For CPU, int8 is efficient.
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Vosk model path (unzipped directory)
VOSK_MODEL_PATH: str = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-ar-0.22-linto-1.1.0")
# Vosk grammar (comma-separated Arabic phrases); improves KWS/short commands
VOSK_GRAMMAR: str = os.getenv(
	"VOSK_GRAMMAR",
	"افتح, اغلق, تشغيل, ايقاف, نعم, لا, امسح, ايميل, الايميل, إيميل, البريد, الرسالة",
)
# If an utterance is shorter than this, pad with silence for Vosk only (ms)
MIN_VOSK_MS: int = int(os.getenv("MIN_VOSK_MS", "1500"))
PAD_SHORT_FOR_VOSK: bool = os.getenv("PAD_SHORT_FOR_VOSK", "1") not in ("0", "false", "False")

CSV_COLUMNS = [
	"timestamp",
	"audio_file",
	"audio_duration_ms",
	"wav2vec2",
	"whisper_turbo",
	"vosk",
	"wav2vec2_time_ms",
	"whisper_time_ms",
	"vosk_time_ms",
	"total_processing_time_ms",
]


def ensure_dirs() -> None:
	os.makedirs(RECORDINGS_DIR, exist_ok=True)
	os.makedirs(RESULTS_DIR, exist_ok=True)

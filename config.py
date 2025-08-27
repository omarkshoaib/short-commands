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

CSV_COLUMNS = [
	"timestamp",
	"audio_file",
	"audio_duration_ms",
	"wav2vec2",
	"whisper_turbo",
]


def ensure_dirs() -> None:
	os.makedirs(RECORDINGS_DIR, exist_ok=True)
	os.makedirs(RESULTS_DIR, exist_ok=True)

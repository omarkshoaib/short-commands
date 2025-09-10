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
TELEMETRY_PATH: str = os.path.join(RESULTS_DIR, os.getenv("TELEMETRY_PATH", "telemetry.jsonl"))

# Device control (cpu|cuda). Default to CPU to avoid cuDNN issues.
STT_DEVICE: str = os.getenv("STT_DEVICE", "cpu").lower()

# Wav2Vec2 Arabic model id (HF)
W2V2_MODEL_ID: str = os.getenv("W2V2_MODEL_ID", "elgeish/wav2vec2-large-xlsr-53-arabic")
# Whisper model (faster-whisper) - e.g., "large-v3-turbo" or "large-v3"
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
# Whisper compute type. For CPU, int8 is efficient.
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# VAD + ring buffer
VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0-3
VAD_FRAME_MS: int = int(os.getenv("VAD_FRAME_MS", "30"))  # 10/20/30ms
RING_BUFFER_MS: int = int(os.getenv("RING_BUFFER_MS", "800"))  # pre-roll
MIN_SPEECH_MS: int = int(os.getenv("MIN_SPEECH_MS", "500"))  # gate too short utterances

# Whisper fallback policy
WHISPER_FALLBACK_LONG_MS: int = int(os.getenv("WHISPER_FALLBACK_LONG_MS", "6000"))
W2V2_CONF_THRESHOLD: float = float(os.getenv("W2V2_CONF_THRESHOLD", "0.60"))
W2V2_ENTROPY_MAX: float = float(os.getenv("W2V2_ENTROPY_MAX", "2.20"))  # higher -> uncertain
SNR_LOW_DB: float = float(os.getenv("SNR_LOW_DB", "10"))  # if below, be stricter
ALWAYS_RUN_WHISPER: bool = os.getenv("ALWAYS_RUN_WHISPER", "0") in ("1", "true", "True")

## KWS removed

# Command grammar (Arabic phrases to snap to)
COMMANDS: list[str] = [
	"امسح الإيميل",
	"افتح",
	"اغلق",
	"تشغيل",
	"ايقاف",
	"نعم",
	"لا",
]
CMD_MAP_MAX_DISTANCE: int = int(os.getenv("CMD_MAP_MAX_DISTANCE", "4"))

CSV_COLUMNS = [
	"timestamp",
	"audio_file",
	"audio_duration_ms",
	"mfcc_label",
	"mfcc_confidence",
	"mfcc_used",
	"wav2vec2",
	"w2v2_confidence",
	"wav2vec2_mapped",
	"whisper_turbo",
	"whisper_used",
	"whisper_mapped",
	"wav2vec2_time_ms",
	"whisper_time_ms",
	"total_processing_time_ms",
]


def ensure_dirs() -> None:
	os.makedirs(RECORDINGS_DIR, exist_ok=True)
	os.makedirs(RESULTS_DIR, exist_ok=True)
    # KWS templates dir no longer used

# MFCC CNN runtime (optional front-end classifier)
MFCC_CNN_ENABLED: bool = os.getenv("MFCC_CNN_ENABLED", "1") in ("1", "true", "True")
MFCC_CNN_GATE_STT: bool = os.getenv("MFCC_CNN_GATE_STT", "1") in ("1", "true", "True")
MFCC_CNN_CONF_THRESHOLD: float = float(os.getenv("MFCC_CNN_CONF_THRESHOLD", "0.75"))
MFCC_CNN_CKPT_PATH: str = os.getenv("MFCC_CNN_CKPT_PATH", os.path.join("checkpoints", "mfcc_cnn", "best.pt"))
MFCC_FEATURE_TYPE: str = os.getenv("MFCC_FEATURE_TYPE", "logmel")
MFCC_NUM_MELS: int = int(os.getenv("MFCC_NUM_MELS", "64"))
MFCC_NUM_MFCC: int = int(os.getenv("MFCC_NUM_MFCC", "40"))
MFCC_WIN_MS: int = int(os.getenv("MFCC_WIN_MS", "25"))
MFCC_HOP_MS: int = int(os.getenv("MFCC_HOP_MS", "10"))
MFCC_FMIN: int = int(os.getenv("MFCC_FMIN", "20"))
MFCC_FMAX: int = int(os.getenv("MFCC_FMAX", "7600"))
MFCC_CMVN: bool = os.getenv("MFCC_CMVN", "1") in ("1", "true", "True")
MFCC_DELTAS: bool = os.getenv("MFCC_DELTAS", "0") in ("1", "true", "True")
MFCC_CNN_REQUIRE_MAPPING: bool = os.getenv("MFCC_CNN_REQUIRE_MAPPING", "1") in ("1", "true", "True")

# Map MFCC-CNN English labels to our Arabic COMMANDS (accept only if mapped)
# Extend as needed if you train on more classes.
MFCC_LABEL_TO_COMMAND: dict[str, str] = {
	"open": "افتح",
	"close": "اغلق",
	"yes": "نعم",
	"no": "لا",
	"start": "تشغيل",
	"stop": "ايقاف",
}

# CNN sliding-window inference (milliseconds)
MFCC_SLIDE_WINDOW_MS: int = int(os.getenv("MFCC_SLIDE_WINDOW_MS", "1000"))
MFCC_SLIDE_HOP_MS: int = int(os.getenv("MFCC_SLIDE_HOP_MS", "500"))

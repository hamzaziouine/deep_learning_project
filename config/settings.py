"""Project-wide path constants. Read by tools and agents."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_SAMPLE = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_roberta"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Gemini model — env-overridable so demos can swap variants when a quota cap
# is hit (e.g. gemini-2.5-flash exhausts at 20/day on free tier, but
# gemini-flash-latest has a separate quota counter). NOT gemini-2.0-flash —
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

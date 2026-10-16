import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _configured_path(name: str, default: Path) -> Path:
    configured = os.getenv(name)
    if not configured:
        return default
    path = Path(configured)
    return path if path.is_absolute() else BASE_DIR / path


DATA_DIR = _configured_path("DATA_DIR", BASE_DIR / "data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = _configured_path("MODEL_DIR", BASE_DIR / "models")
REPORT_DIR = _configured_path("REPORT_DIR", BASE_DIR / "reports")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
FALSE_POSITIVE_COST = float(os.getenv("FALSE_POSITIVE_COST", "1.0"))
FALSE_NEGATIVE_COST = float(os.getenv("FALSE_NEGATIVE_COST", "3.0"))

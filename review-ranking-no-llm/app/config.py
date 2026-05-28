import os
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()

# HuggingFace auth — reads HF_ACCESS_TOKEN from .env and exposes it as HF_TOKEN,
# which the huggingface_hub library picks up automatically to suppress the warning.
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
if HF_ACCESS_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_ACCESS_TOKEN)

DB_USER = os.getenv("DB_USER")
DB_PORT = os.getenv("DB_PORT")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")

# Blend deterministic text-side signal vs OpenCV image quality when both exist.
TEXT_BLEND_WEIGHT = float(os.getenv("NO_LLM_TEXT_BLEND_WEIGHT", "0.40"))
IMAGE_BLEND_WEIGHT = float(os.getenv("NO_LLM_IMAGE_BLEND_WEIGHT", "0.60"))

# Weights inside the text-side signal (no embedding similarity).
WEIGHTS = {
    "text_length": 0.40,
    "rating": 0.35,
    "recency": 0.25,
}

IMAGE_QUALITY_MAX_IMAGES = int(os.getenv("IMAGE_QUALITY_MAX_IMAGES", "15"))
IMAGE_QUALITY_REQUEST_TIMEOUT_S = float(os.getenv("IMAGE_QUALITY_REQUEST_TIMEOUT_S", "8"))

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "30"))

# How aggressively to reward human presence in review images (0.0–1.0).
# Uses interpolation: score = raw_clip + prominence × weight × (1 − raw_clip)
# 0.0 → human presence ignored; 1.0 → full-body shot always scores 1.0.
# 0.70 balances product-similarity signal with rewarding real-usage photos.
CLIP_HUMAN_WEIGHT = float(os.getenv("CLIP_HUMAN_WEIGHT", "0.70"))

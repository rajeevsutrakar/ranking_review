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

# Blend deterministic text-side signal vs image score in the final review ranking.
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

# ── CLIP scoring ──────────────────────────────────────────────────────────────

# How aggressively wearing/usage context boosts the adjusted CLIP score.
# Interpolation: adjusted = raw_clip + boost_factor × weight × (1 − raw_clip)
# 0.0 → human presence ignored; 0.70 is a strong but bounded boost.
CLIP_HUMAN_WEIGHT = float(os.getenv("CLIP_HUMAN_WEIGHT", "0.70"))

# Score multiplier for images classified as packaging/delivery materials.
# Packaging (courier bags, boxes) may share brand CLIP semantics with the product
# but gives customers zero purchase signal → penalise heavily.
# 0.20 → packaging CLIP score reduced to 20% of raw value before blending.
PACKAGING_SCORE_PENALTY = float(os.getenv("PACKAGING_SCORE_PENALTY", "0.20"))

# ── Image similarity blend ────────────────────────────────────────────────────

# CLIP product-similarity is the primary signal; OpenCV quality is a modifier.
# 65 / 35 ensures product relevance dominates while still rewarding clear photos.
CLIP_WEIGHT_IN_BLEND   = float(os.getenv("CLIP_WEIGHT_IN_BLEND",   "0.65"))
OPENCV_WEIGHT_IN_BLEND = float(os.getenv("OPENCV_WEIGHT_IN_BLEND", "0.35"))

"""
CLIP similarity client — openai/clip-vit-base-patch32.

pipeline_shopify_product has exactly ONE image per product.
Reference building:
  1. Try media_url → embed as image (image-to-image).
  2. If no media_url → embed title as "a product photo of <title>" (text-to-image).
  3. If neither → return None.

All embeddings are L2-normalised → cosine similarity = np.dot(a, b).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests

import app.config  # ensures HF_TOKEN is set from .env before any HF Hub call
from app.models.clip_score_cache import CLIPScoreCache

if TYPE_CHECKING:
    from PIL import Image as PILImage

_MODEL_NAME = "openai/clip-vit-base-patch32"
_REQUEST_TIMEOUT_S = 10

_clip_model = None
_clip_processor = None

_DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "clip_score_cache.json"

# Three-tier human prominence prompts — embedded once, reused for every image.
# Tier 1 (full):    clear full-body shot → full CLIP_HUMAN_WEIGHT interpolation
# Tier 2 (partial): selfie / mirror shot / seated / upper-body visible → half interpolation
# Tier 3 (none):    packaging / fabric / no person → raw CLIP score only
_FULL_BODY_PROMPT = "a person wearing the complete outfit from head to toe clearly visible"
_HUMAN_PROMPT     = "a selfie or photo of a person wearing or showing the product"
_NO_HUMAN_PROMPT  = "a product photo, packaging, or fabric with no person"
_full_body_emb: np.ndarray | None = None
_human_emb:     np.ndarray | None = None
_no_human_emb:  np.ndarray | None = None


# ── model ─────────────────────────────────────────────────────────────────────

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP requires 'transformers', 'torch', and 'Pillow'. "
                "Install with: pip install transformers torch Pillow"
            ) from exc
        _clip_processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(_MODEL_NAME)
        _clip_model.eval()
    return _clip_model, _clip_processor


# ── image fetch + embed ───────────────────────────────────────────────────────

def _fetch_pil(url: str) -> PILImage.Image | None:
    try:
        from PIL import Image
        r = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT_S,
            headers={"User-Agent": "ProductReviewRankerCLIP/1.0"},
        )
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def embed_image_from_url(url: str) -> np.ndarray | None:
    """
    L2-normalised CLIP image embedding [512], or None on failure.

    Uses vision_model → visual_projection directly instead of get_image_features()
    because transformers 5.x changed that method to return BaseModelOutputWithPooling
    instead of a plain tensor.
    """
    import torch
    pil_img = _fetch_pil(url)
    if pil_img is None:
        return None
    model, processor = _get_clip()
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        vis_out = model.vision_model(
            pixel_values=inputs["pixel_values"], return_dict=True
        )
        # pooler_output = CLS token, shape (1, 768)
        features = model.visual_projection(vis_out.pooler_output)  # (1, 512)
    return _l2_normalize(features[0].cpu().numpy().astype(np.float32))


def embed_text(prompt: str) -> np.ndarray:
    """
    L2-normalised CLIP text embedding [512].

    Uses text_model → text_projection directly for the same version-safety reason.
    """
    import torch
    model, processor = _get_clip()
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        # pooler_output = EOS token, shape (1, 512)
        features = model.text_projection(txt_out.pooler_output)  # (1, 512)
    return _l2_normalize(features[0].cpu().numpy().astype(np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ── reference embedding (single product image) ────────────────────────────────

def build_reference_embedding(
    media_url: str | None,
    title: str | None,
) -> tuple[np.ndarray | None, str]:
    """
    Build the reference embedding for one product.
    pipeline_shopify_product has exactly one row per product.

    Returns (embedding, mode) where mode = "image" | "text" | "none".
    """
    if media_url:
        emb = embed_image_from_url(media_url)
        if emb is not None:
            return emb, "image"

    if title:
        # Action-oriented prompt: better CLIP discrimination for review images
        # (people wearing/using the product) than a plain "product photo" description.
        return embed_text(f"a person wearing or using {title}"), "text"

    return None, "none"


# ── human presence detection ─────────────────────────────────────────────────

def _get_human_detection_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (full_body_emb, human_emb, no_human_emb) — computed once, cached in module globals.
    Cost: 3 CLIP text forward passes on first call, then zero.
    """
    global _full_body_emb, _human_emb, _no_human_emb
    if _human_emb is None:
        _full_body_emb = embed_text(_FULL_BODY_PROMPT)
        _human_emb     = embed_text(_HUMAN_PROMPT)
        _no_human_emb  = embed_text(_NO_HUMAN_PROMPT)
    return _full_body_emb, _human_emb, _no_human_emb


def human_prominence_score(image_emb: np.ndarray) -> float:
    """
    Three-tier CLIP classification of how prominently a person+outfit appears.

    Returns:
        1.0 — full body clearly visible (clear outfit shot, person dominant in frame)
        0.5 — partial visibility (mirror selfie, seated, person in background)
        0.0 — no human or person too distant/tiny to show the product

    Thresholds (0.02 / 0.01) are margins over the no-human baseline required to
    qualify for each tier. CLIP cosine differences for these prompts typically
    fall in the 0.01–0.05 range, so these are deliberately conservative.
    """
    full_body_emb, human_emb, no_human_emb = _get_human_detection_embeddings()
    full_sim     = cosine_similarity(image_emb, full_body_emb)
    partial_sim  = cosine_similarity(image_emb, human_emb)
    no_human_sim = cosine_similarity(image_emb, no_human_emb)

    if full_sim > no_human_sim + 0.02:
        return 1.0
    if partial_sim > no_human_sim:
        return 0.5
    return 0.0


# ── score review images + update cache ────────────────────────────────────────

def compute_clip_scores(
    product_id: int,
    review_image_urls: list[str],
    reference_embedding: np.ndarray,
    cache: CLIPScoreCache,
) -> dict[str, float]:
    """
    Compute CLIP cosine similarity for each review image vs. the reference.

    Human prominence interpolation:
        score = raw_clip + prominence × CLIP_HUMAN_WEIGHT × (1 − raw_clip)

    This pulls the score toward 1.0 proportionally — full-body shots of someone
    clearly wearing the product rank highest, while product similarity still
    differentiates images at the same prominence tier. Never exceeds 1.0.

    Updates running averages in cache (does NOT call cache.save — caller saves).
    Returns {image_url: updated_avg_clip_score}.
    """
    from app.config import CLIP_HUMAN_WEIGHT

    raw: dict[str, float] = {}
    for url in review_image_urls:
        emb = embed_image_from_url(url)
        if emb is not None:
            score = cosine_similarity(emb, reference_embedding)
            if CLIP_HUMAN_WEIGHT > 0:
                prominence = human_prominence_score(emb)
                if prominence > 0:
                    score = score + prominence * CLIP_HUMAN_WEIGHT * (1.0 - score)
            raw[url] = score
    return cache.update_batch(product_id, raw)


def get_or_load_cache(cache_path: Path | None = None) -> CLIPScoreCache:
    return CLIPScoreCache(cache_path or _DEFAULT_CACHE_PATH)

"""
CLIP similarity client — openai/clip-vit-base-patch32.

Scoring pipeline per review image
──────────────────────────────────
1. Embed review image (vision encoder).
2. Compute raw product similarity: cosine(review_emb, product_reference_emb).
3. Classify image into one of four tiers via zero-shot CLIP text prompts:
     full_worn  → person clearly wearing the complete product (boost 1.0)
     partial    → selfie, mirror shot, upper-body, lifestyle photo  (boost 0.6)
     display    → flat-lay, hanger, mannequin (boost 0.15)
     packaging  → courier bag, delivery box → penalty
4. Apply adjustment:
     packaging  → score × PACKAGING_SCORE_PENALTY
     otherwise  → score + boost_factor × CLIP_HUMAN_WEIGHT × (1 − score)

No caching — the main pipeline filters score IS NULL so each review image
is processed exactly once. No running-average state to go stale.

Product reference priority (build_reference_embedding):
  1. media_url  → image-to-image (most accurate)
  2. title only → "a person wearing or using <title>" (text-to-image, 0.75× scaled)
  3. neither    → CLIP skipped, opencv score only
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import requests

import app.config  # ensures HF_TOKEN is set from .env before any HF Hub call

if TYPE_CHECKING:
    from PIL import Image as PILImage

_MODEL_NAME = "openai/clip-vit-base-patch32"
_REQUEST_TIMEOUT_S = 10

_clip_model = None
_clip_processor = None

# ── Four-tier image classification prompts ────────────────────────────────────
#
# Tier 1 — full_worn (boost 1.0):
#   Real photo, person wearing the complete product, full body visible.
# Tier 2 — partial (boost 0.6):
#   Real photo, person wearing/using product — selfie, mirror, upper-body, lifestyle.
# Tier 3 — display (boost 0.15):
#   The product itself clearly shown (flat-lay, hanger) but no person.
# Tier 4 — packaging (penalty):
#   Specifically physical delivery packaging (poly mailer, cardboard box).
#   Prompt avoids broad terms like "printed illustration" which can match
#   decorative backgrounds in genuine worn-product photos.
#
_WORN_FULL_PROMPT    = "a real photograph of a person wearing a complete outfit, full body visible from head to toe"
_WORN_PARTIAL_PROMPT = "a real photo of a person wearing clothing — selfie, mirror shot, upper body, or casual lifestyle photo"
_PRODUCT_DISPLAY_PROMPT = "a garment or clothing item displayed as a product photo — flat lay, on a hanger, or on a mannequin"
_PACKAGING_PROMPT    = "a flat plastic poly mailer bag or cardboard shipping box used for product delivery and packaging"

_worn_full_emb:    np.ndarray | None = None
_worn_partial_emb: np.ndarray | None = None
_product_disp_emb: np.ndarray | None = None
_packaging_emb:    np.ndarray | None = None


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
    from PIL import Image, UnidentifiedImageError
    from app.services.run_logger import log_warn
    fname = url.rsplit("/", 1)[-1].split("?")[0]
    try:
        r = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT_S,
            headers={"User-Agent": "ProductReviewRankerCLIP/1.0"},
        )
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        log_warn(f"[clip] HTTP {status} fetching image: {fname}  url={url}")
    except requests.exceptions.Timeout:
        log_warn(f"[clip] Timeout fetching image: {fname}  url={url}")
    except requests.exceptions.ConnectionError:
        log_warn(f"[clip] Connection error fetching image: {fname}  url={url}")
    except UnidentifiedImageError:
        log_warn(f"[clip] PIL cannot identify image format: {fname}  url={url}")
    except Exception as exc:
        log_warn(f"[clip] Unexpected fetch/decode error ({type(exc).__name__}): {fname}  url={url}")
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
        features = model.visual_projection(vis_out.pooler_output)  # (1, 512)
    return _l2_normalize(features[0].cpu().numpy().astype(np.float32))


def embed_text(prompt: str) -> np.ndarray:
    """
    L2-normalised CLIP text embedding [512].

    Uses text_model → text_projection directly for version-safety.
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
    Returns (embedding, mode) where mode = "image" | "text" | "none".
    """
    if media_url and not media_url.startswith(("http://", "https://")):
        from app.services.run_logger import log_warn
        log_warn(f"[clip] Invalid media_url (not an HTTP URL) — value={media_url!r}. Skipping image reference.")

    if media_url and media_url.startswith(("http://", "https://")):
        emb = embed_image_from_url(media_url)
        if emb is not None:
            return emb, "image"

    if title:
        # Action-oriented prompt: better CLIP discrimination for review images
        # (people wearing/using the product) than a plain product description.
        return embed_text(f"a person wearing or using {title}"), "text"

    return None, "none"


# ── four-tier image classification ───────────────────────────────────────────

def _get_classification_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (worn_full_emb, worn_partial_emb, product_disp_emb, packaging_emb).
    Computed once on first call, cached in module globals.
    """
    global _worn_full_emb, _worn_partial_emb, _product_disp_emb, _packaging_emb
    if _worn_full_emb is None:
        _worn_full_emb    = embed_text(_WORN_FULL_PROMPT)
        _worn_partial_emb = embed_text(_WORN_PARTIAL_PROMPT)
        _product_disp_emb = embed_text(_PRODUCT_DISPLAY_PROMPT)
        _packaging_emb    = embed_text(_PACKAGING_PROMPT)
    return _worn_full_emb, _worn_partial_emb, _product_disp_emb, _packaging_emb


def classify_image_type(image_emb: np.ndarray) -> tuple[float, str]:
    """
    Classify a review image into one of four tiers using zero-shot CLIP.

    Returns (boost_factor, tier_name):
        (1.0,  "full_worn")  — full-body worn photo
        (0.6,  "partial")    — selfie / mirror / lifestyle
        (0.15, "display")    — product displayed, no person
        (0.0,  "packaging")  — delivery bag / box

    Packaging requires pkg_sim > best_positive + 0.03 to prevent false positives
    from decorative walls, patterned backgrounds, or busy indoor scenes.
    """
    full_emb, partial_emb, disp_emb, pkg_emb = _get_classification_embeddings()

    full_sim    = cosine_similarity(image_emb, full_emb)
    partial_sim = cosine_similarity(image_emb, partial_emb)
    disp_sim    = cosine_similarity(image_emb, disp_emb)
    pkg_sim     = cosine_similarity(image_emb, pkg_emb)

    best_positive = max(full_sim, partial_sim, disp_sim)

    if pkg_sim > best_positive + 0.03:
        return 0.0, "packaging"

    if full_sim > disp_sim + 0.01 and full_sim > pkg_sim + 0.01:
        return 1.0, "full_worn"

    if partial_sim > disp_sim and partial_sim > pkg_sim:
        return 0.6, "partial"

    return 0.15, "display"


# ── score review images ────────────────────────────────────────────────────────

def compute_clip_scores(
    product_id: int,
    review_image_urls: list[str],
    reference_embedding: np.ndarray,
) -> dict[str, float]:
    """
    Compute adjusted CLIP score for each review image vs. the product reference.

    No caching — each review image is processed exactly once because the pipeline
    filters score IS NULL in the DB query. Running averages would only accumulate
    stale values when the scoring formula changes.

    Scoring per image:
      1. raw_clip  = cosine_similarity(review_emb, reference_emb)
      2. tier      = classify_image_type(review_emb)
      3a. packaging  → adjusted = raw_clip × PACKAGING_SCORE_PENALTY
      3b. otherwise  → adjusted = raw_clip + boost × CLIP_HUMAN_WEIGHT × (1 − raw_clip)

    A log line is written per image so tier assignments can be inspected
    in runtime_no_llm.log when debugging unexpected scores.

    Returns {image_url: adjusted_clip_score}.
    """
    from app.config import CLIP_HUMAN_WEIGHT, PACKAGING_SCORE_PENALTY
    from app.services.run_logger import log_info

    scores: dict[str, float] = {}
    for url in review_image_urls:
        emb = embed_image_from_url(url)
        if emb is None:
            continue

        raw_clip = cosine_similarity(emb, reference_embedding)
        boost_factor, tier = classify_image_type(emb)

        if tier == "packaging":
            score = raw_clip * PACKAGING_SCORE_PENALTY
        elif boost_factor > 0:
            score = raw_clip + boost_factor * CLIP_HUMAN_WEIGHT * (1.0 - raw_clip)
        else:
            score = raw_clip
        score = min(0.99, max(0.0, score))

        fname = url.rsplit("/", 1)[-1].split("?")[0]
        log_info(
            f"[clip] pid={product_id} tier={tier:10s} "
            f"raw={raw_clip:.3f} adjusted={score:.3f}  {fname}"
        )
        scores[url] = score

    return scores


def compute_clip_scores_text_ref(
    product_id: int,
    review_image_urls: list[str],
    title: str,
) -> dict[str, float]:
    """
    Score review images when no product reference image is available — only a title.

    Problem with image-reference approach applied to text:
      additive formula (raw + boost × HW × (1-raw)) inflates ANY human photo
      regardless of product relevance, because the text embedding mixes
      "person wearing" + "{title}" semantics together.

    Solution — multiplicative formula with a product-only prompt:
      1. product_sim = cosine(image_emb, embed_text("a photo of {title}"))
         → pure product relevance, no person bias baked in
      2. tier = classify_image_type(image_emb)
         → independent human-detection signal
      3. score = product_sim × (1.0 + boost × CLIP_HUMAN_WEIGHT)   [multiplicative]
         → a generic person photo (low product_sim) stays low even with full boost
         → a relevant worn image (higher product_sim) gets a meaningful lift

    Scores are naturally lower than image-reference mode (text-to-image CLIP
    cosine similarities are ~0.15–0.35), which correctly reflects lower confidence.
    """
    from app.config import CLIP_HUMAN_WEIGHT, PACKAGING_SCORE_PENALTY
    from app.services.run_logger import log_info

    product_emb = embed_text(f"a photo of {title}")

    scores: dict[str, float] = {}
    for url in review_image_urls:
        emb = embed_image_from_url(url)
        if emb is None:
            continue

        product_sim = cosine_similarity(emb, product_emb)
        boost_factor, tier = classify_image_type(emb)

        if tier == "packaging":
            score = product_sim * PACKAGING_SCORE_PENALTY
        else:
            score = product_sim * (1.0 + boost_factor * CLIP_HUMAN_WEIGHT)
        score = min(0.99, max(0.0, score))

        fname = url.rsplit("/", 1)[-1].split("?")[0]
        log_info(
            f"[clip-text] pid={product_id} tier={tier:10s} "
            f"prod_sim={product_sim:.3f} adjusted={score:.3f}  {fname}"
        )
        scores[url] = score

    return scores

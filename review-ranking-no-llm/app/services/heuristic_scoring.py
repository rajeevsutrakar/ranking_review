"""
Deterministic review score: OpenCV image quality + rating + recency + text length.
No embeddings, no LLMs. Aligns with main app scoring logic but replaces similarity
with a capped normalized text-length signal.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

from app.config import (
    IMAGE_BLEND_WEIGHT,
    RECENCY_HALF_LIFE_DAYS,
    TEXT_BLEND_WEIGHT,
    WEIGHTS,
)
from app.services.image_quality import review_has_image_url


def _has_meaningful_text(review: Any) -> bool:
    return bool((getattr(review, "text", None) or "").strip())


def _normalized_weights() -> dict[str, float]:
    total = sum(WEIGHTS.values())
    if total <= 0:
        raise ValueError("WEIGHTS sum must be greater than zero.")
    return {key: value / total for key, value in WEIGHTS.items()}


def _text_length_norm(review: Any) -> float:
    if not _has_meaningful_text(review):
        return 0.0
    return min(1.0, len(str(review.text).strip()) / 800.0)


def compute_recency_factor(review: Any) -> float:
    age_seconds = max(0, time.time() - float(review.timestamp))
    age_days = age_seconds / 86400
    return math.exp(-math.log(2) * (age_days / RECENCY_HALF_LIFE_DAYS))


def compute_text_side_signal(review: Any) -> float:
    """~[0, 1] when review has non-empty text; else 0."""
    if not _has_meaningful_text(review):
        return 0.0
    w = _normalized_weights()
    r_norm = min(1.0, max(0.0, int(getattr(review, "rating", 0) or 0) / 5.0))
    return (
        w["text_length"] * _text_length_norm(review)
        + w["rating"] * r_norm
        + w["recency"] * compute_recency_factor(review)
    )


def rating_recency_only(review: Any) -> float:
    """For rows with no usable text: still rank by stars and recency."""
    r_norm = min(1.0, max(0.0, int(getattr(review, "rating", 0) or 0) / 5.0))
    return 0.35 * r_norm + 0.65 * compute_recency_factor(review)


def compute_heuristic_product_score(review: Any) -> float:
    """
    Single scalar for ordering. Image-URL reviews use OpenCV + text-side blend when
    decode succeeds; otherwise fall back to text/rating/recency while staying in the
    image-first bucket (handled in sort key, not here).
    """
    text_signal = compute_text_side_signal(review)
    image_score = float(getattr(review, "image_quality_score", 0.0) or 0.0)
    has_url = review_has_image_url(review)

    if not has_url:
        if _has_meaningful_text(review):
            return min(1.0, text_signal)
        return min(1.0, rating_recency_only(review))

    if image_score <= 0 and not _has_meaningful_text(review):
        return min(1.0, rating_recency_only(review))

    if image_score <= 0:
        return min(1.0, text_signal + 0.03)

    if not _has_meaningful_text(review):
        return min(1.0, image_score)

    blend_total = TEXT_BLEND_WEIGHT + IMAGE_BLEND_WEIGHT
    if blend_total <= 0:
        return min(1.0, text_signal)
    w_text = TEXT_BLEND_WEIGHT / blend_total
    w_img = IMAGE_BLEND_WEIGHT / blend_total
    return min(1.0, w_text * text_signal + w_img * image_score)


def sort_reviews_deterministic(reviews: Sequence[Any]) -> list[Any]:
    """
    Image URL first, then text-only. Within each bucket: higher heuristic score,
    then higher star rating, then more recent.
    """
    def key(r: Any) -> tuple:
        bucket = 0 if review_has_image_url(r) else 1
        score = compute_heuristic_product_score(r)
        rating = int(getattr(r, "rating", 0) or 0)
        rec = compute_recency_factor(r)
        rid = int(getattr(r, "review_id", 0) or 0)
        return (bucket, -score, -rating, -rec, rid)

    return sorted(reviews, key=key)


def assign_product_scores(reviews: Sequence[Any]) -> None:
    for r in reviews:
        r.score = compute_heuristic_product_score(r)

"""
OpenCV-based image quality: blur (Laplacian variance), brightness, resolution.
Per-image score in [0, 1]. Aggregate = mean of successfully scored images.
0 URLs / 0 images → 0. No successful decode → 0.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import requests

from app.config import IMAGE_QUALITY_MAX_IMAGES, IMAGE_QUALITY_REQUEST_TIMEOUT_S

_cv2_module = None


def _get_cv2():
    global _cv2_module
    if _cv2_module is None:
        import cv2

        _cv2_module = cv2
    return _cv2_module


def _normalize_blur(laplacian_var: float) -> float:
    # Real photos (even very sharp ones): Laplacian variance 50–600.
    # Pure noise / corrupted images: 3,000–50,000+.
    # Values above 1200 are almost certainly noise, not genuine sharpness.
    if laplacian_var > 1200.0:
        return 0.0
    return float(min(1.0, max(0.0, laplacian_var / 350.0)))


def _normalize_brightness(gray: np.ndarray) -> float:
    """Brightness score: penalize very dark/very bright images."""
    mean_b = float(np.mean(gray)) / 255.0
    # Ideal brightness range: 0.3-0.7 (not too dark, not too bright)
    if 0.3 <= mean_b <= 0.7:
        return 1.0
    # Outside range: score decreases linearly
    if mean_b < 0.3:
        return max(0.0, mean_b / 0.3)  # 0 at black, 1.0 at 0.3
    else:  # mean_b > 0.7
        return max(0.0, (1.0 - mean_b) / 0.3)  # 1.0 at 0.7, 0 at 1.0


def _normalize_resolution(height: int, width: int) -> float:
    pixels = height * width
    ref = 640 * 640
    return float(min(1.0, pixels / ref))


def _decode_image_bytes(data: bytes) -> np.ndarray | None:
    cv2 = _get_cv2()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _score_single_bgr(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    cv2 = _get_cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    h, w = gray.shape[:2]
    blur_n = _normalize_blur(blur_var)
    bright_n = _normalize_brightness(gray)
    res_n = _normalize_resolution(h, w)
    return float(0.42 * blur_n + 0.28 * bright_n + 0.30 * res_n)


def _extract_urls(images: Any) -> List[str]:
    if images is None:
        return []
    if isinstance(images, str):
        s = images.strip()
        return [s] if s else []
    if isinstance(images, dict):
        for key in ("url", "src", "image", "image_url"):
            v = images.get(key)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
        return []
    if not isinstance(images, Sequence):
        return []
    urls: List[str] = []
    for item in images:
        if isinstance(item, str) and item.strip():
            urls.append(item.strip())
        elif isinstance(item, dict):
            for key in ("url", "src", "image", "image_url"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    urls.append(v.strip())
                    break
    return urls


def first_image_url(images: Any) -> str | None:
    """First usable image URL for vision APIs, or None."""
    urls = _extract_urls(images)
    return urls[0] if urls else None


def _fetch_image_bytes(url: str) -> bytes | None:
    try:
        r = requests.get(
            url,
            timeout=IMAGE_QUALITY_REQUEST_TIMEOUT_S,
            headers={"User-Agent": "ProductReviewRankerNoLLM/1.0"},
        )
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _score_one_url(url: str) -> float | None:
    raw = _fetch_image_bytes(url)
    if not raw:
        return None
    img = _decode_image_bytes(raw)
    if img is None:
        return None
    return _score_single_bgr(img)


def compute_image_quality_score(images: Any, max_images: int | None = None) -> float:
    """
    0 images → 0.0.
    1..N images → mean OpenCV quality over each successfully loaded image (failed URLs skipped).
    Capped at max_images for safety (default from config).
    """
    cap = max_images if max_images is not None else IMAGE_QUALITY_MAX_IMAGES
    cap = max(0, int(cap))

    urls = _extract_urls(images)[:cap]
    if not urls:
        return 0.0

    scores: List[float] = []
    for url in urls:
        s = _score_one_url(url)
        if s is not None:
            scores.append(float(s))

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def compute_image_quality_scores_per_url(images: Any, max_images: int | None = None) -> dict[str, float]:
    """
    Return individual OpenCV scores for each image URL.
    Format: {image_url: quality_score}
    Failed URLs are skipped (not included in result).
    """
    cap = max_images if max_images is not None else IMAGE_QUALITY_MAX_IMAGES
    cap = max(0, int(cap))

    urls = _extract_urls(images)[:cap]
    if not urls:
        return {}

    url_scores: dict[str, float] = {}
    for url in urls:
        s = _score_one_url(url)
        if s is not None:
            url_scores[url] = float(s)

    return url_scores


def set_image_quality_scores(reviews: Sequence[Any]) -> None:
    for review in reviews:
        images = getattr(review, "images", None) or []
        score = compute_image_quality_score(images)
        review.image_quality_score = score


def review_has_image_url(review: Any) -> bool:
    """True if this review has at least one image URL (same notion as the main app)."""
    return bool(first_image_url(getattr(review, "images", None)))


def set_clip_similarity_scores(
    reviews: Sequence[Any],
    product_id: int,
    similarity_map: Any,
) -> None:
    """
    Attach blended (opencv+clip)/2 scores from ImageSimilarityMap to each review.
    Takes the mean across all image URLs that exist in the map.
    Reviews with no match keep clip_similarity_score = 0.0.
    """
    product_scores: dict = similarity_map.get_product_images(product_id)
    for review in reviews:
        urls = _extract_urls(getattr(review, "images", None) or [])
        scores = [product_scores[url] for url in urls if url in product_scores]
        review.clip_similarity_score = sum(scores) / len(scores) if scores else 0.0

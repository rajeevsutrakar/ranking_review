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
from app.services.run_logger import log_warn

_cv2_module = None


def _get_cv2():
    global _cv2_module
    if _cv2_module is None:
        import cv2

        _cv2_module = cv2
    return _cv2_module


def _normalize_blur(laplacian_var: float) -> float:
    # Real photos on smooth backgrounds: Laplacian variance 50–600.
    # Real photos of printed/patterned subjects (textured fabric, busy
    # backgrounds): legitimately 1200–6000 from genuine pattern detail.
    # Sensor noise / corrupted images: 8000+.
    if laplacian_var > 6000.0:
        return 0.0
    return float(min(1.0, max(0.0, laplacian_var / 350.0)))


def _normalize_brightness(gray: np.ndarray) -> float:
    """Brightness score: penalize very dark/very bright images."""
    mean_b = float(np.mean(gray)) / 255.0
    # Ideal range: 0.2-0.7. Lower bound is 0.2 (not 0.3) so that legitimate
    # photos of dark-coloured products (black kurta, dark fabric) are not penalised.
    if 0.2 <= mean_b <= 0.7:
        return 1.0
    if mean_b < 0.2:
        return max(0.0, mean_b / 0.2)  # 0 at black, 1.0 at 0.2
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


def _score_single_bgr(img: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return (total, blur_n, bright_n, res_n) or None if image is invalid."""
    if img is None or img.size == 0:
        return None
    cv2 = _get_cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Centre 60% crop for blur measurement only — keeps the metric focused on
    # the product and away from background texture. Brightness and resolution
    # still use the full frame.
    yc, xc = h // 5, w // 5
    blur_region = gray[yc:h - yc, xc:w - xc]
    lap = cv2.Laplacian(blur_region, cv2.CV_64F)
    blur_var = float(lap.var())
    blur_n = _normalize_blur(blur_var)
    bright_n = _normalize_brightness(gray)
    res_n = _normalize_resolution(h, w)
    # Composite = blur only. Brightness and resolution are degenerate on
    # Shopify-CDN catalogue images (98%+ saturate at 1.0) so they contribute
    # no discriminative signal. They are still returned per-dimension so
    # downstream consumers can use them independently.
    total = float(blur_n)
    return total, blur_n, bright_n, res_n


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
    fname = url.rsplit("/", 1)[-1].split("?")[0]
    try:
        r = requests.get(
            url,
            timeout=IMAGE_QUALITY_REQUEST_TIMEOUT_S,
            headers={"User-Agent": "ProductReviewRankerNoLLM/1.0"},
        )
        r.raise_for_status()
        return r.content
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        log_warn(f"[opencv] HTTP {status} fetching image: {fname}  url={url}")
    except requests.exceptions.Timeout:
        log_warn(f"[opencv] Timeout fetching image: {fname}  url={url}")
    except requests.exceptions.ConnectionError:
        log_warn(f"[opencv] Connection error fetching image: {fname}  url={url}")
    except Exception as exc:
        log_warn(f"[opencv] Unexpected fetch error ({type(exc).__name__}): {fname}  url={url}")
    return None


def _score_one_url(url: str) -> tuple[float, float, float, float] | None:
    raw = _fetch_image_bytes(url)
    if not raw:
        return None
    img = _decode_image_bytes(raw)
    if img is None:
        fname = url.rsplit("/", 1)[-1].split("?")[0]
        log_warn(f"[opencv] Decode failed (unsupported format or corrupt data): {fname}  url={url}")
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
        result = _score_one_url(url)
        if result is not None:
            scores.append(result[0])

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def compute_image_quality_score_with_components(
    images: Any, max_images: int | None = None
) -> tuple[float, float, float, float]:
    """
    Returns (opencv_score, blur_score, brightness_score, resolution_score).
    All values are means across successfully decoded images. Returns (0,0,0,0) if none.
    """
    cap = max_images if max_images is not None else IMAGE_QUALITY_MAX_IMAGES
    cap = max(0, int(cap))

    urls = _extract_urls(images)[:cap]
    if not urls:
        return 0.0, 0.0, 0.0, 0.0

    totals: List[float] = []
    blurs: List[float] = []
    brights: List[float] = []
    ress: List[float] = []
    for url in urls:
        result = _score_one_url(url)
        if result is not None:
            total, blur_n, bright_n, res_n = result
            totals.append(total)
            blurs.append(blur_n)
            brights.append(bright_n)
            ress.append(res_n)

    if not totals:
        return 0.0, 0.0, 0.0, 0.0
    n = len(totals)
    return (
        float(sum(totals) / n),
        float(sum(blurs) / n),
        float(sum(brights) / n),
        float(sum(ress) / n),
    )


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
        result = _score_one_url(url)
        if result is not None:
            url_scores[url] = float(result[0])

    return url_scores


def set_image_quality_scores(reviews: Sequence[Any]) -> None:
    for review in reviews:
        images = getattr(review, "images", None) or []
        opencv, blur, brightness, resolution = compute_image_quality_score_with_components(images)
        review.image_quality_score = opencv
        review.blur_score = blur
        review.brightness_score = brightness
        review.resolution_score = resolution


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

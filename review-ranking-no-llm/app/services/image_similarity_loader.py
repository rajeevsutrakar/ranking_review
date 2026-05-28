"""
Load review images into ImageSimilarityMap with blended scores.

Score stored per image:
    products[product_id][image_url] = (opencv_score + clip_score) / 2

Reference image for CLIP comes from pipeline_shopify_product.media_url (one row
per product). Falls back to title text-prompt when media_url is missing.
CLIP scores are cached in clip_score_cache.json so re-runs cost nothing.
"""
from __future__ import annotations

from typing import Dict, List

from app.models.image_similarity import ImageSimilarityMap
from app.services.clip_client import (
    build_reference_embedding,
    compute_clip_scores,
    get_or_load_cache,
)
from app.services.db_client import (
    _build_conn,
    _get_psycopg2,
    fetch_product_reference_images,
)
from app.services.image_quality import compute_image_quality_scores_per_url
from app.services.run_logger import log_info

# Text-to-image CLIP is inherently less discriminative than image-to-image.
# Scale down text-reference CLIP scores before blending so noisy text scores
# don't dominate the combined ranking.
_TEXT_CLIP_SCALE = 0.75


def _fetch_review_images(
    product_ids: List[int] | None,
) -> Dict[int, Dict[str, float]]:
    """
    Query shopify_product_review and return:
        {product_id: {image_url: opencv_score}}
    """
    _, RealDictCursor = _get_psycopg2()

    product_filter = ""
    params: list = []
    if product_ids:
        placeholders = ",".join(["%s"] * len(product_ids))
        product_filter = f"AND product_id IN ({placeholders})"
        params = product_ids

    query = f"""
        SELECT product_id, images
        FROM shopify_product_review
        WHERE images IS NOT NULL
          AND jsonb_array_length(images) > 0
          AND status = TRUE
          AND show_at_frontend = TRUE
          {product_filter}
        ORDER BY product_id
    """

    result: Dict[int, Dict[str, float]] = {}
    with _build_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                pid = int(row["product_id"])
                url_scores = compute_image_quality_scores_per_url(row["images"])
                if pid not in result:
                    result[pid] = {}
                for url, score in url_scores.items():
                    result[pid][url] = score

    return result


def load_image_similarity_map(product_ids: List[int] | None = None) -> ImageSimilarityMap:
    """
    Build ImageSimilarityMap where each score = (opencv_score + clip_score) / 2.

    Steps
    -----
    1. Query all review images → opencv_score per (product, url).
    2. Fetch one reference row per product from pipeline_shopify_product.
    3. Build CLIP reference embedding (image URL → text fallback).
    4. For each review image compute clip_score, update running-avg cache.
    5. Blend: combined = (opencv_score + clip_score) / 2.
    6. Store in ImageSimilarityMap, sorted by URL within each product.
    7. Save CLIP cache to disk.
    """
    similarity_map = ImageSimilarityMap()
    cache = get_or_load_cache()

    # Step 1 — opencv scores per review image
    opencv_map = _fetch_review_images(product_ids)
    if not opencv_map:
        return similarity_map

    # Step 2 — reference images from pipeline_shopify_product
    all_pids = list(opencv_map.keys())
    ref_rows_by_pid = fetch_product_reference_images(all_pids)

    # Step 3–5 — per product: build reference, score, blend
    for pid, url_opencv in opencv_map.items():
        row = ref_rows_by_pid.get(pid, {})
        media_url = row.get("media_url")
        title = row.get("title")

        ref_emb, mode = build_reference_embedding(media_url, title)

        if mode != "image":
            log_info(
                f"[similarity] product_id={pid} reference mode={mode!r} "
                f"(media_url={'present' if media_url else 'missing'}, "
                f"title={'present' if title else 'missing'})"
            )

        review_urls = list(url_opencv.keys())

        if ref_emb is not None:
            # clip_scores = {url: avg_clip_score} (running avg updated in cache)
            clip_scores = compute_clip_scores(pid, review_urls, ref_emb, cache)
            # Text-reference scores are less reliable — scale down before blending
            # so OpenCV quality has proportionally more influence on the final rank.
            if mode == "text":
                clip_scores = {url: s * _TEXT_CLIP_SCALE for url, s in clip_scores.items()}
        else:
            clip_scores = {}

        # Step 5–6 — blend and store
        similarity_map.add_product(pid)
        for url in sorted(review_urls):
            opencv_score = url_opencv[url]
            clip_score = clip_scores.get(url)
            if clip_score is not None:
                combined = (opencv_score + clip_score) / 2.0
            else:
                combined = opencv_score  # CLIP unavailable → opencv only
            similarity_map.add_image(pid, url, score=combined)

    # Step 7 — persist CLIP cache
    cache.save()

    return similarity_map


def load_image_similarity_map_for_product(product_id: int) -> ImageSimilarityMap:
    """Load blended scores for a single product."""
    return load_image_similarity_map(product_ids=[product_id])

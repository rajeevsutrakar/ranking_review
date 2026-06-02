"""
Load review images into ImageSimilarityMap with blended scores.

Score stored per image:
    blended = CLIP_WEIGHT_IN_BLEND × adjusted_clip + OPENCV_WEIGHT_IN_BLEND × opencv_score

adjusted_clip already incorporates:
  - product reference similarity (primary)
  - human-wearing bonus (full-body / selfie / lifestyle tiers)
  - packaging penalty (courier bags, delivery envelopes)

No caching — the query filters score IS NULL, so each review image is
processed exactly once. There is nothing to cache.
"""
from __future__ import annotations

import traceback
from typing import Dict, List

from app.config import CLIP_WEIGHT_IN_BLEND, OPENCV_WEIGHT_IN_BLEND
from app.models.image_similarity import ImageSimilarityMap
from app.services.clip_client import (
    build_reference_embedding,
    compute_clip_scores,
    compute_clip_scores_text_ref,
)
from app.services.db_client import (
    _build_conn,
    _get_psycopg2,
    fetch_product_reference_images,
)
from app.services.image_quality import compute_image_quality_scores_per_url
from app.services.run_logger import log_info


def _fetch_review_images(
    product_ids: List[int] | None,
) -> Dict[int, Dict[str, float]]:
    """
    Query unscored review images and return:
        {product_id: {image_url: opencv_score}}

    Filters score IS NULL to match the main pipeline query — only new reviews
    are processed, so each image is CLIP-scored exactly once.
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
          AND score IS NULL
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
    Build ImageSimilarityMap for unscored review images only.

    Steps
    -----
    1. Query new (score IS NULL) review images → opencv_score per (product, url).
    2. Fetch one reference row per product from pipeline_shopify_product.
    3. Build CLIP reference embedding (product image → title text fallback).
    4. For each review image compute adjusted_clip (product similarity + tier
       boost or packaging penalty).
    5. Blend: blended = CLIP_WEIGHT × adjusted_clip + OPENCV_WEIGHT × opencv_score
    6. Store in ImageSimilarityMap.
    """
    similarity_map = ImageSimilarityMap()

    # Step 1 — opencv scores per review image
    opencv_map = _fetch_review_images(product_ids)
    if not opencv_map:
        return similarity_map

    # Step 2 — reference images from pipeline_shopify_product
    all_pids = list(opencv_map.keys())
    ref_rows_by_pid = fetch_product_reference_images(all_pids)

    # Step 3–5 — per product: build reference, score, blend
    for pid, url_opencv in opencv_map.items():
        try:
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

            if mode == "image" and ref_emb is not None:
                # Image-to-image reference: additive boost formula (most accurate)
                clip_scores = compute_clip_scores(pid, review_urls, ref_emb)
            elif mode == "text" and title:
                # Text-only reference: multiplicative formula so human boost requires
                # product relevance — prevents any human photo from ranking high
                clip_scores = compute_clip_scores_text_ref(pid, review_urls, title)
            else:
                clip_scores = {}

            # Step 5–6 — blend and store
            similarity_map.add_product(pid)
            for url in sorted(review_urls):
                opencv_score = url_opencv[url]
                clip_score = clip_scores.get(url)
                if clip_score is not None:
                    blended = CLIP_WEIGHT_IN_BLEND * clip_score + OPENCV_WEIGHT_IN_BLEND * opencv_score
                else:
                    # CLIP unavailable — fall back to opencv quality only
                    blended = opencv_score
                similarity_map.add_image(pid, url, score=min(0.99, max(0.0, blended)))

        except Exception as exc:
            log_info(
                f"[similarity] product_id={pid} scoring failed: {exc} — skipping product.\n"
                f"Traceback:\n{traceback.format_exc()}"
            )

    return similarity_map


def load_image_similarity_map_for_product(product_id: int) -> ImageSimilarityMap:
    """Load blended scores for a single product."""
    return load_image_similarity_map(product_ids=[product_id])

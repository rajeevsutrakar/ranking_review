"""
Review ranking without any LLM or embedding API calls.

Flow (all products):
  One DB query → group by product_id → load ImageSimilarityMap (opencv+clip
  blended scores) → for each product: set image quality + clip similarity
  scores → deterministic heuristic score → sort image-first.

Flow (single product):
  One DB query for that product_id → same scoring + sort.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

from app.models.review import Review
from app.services.db_client import (
    bulk_update_opencv_scores,
    fetch_all_reviews_grouped,
    upsert_image_similarity_map,
)
from app.services.heuristic_scoring import assign_product_scores, sort_reviews_deterministic
from app.services.image_quality import (
    review_has_image_url,
    set_clip_similarity_scores,
    set_image_quality_scores,
)
from app.services.image_similarity_loader import load_image_similarity_map
from app.services.run_logger import log_error, log_info

MAX_REVIEWS_PER_PRODUCT = 500


def _rank_product(product_id: int, rows: list, similarity_map) -> list:
    reviews = [Review.from_record(row) for row in rows]
    set_image_quality_scores(reviews)
    set_clip_similarity_scores(reviews, product_id, similarity_map)
    assign_product_scores(reviews)
    return sort_reviews_deterministic(reviews)


def _log_clip_summary(product_id: int, ranked_reviews: list) -> None:
    scores = [r.clip_similarity_score for r in ranked_reviews if r.clip_similarity_score > 0]
    if scores:
        log_info(
            f"[no-llm] product_id={product_id} clip_similarity: "
            f"scored={len(scores)}/{len(ranked_reviews)} "
            f"avg={sum(scores)/len(scores):.4f} "
            f"max={max(scores):.4f} "
            f"min={min(scores):.4f}"
        )
    else:
        log_info(f"[no-llm] product_id={product_id} clip_similarity: no images scored.")


def _serialize(ranked_by_product: dict, similarity_map) -> dict:
    return {
        "pipeline": "no_llm_opencv_heuristic_clip",
        "ranked_reviews": {
            product_id: [
                {
                    "review_id": r.review_id,
                    "score": round(float(r.score), 6),
                    "opencv_image_score": round(float(r.image_quality_score), 6),
                    "rating": int(r.rating or 0),
                    "has_image_url": review_has_image_url(r),
                }
                for r in ranked_reviews
            ]
            for product_id, ranked_reviews in ranked_by_product.items()
        },
        "image_similarity_map": {
            pid: dict(sorted(url_scores.items(), key=lambda x: x[1], reverse=True))
            for pid, url_scores in similarity_map.to_dict().items()
        },
    }


_OPENCV_CSV_HEADERS = [
    "product_id", "review_id", "rating", "has_image_url",
    "blur_score", "brightness_score", "resolution_score", "opencv_image_score",
]

_IMAGE_SIMILARITY_CSV_HEADERS = [
    "product_id", "image_url", "clip_opencv_blended_score",
]


def _write_opencv_csv(ranked_by_product: dict) -> None:
    path = Path(__file__).resolve().parent / "opencv_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OPENCV_CSV_HEADERS)
        writer.writeheader()
        for product_id, reviews in ranked_by_product.items():
            for r in reviews:
                writer.writerow({
                    "product_id": product_id,
                    "review_id": r.review_id,
                    "rating": int(r.rating or 0),
                    "has_image_url": review_has_image_url(r),
                    "blur_score": round(float(r.blur_score), 6),
                    "brightness_score": round(float(r.brightness_score), 6),
                    "resolution_score": round(float(r.resolution_score), 6),
                    "opencv_image_score": round(float(r.image_quality_score), 6),
                })
    log_info(f"OpenCV scores written to {path}")


def _write_image_similarity_csv(similarity_map) -> None:
    path = Path(__file__).resolve().parent / "image_similarity_scores.csv"
    sim_dict = similarity_map.to_dict()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_IMAGE_SIMILARITY_CSV_HEADERS)
        writer.writeheader()
        for product_id, url_scores in sim_dict.items():
            sorted_urls = sorted(url_scores.items(), key=lambda x: x[1], reverse=True)
            for image_url, score in sorted_urls:
                writer.writerow({
                    "product_id": product_id,
                    "image_url": image_url,
                    "clip_opencv_blended_score": round(float(score), 6),
                })
    log_info(f"Image similarity scores written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rank-reviews",
        description="Rank product reviews using OpenCV + CLIP image similarity + heuristic scoring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_REVIEWS_PER_PRODUCT,
        metavar="N",
        help=f"Max reviews per product (default: {MAX_REVIEWS_PER_PRODUCT}).",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        metavar="N",
        help="Max number of products to process (omit for all).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    ranked_by_product: dict[int, list] = {}

    log_info("[no-llm] Fetching all product reviews in one query.")
    try:
        grouped = fetch_all_reviews_grouped(
            limit_per_product=args.limit,
            max_products=args.max_products,
        )
    except Exception as exc:
        log_error(f"DB fetch failed: {exc}")
        sys.exit(1)

    if not grouped:
        log_info("No reviews found. Exiting.")
        sys.exit(0)

    log_info(f"[no-llm] Loaded {len(grouped)} products from DB.")

    log_info("[no-llm] Loading ImageSimilarityMap (opencv + clip blended scores).")
    try:
        similarity_map = load_image_similarity_map(product_ids=list(grouped.keys()))
    except Exception as exc:
        log_error(
            f"ImageSimilarityMap load failed: {exc}. Continuing without CLIP scores.\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        from app.models.image_similarity import ImageSimilarityMap
        similarity_map = ImageSimilarityMap()

    map_summary = similarity_map.summary()
    log_info(
        f"[no-llm] ImageSimilarityMap loaded: "
        f"{map_summary['total_products']} products, "
        f"{map_summary['total_images']} images scored."
    )

    for idx, (product_id, rows) in enumerate(grouped.items(), start=1):
        log_info(
            f"[no-llm] Scoring product_id={product_id} "
            f"({idx}/{len(grouped)}, {len(rows)} reviews)."
        )
        ranked_by_product[product_id] = _rank_product(product_id, rows, similarity_map)
        _log_clip_summary(product_id, ranked_by_product[product_id])

    # ── write opencv scores back to DB ───────────────────────────────────────
    opencv_scores: dict[int, float] = {
        r.review_id: float(r.image_quality_score)
        for reviews in ranked_by_product.values()
        for r in reviews
    }
    try:
        updated = bulk_update_opencv_scores(opencv_scores)
        log_info(f"[no-llm] Updated opencv_image_score for {updated} reviews in DB.")
    except Exception as exc:
        log_error(f"DB write failed for opencv scores: {exc}")

    # ── write image similarity map to DB ─────────────────────────────────────
    try:
        upserted = upsert_image_similarity_map(similarity_map.to_dict())
        log_info(f"[no-llm] Upserted image_similarity_map for {upserted} products in DB.")
    except Exception as exc:
        log_error(f"DB write failed for image similarity map: {exc}")

    result = _serialize(ranked_by_product, similarity_map)
    output_path = Path(__file__).resolve().parent / "ranked_results_no_llm.log"
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, indent=2))
        fh.write("\n\n")

    log_info(f"Result appended to {output_path}")

    _write_opencv_csv(ranked_by_product)
    _write_image_similarity_csv(similarity_map)

    log_info(f"Run finished in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

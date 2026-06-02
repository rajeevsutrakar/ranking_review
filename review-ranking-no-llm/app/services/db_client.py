from collections import defaultdict
from importlib import import_module
from typing import Any, Dict, List

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def _get_psycopg2():
    try:
        psycopg2 = import_module("psycopg2")
        psycopg2_extras = import_module("psycopg2.extras")
        return psycopg2, psycopg2_extras.RealDictCursor
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for DB access. Install with: pip install psycopg2-binary"
        ) from exc


def _build_conn():
    if not all([DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER]):
        raise ValueError("Database configuration is incomplete.")
    psycopg2, _ = _get_psycopg2()
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


_REVIEW_COLUMNS = """
    product_id,
    id AS review_id,
    review AS text,
    rating,
    EXTRACT(EPOCH FROM created_at) AS timestamp,
    images,
    CASE
        WHEN images IS NULL THEN FALSE
        WHEN jsonb_typeof(images) = 'array' THEN jsonb_array_length(images) > 0
        ELSE FALSE
    END AS has_image
"""

_REVIEW_FILTER = """
    status = TRUE
    AND show_at_frontend = TRUE
    AND score IS NULL
    AND (
        (review IS NOT NULL AND btrim(review) <> '')
        OR (
            images IS NOT NULL
            AND jsonb_typeof(images) = 'array'
            AND jsonb_array_length(images) > 0
        )
    )
"""


def fetch_reviews_for_product(product_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """Single-product fetch — used when a specific product_id is passed on the CLI."""
    _, RealDictCursor = _get_psycopg2()
    query = f"""
        SELECT {_REVIEW_COLUMNS}
        FROM shopify_product_review
        WHERE {_REVIEW_FILTER}
          AND product_id = %s
        ORDER BY id
        LIMIT %s
    """
    with _build_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (product_id, limit))
            return [dict(row) for row in cur.fetchall()]


def fetch_product_reference_images(
    product_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Fetch the best media_url and title for each product from pipeline_shopify_product.
    Returns {product_id: {"media_url": str|None, "title": str|None}}.

    Uses MAX aggregation per field so media_url and title are resolved independently —
    a product can have title on one row and media_url on another and both are returned.

    media_url: best valid HTTP URL across all rows for that product.
    title:     any non-empty title across all rows for that product.
    """
    if not product_ids:
        return {}
    _, RealDictCursor = _get_psycopg2()
    placeholders = ",".join(["%s"] * len(product_ids))
    query = f"""
        SELECT DISTINCT ON (id)
            id AS product_id,
            media_url,
            title
        FROM pipeline_shopify_product
        WHERE id IN ({placeholders})
        ORDER BY
            id,
            (TRIM(COALESCE(media_url, '')) LIKE 'http%%') DESC,
            (TRIM(COALESCE(title,     '')) <> ''         ) DESC
    """
    from app.services.run_logger import log_info
    result: Dict[int, Dict[str, Any]] = {}
    with _build_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, product_ids)
            for row in cur.fetchall():
                pid = int(row["product_id"])
                media_url = (row.get("media_url") or "").strip() or None
                title = (row.get("title") or "").strip() or None
                result[pid] = {"media_url": media_url, "title": title}
                log_info(
                    f"[ref-fetch] product_id={pid} "
                    f"media_url={'missing' if not media_url else media_url} | "
                    f"title={'missing' if not title else repr(title)}"
                )

    missing_pids = [pid for pid in product_ids if pid not in result]
    for pid in missing_pids:
        log_info(
            f"[ref-fetch] product_id={pid} — "
            f"no row found in pipeline_shopify_product (product_id not in table)"
        )

    return result


def fetch_all_reviews_grouped(
    limit_per_product: int = 500,
    max_products: int | None = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Single query that returns all products' reviews grouped by product_id.
    Uses a window function to cap rows per product.
    If max_products is set, only the first N products (by product_id) are returned.
    """
    _, RealDictCursor = _get_psycopg2()

    params = []
    product_cte = ""

    # Build optional CTE to limit product count
    if max_products is not None and max_products > 0:
        product_cte = """
        limited_products AS (
            SELECT DISTINCT product_id
            FROM shopify_product_review
            WHERE """ + _REVIEW_FILTER + """ AND product_id IS NOT NULL
            ORDER BY product_id
            LIMIT %s
        ),
        """
        params.append(max_products)

    params.append(limit_per_product)

    product_filter = ""
    if max_products is not None and max_products > 0:
        product_filter = "AND product_id IN (SELECT product_id FROM limited_products)"

    query = f"""
        WITH {product_cte}
        ranked AS (
            SELECT
                {_REVIEW_COLUMNS},
                ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY id) AS rn
            FROM shopify_product_review
            WHERE {_REVIEW_FILTER}
              AND product_id IS NOT NULL
              {product_filter}
        )
        SELECT product_id, review_id, text, rating, timestamp, images, has_image
        FROM ranked
        WHERE rn <= %s
        ORDER BY product_id, review_id
    """
    with _build_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for row in cur.fetchall():
                pid = int(row["product_id"])
                grouped[pid].append(dict(row))
    return dict(grouped)


def _parse_products_field(raw) -> Dict[str, float]:
    """
    Parse the products JSONB field — handles both storage formats:
      Old (object):  {"url": score, ...}         ← JSONB sorts keys alphabetically
      New (array):   [[score, "url"], ...]        ← JSONB preserves array order
    Returns {url: score}.
    """
    if not raw:
        return {}
    if isinstance(raw, list):
        result = {}
        for item in raw:
            try:
                result[item[1]] = float(item[0])
            except (IndexError, TypeError, ValueError):
                pass  # skip malformed [score, url] entries
        return result
    return dict(raw)


def upsert_image_similarity_map(similarity_data: Dict[int, Dict[str, float]]) -> int:
    """
    Upsert blended (opencv+clip) scores into product_review_image_similarity.

    Storage format: JSONB array [[score, url], ...] sorted by score descending.
    Arrays preserve insertion order in PostgreSQL JSONB (unlike objects which
    sort keys alphabetically), so consumers always read data in score order.

    Logic per product_id
    --------------------
    - Exists  → merge new {url: score} into existing (new score wins), re-sort.
    - Missing → insert a fresh row.

    Returns number of rows upserted.
    """
    if not similarity_data:
        return 0

    import json as _json

    _, RealDictCursor = _get_psycopg2()
    extras = import_module("psycopg2.extras")
    product_ids = list(similarity_data.keys())
    placeholders = ",".join(["%s"] * len(product_ids))

    with _build_conn() as conn:
        # ── fetch existing rows ──────────────────────────────────────────────
        existing: Dict[int, Dict[str, float]] = {}
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT product_id, products "
                f"FROM product_review_image_similarity "
                f"WHERE product_id IN ({placeholders})",
                product_ids,
            )
            for row in cur.fetchall():
                existing[int(row["product_id"])] = _parse_products_field(row["products"])

        # ── merge + sort → store as [[score, url], ...] ──────────────────────
        rows: list[tuple] = []
        for pid, new_scores in similarity_data.items():
            merged = {**existing.get(pid, {}), **new_scores}   # new overrides old URL
            sorted_pairs = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
            rows.append((pid, _json.dumps([[score, url] for url, score in sorted_pairs])))

        # ── bulk upsert ───────────────────────────────────────────────────────
        with conn.cursor() as cur:
            extras.execute_values(
                cur,
                """
                INSERT INTO product_review_image_similarity (product_id, products, created_at)
                VALUES %s
                ON CONFLICT (product_id)
                DO UPDATE SET
                    products = EXCLUDED.products
                """,
                rows,
                template="(%s, %s::jsonb, NOW())",
            )
        conn.commit()
        return len(rows)


def fetch_image_similarity_scores(
    product_ids: List[int] | None = None,
) -> Dict[int, Dict[str, float]]:
    """
    Read image similarity scores from product_review_image_similarity.
    Handles both old object format and new array format.
    Returns {product_id: {image_url: score}} sorted highest score first.
    """
    _, RealDictCursor = _get_psycopg2()

    product_filter = ""
    params: list = []
    if product_ids:
        placeholders = ",".join(["%s"] * len(product_ids))
        product_filter = f"WHERE product_id IN ({placeholders})"
        params = product_ids

    query = f"""
        SELECT product_id, products
        FROM product_review_image_similarity
        {product_filter}
        ORDER BY product_id
    """

    result: Dict[int, Dict[str, float]] = {}
    with _build_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                pid = int(row["product_id"])
                scores = _parse_products_field(row["products"])
                result[pid] = dict(
                    sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                )
    return result


def repair_image_similarity_sort(product_ids: List[int] | None = None) -> int:
    """
    One-time migration: convert existing {url: score} object rows to the new
    [[score, url], ...] array format so order is preserved in JSONB storage.

    Returns number of rows migrated.
    """
    import json as _json

    extras = import_module("psycopg2.extras")
    existing = fetch_image_similarity_scores(product_ids)
    if not existing:
        return 0

    rows = [
        (pid, _json.dumps([[score, url] for url, score in scores.items()]))
        for pid, scores in existing.items()
    ]

    with _build_conn() as conn:
        with conn.cursor() as cur:
            extras.execute_values(
                cur,
                """
                UPDATE product_review_image_similarity AS t
                SET products = v.products::jsonb
                FROM (VALUES %s) AS v(product_id, products)
                WHERE t.product_id = v.product_id::bigint
                """,
                rows,
                template="(%s, %s)",
            )
        conn.commit()
        return len(rows)


def bulk_update_opencv_scores(review_scores: Dict[int, float]) -> int:
    """
    Write score back to shopify_product_review for each review.

    Parameters
    ----------
    review_scores : {review_id: score}

    Returns
    -------
    Number of rows updated.
    """
    if not review_scores:
        return 0

    extras = import_module("psycopg2.extras")
    rows = [(score, rid) for rid, score in review_scores.items()]

    with _build_conn() as conn:
        with conn.cursor() as cur:
            extras.execute_values(
                cur,
                """
                UPDATE shopify_product_review AS t
                SET    score = v.score
                FROM   (VALUES %s) AS v(score, review_id)
                WHERE  t.id = v.review_id::bigint
                """,
                rows,
                template="(%s, %s)",
            )
        conn.commit()
        return len(rows)

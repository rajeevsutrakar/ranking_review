# Review Ranking — No LLM

A fully deterministic product review ranking pipeline that scores and sorts customer reviews using **local OpenCV image analysis**, a **local CLIP model**, and **heuristic text scoring** — with zero calls to any external LLM or embedding API.

---

## How It Works

```
PostgreSQL DB  (query: score IS NULL — only new, unscored reviews)
     │
     ▼
Fetch reviews (grouped by product)
     │
     ▼
Build ImageSimilarityMap  (for new review images only)
  ├── OpenCV: blur + brightness + resolution score per image URL
  ├── CLIP: cosine similarity of review image vs product reference
  │    ├── image-to-image if media_url exists (most accurate)
  │    └── text-to-image fallback via title × 0.75 scale
  ├── Four-tier image classification (zero-shot CLIP, no extra model)
  │    ├── full_worn  → boost 1.0  (person wearing product, full body)
  │    ├── partial    → boost 0.6  (selfie, mirror, lifestyle)
  │    ├── display    → boost 0.15 (flat-lay, hanger, no person)
  │    └── packaging  → penalty   (delivery bag, shipping box)
  └── blended = 0.65 × adjusted_clip + 0.35 × opencv_score
     │
     ▼
Heuristic scoring per review
  ├── Image reviews  → 60% blended image score + 40% text signal
  └── Text-only      → text length + rating + recency
     │
     ▼
Sort: image-first, then by score desc, then rating desc, then recency desc
     │
     ▼
Write results
  ├── opencv scores → shopify_product_review.score (DB)
  ├── image similarity map → product_review_image_similarity.products (DB, JSONB array)
  └── ranked_results_no_llm.log (JSON append)
```

---

## Scoring Formula

### Image Score (OpenCV)

Per-image quality score computed locally using OpenCV:

```
opencv_score = 0.42 × blur_score + 0.28 × brightness_score + 0.30 × resolution_score
```

| Component | How measured | Notes |
|---|---|---|
| `blur_score` | Laplacian variance / 350, capped at 1.0 | Values > 1200 → 0.0 (noise/corrupted image) |
| `brightness_score` | Mean pixel intensity | Ideal range 0.2–0.7; outside → linear decay |
| `resolution_score` | pixels / (640×640), capped at 1.0 | 640×640 or larger → full score |

### CLIP Score

```
raw_clip = cosine_similarity(review_image_embedding, product_reference_embedding)
```

**Product reference priority:**
1. `pipeline_shopify_product.media_url` → image-to-image comparison (most accurate)
2. Title only → `"a person wearing or using {title}"` text prompt (fallback, 0.75× weight in blend)
3. Neither → CLIP scoring skipped, opencv score only

**Four-tier image classification** (zero-shot CLIP, 4 text prompts, no extra model):

| Tier | boost_factor | Condition | Example |
|---|---|---|---|
| Full-body worn | `1.0` | full-worn sim dominates display sim by >0.02 | Standing full-length outfit photo |
| Partial / selfie | `0.6` | partial-worn sim > display sim | Mirror selfie, lifestyle shot, upper-body |
| Product displayed | `0.15` | display sim wins, no packaging | Flat-lay, hanger, mannequin |
| Packaging | penalty | packaging sim ≥ best positive | Courier bag, delivery mailer, brand bag |

**Adjusted CLIP score:**

```
# Human-wearing tiers:
adjusted_clip = raw_clip + boost_factor × CLIP_HUMAN_WEIGHT × (1 − raw_clip)

# Packaging tier:
adjusted_clip = raw_clip × PACKAGING_SCORE_PENALTY
```

With defaults `CLIP_HUMAN_WEIGHT=0.70`, `PACKAGING_SCORE_PENALTY=0.20`:
- Full-body worn, raw_clip=0.30 → adjusted = **0.79**
- Selfie/mirror, raw_clip=0.28 → adjusted = **0.58**
- Flat-lay (product shown), raw_clip=0.45 → adjusted = **0.51** (small boost only)
- Packaging bag, raw_clip=0.20 → adjusted = **0.04** (penalised)

**Blended image score** stored in `product_review_image_similarity`:

```
blended = CLIP_WEIGHT_IN_BLEND × adjusted_clip + OPENCV_WEIGHT_IN_BLEND × opencv_score
        = 0.65 × adjusted_clip + 0.35 × opencv_score
```

CLIP (product relevance + usage context) is the primary signal at 65%.
OpenCV quality (blur, brightness, resolution) is the modifier at 35%.

### Final Review Score

```
text_signal = 0.40 × text_length_norm + 0.35 × rating_norm + 0.25 × recency_decay

# Review with image (successfully scored):
final_score = 0.60 × blended_score + 0.40 × text_signal

# Review with image URL but OpenCV failed + has text:
final_score = text_signal + 0.03   (small boost for attempted image)

# Text-only review:
final_score = text_signal

# No text and no image:
final_score = 0.35 × rating_norm + 0.65 × recency_decay
```

`recency_decay = exp(−ln(2) × age_days / RECENCY_HALF_LIFE_DAYS)` — halves every 30 days by default.

### Sort Order

```
(image_bucket ASC, score DESC, rating DESC, recency DESC, review_id ASC)
```

Image reviews always appear before text-only reviews within a product.

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL with the tables described below
- `pip install -e .` from the project root

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db
DB_USER=your_user
DB_PASSWORD=your_password

# HuggingFace (optional — suppresses auth warning)
HF_ACCESS_TOKEN=hf_...

# Final review ranking: image score vs text signal
NO_LLM_IMAGE_BLEND_WEIGHT=0.60
NO_LLM_TEXT_BLEND_WEIGHT=0.40

# Image similarity blend: CLIP relevance vs OpenCV quality
CLIP_WEIGHT_IN_BLEND=0.65
OPENCV_WEIGHT_IN_BLEND=0.35

# CLIP scoring
CLIP_HUMAN_WEIGHT=0.70
PACKAGING_SCORE_PENALTY=0.20

# Other
RECENCY_HALF_LIFE_DAYS=30
IMAGE_QUALITY_MAX_IMAGES=15
IMAGE_QUALITY_REQUEST_TIMEOUT_S=8
```

---

## Running

```bash
# Rank all products (default: up to 500 reviews per product)
python -m app.main

# Limit reviews per product
python -m app.main --limit 100

# Process only the first N products
python -m app.main --max-products 10

# If installed via pip install -e .
rank-reviews --limit 100
```

**First run** downloads `openai/clip-vit-base-patch32` (~600MB) from HuggingFace. Subsequent runs use the local cached model.

---

## Output

| File / Table | Content |
|---|---|
| `app/ranked_results_no_llm.log` | JSON append per run — ranked review IDs + scores per product |
| `app/runtime_no_llm.log` | Timestamped run log — includes per-image `[clip]` tier lines for debugging |
| `shopify_product_review.score` | OpenCV image quality score written back per review |
| `product_review_image_similarity.products` | JSONB array `[[score, url], ...]` sorted by score desc per product |

---

## Database Tables

### Read from

| Table | Columns used |
|---|---|
| `shopify_product_review` | `id`, `product_id`, `review`, `rating`, `created_at`, `images`, `status`, `show_at_frontend`, `score` |
| `pipeline_shopify_product` | `id`, `media_url`, `title` |

**Review filter** (only unscored, visible reviews with content):
```sql
status = TRUE
AND show_at_frontend = TRUE
AND score IS NULL
AND (review IS NOT NULL OR images array is non-empty)
```

### Written to

| Table | What is written |
|---|---|
| `shopify_product_review` | `score` = opencv image quality score (float) |
| `product_review_image_similarity` | `products` = `[[score, url], ...]` JSONB array, sorted score desc |

**`product_review_image_similarity` schema:**
```sql
CREATE TABLE product_review_image_similarity (
    product_id  BIGINT PRIMARY KEY,
    products    JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMP DEFAULT NOW()
);
```

> **Note:** `products` is stored as a JSONB **array** (not object) so that score-descending order is preserved on read. PostgreSQL JSONB objects always re-sort keys alphabetically.

---

## Project Structure

```
app/
├── main.py                         Entry point — CLI, orchestration
├── config.py                       Env vars, weights, constants
│
├── models/
│   ├── review.py                   Review dataclass + DB row parser
│   └── image_similarity.py         ImageSimilarityMap: {product_id: {url: score}}
│
└── services/
    ├── db_client.py                All PostgreSQL queries (fetch + write)
    ├── clip_client.py              CLIP model wrapper — embed, cosine sim,
    │                               four-tier image classification
    ├── image_quality.py            OpenCV blur/brightness/resolution scorer
    ├── image_similarity_loader.py  Builds ImageSimilarityMap (opencv + CLIP blend)
    ├── heuristic_scoring.py        Text signal, recency decay, final score + sort
    └── run_logger.py               Append-only log to runtime_no_llm.log
```

---

## Configuration Reference

**Final review ranking**

| Variable | Default | Description |
|---|---|---|
| `NO_LLM_IMAGE_BLEND_WEIGHT` | `0.60` | Weight of image blended score in final review score |
| `NO_LLM_TEXT_BLEND_WEIGHT` | `0.40` | Weight of text signal (length + rating + recency) in final review score |
| `RECENCY_HALF_LIFE_DAYS` | `30` | Days for recency score to halve |

**Image similarity blend**

| Variable | Default | Description |
|---|---|---|
| `CLIP_WEIGHT_IN_BLEND` | `0.65` | CLIP product-relevance share in the blended image score |
| `OPENCV_WEIGHT_IN_BLEND` | `0.35` | OpenCV quality share in the blended image score |

**CLIP scoring**

| Variable | Default | Description |
|---|---|---|
| `CLIP_HUMAN_WEIGHT` | `0.70` | Strength of the wearing-usage boost (0=off, 1=always max) |
| `PACKAGING_SCORE_PENALTY` | `0.20` | Multiplier applied to CLIP score for packaging/delivery images |

**Image fetch**

| Variable | Default | Description |
|---|---|---|
| `IMAGE_QUALITY_MAX_IMAGES` | `15` | Max images scored per review |
| `IMAGE_QUALITY_REQUEST_TIMEOUT_S` | `8` | HTTP timeout for image fetches |

---

## CLIP Image Classification Tiers

The pipeline uses zero-shot CLIP classification (4 text prompts, no extra model) to assign each review image a tier that drives its adjusted CLIP score:

| Tier | boost_factor | Adjusted score | Example |
|---|---|---|---|
| Full-body worn | `1.0` | `raw + 1.0 × 0.70 × (1−raw)` | Standing full-length outfit photo |
| Partial / selfie | `0.6` | `raw + 0.6 × 0.70 × (1−raw)` | Mirror selfie, lifestyle shot, upper-body |
| Product displayed | `0.15` | `raw + 0.15 × 0.70 × (1−raw)` | Flat-lay, hanger, mannequin |
| Packaging | penalty | `raw × 0.20` | Courier bag, delivery mailer, brand bag |

Mirror selfies and casual worn photos are not penalised — image quality (OpenCV) differentiates them from studio shots at the same tier. Packaging images are penalised regardless of raw CLIP similarity, since brand illustrations on mailer bags can otherwise achieve deceptively high product similarity scores.

---

## Known Limitations

- CLIP runs on CPU by default — scoring 100 images takes ~2–5 minutes depending on hardware.
- Noise/corrupted images (Laplacian variance > 1200) are automatically scored 0 on the blur component.
- Each review is scored exactly once (filtered by `score IS NULL`). Re-scoring requires resetting `shopify_product_review.score` to NULL for the affected reviews.

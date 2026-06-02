# Scoring Weights Reference

All default values are defined in `app/config.py` and can be overridden via `.env`.

---

## 1. OpenCV Image Quality Score
**File:** `app/services/image_quality.py`

Formula: `opencv_score = blur_n`

| Sub-score   | In composite      | How it is computed                                                          |
|-------------|-------------------|-----------------------------------------------------------------------------|
| Blur        | Yes               | Centre 60% crop → Laplacian variance / 350, capped at 1.0. Values > 6000 → 0 |
| Brightness  | No (per-dim only) | 1.0 if mean brightness in [0.2–0.7], linear penalty outside range            |
| Resolution  | No (per-dim only) | pixels / (640×640), capped at 1.0                                            |

Brightness and resolution are computed and returned for downstream use but
excluded from the composite — both saturated at 1.0 for 98%+ of Shopify-CDN
catalogue images and provide no discriminative signal.

Output range: [0.0, 1.0]

---

## 2. CLIP Tier Boosts — Image Reference Mode
**File:** `app/services/clip_client.py`  
**Config:** `CLIP_HUMAN_WEIGHT = 0.70`, `PACKAGING_SCORE_PENALTY = 0.20`

Formula: `adjusted = raw_clip + boost_factor × CLIP_HUMAN_WEIGHT × (1 − raw_clip)`

| Tier        | Boost Factor | Description                                          | Adjusted score formula              |
|-------------|-------------|------------------------------------------------------|-------------------------------------|
| full_worn   | 1.0         | Full-body photo, person wearing the product          | `raw + 1.0 × 0.70 × (1 − raw)`     |
| partial     | 0.6         | Selfie / mirror shot / upper-body / lifestyle photo  | `raw + 0.6 × 0.70 × (1 − raw)`     |
| display     | 0.15        | Flat-lay, hanger, mannequin — product only, no person| `raw + 0.15 × 0.70 × (1 − raw)`    |
| packaging   | penalty     | Courier bag / delivery box                           | `raw × 0.20`                        |

Output capped at **0.99** (1.0 reserved as unreachable).

---

## 3. CLIP Scoring — Text Reference Mode
**File:** `app/services/clip_client.py` · `compute_clip_scores_text_ref()`  
Used when product has **no valid media_url**, only a title.

Formula: `adjusted = product_sim × (1.0 + boost_factor × CLIP_HUMAN_WEIGHT)`

| Signal         | Description                                              |
|----------------|----------------------------------------------------------|
| `product_sim`  | `cosine(image_emb, embed_text("a photo of {title}"))` — pure product relevance |
| Tier boost     | Same four tiers as above, applied multiplicatively       |
| Packaging      | `product_sim × 0.20`                                     |

Multiplicative formula ensures low product relevance stays low even with full human boost.

Output capped at **0.99**.

---

## 4. Blended Similarity Score (CLIP + OpenCV)
**File:** `app/services/image_similarity_loader.py`  
**Config:** `CLIP_WEIGHT_IN_BLEND = 0.65`, `OPENCV_WEIGHT_IN_BLEND = 0.35`

Formula: `blended = 0.65 × adjusted_clip + 0.35 × opencv_score`

| Signal         | Weight | Notes                                     |
|----------------|--------|-------------------------------------------|
| CLIP adjusted  | 0.65   | Product relevance + human tier boost      |
| OpenCV quality | 0.35   | Blur + brightness + resolution composite  |

Output capped at **0.99**. Stored per image URL in `product_review_image_similarity`.  
If CLIP is unavailable (no product reference): `blended = opencv_score` only.

---

## 5. Final Review Heuristic Score
**File:** `app/services/heuristic_scoring.py`  
**Config:** `TEXT_BLEND_WEIGHT = 0.40`, `IMAGE_BLEND_WEIGHT = 0.60`

### Review with image + text
Formula: `score = 0.40 × text_signal + 0.60 × opencv_image_score`

### Review with text only (no image URL)
Formula: `score = text_signal`

### Review with image URL but no text
Formula: `score = opencv_image_score`

### Review with no image and no text (rating + recency only)
Formula: `score = 0.35 × rating_norm + 0.65 × recency`

---

## 6. Text-Side Signal Breakdown
**File:** `app/services/heuristic_scoring.py`  
**Config:** `WEIGHTS = { text_length: 0.40, rating: 0.35, recency: 0.25 }`

Formula (normalised weights, always sum to 1.0):

| Component    | Weight | How it is computed                                         |
|--------------|--------|------------------------------------------------------------|
| Text length  | 0.40   | `min(1.0, len(text) / 800)`                                |
| Rating       | 0.35   | `rating / 5.0`, clamped to [0, 1]                          |
| Recency      | 0.25   | `exp(−ln(2) × age_days / 30)` — half-life 30 days         |

**Config:** `RECENCY_HALF_LIFE_DAYS = 30`

---

## 7. Sort Order (Deterministic Tie-breaking)
**File:** `app/services/heuristic_scoring.py` · `sort_reviews_deterministic()`

Reviews are sorted by this key in order:

1. **Bucket** — reviews with image URL rank above text-only reviews
2. **Score** — higher heuristic score first
3. **Rating** — higher star rating first
4. **Recency** — more recent first
5. **review_id** — ascending (stable tie-break)

---

## .env Overrides

All weights can be changed at runtime via `.env` without touching code:

```
NO_LLM_TEXT_BLEND_WEIGHT=0.40
NO_LLM_IMAGE_BLEND_WEIGHT=0.60
CLIP_WEIGHT_IN_BLEND=0.65
OPENCV_WEIGHT_IN_BLEND=0.35
CLIP_HUMAN_WEIGHT=0.70
PACKAGING_SCORE_PENALTY=0.20
RECENCY_HALF_LIFE_DAYS=30
IMAGE_QUALITY_MAX_IMAGES=15
IMAGE_QUALITY_REQUEST_TIMEOUT_S=8
```

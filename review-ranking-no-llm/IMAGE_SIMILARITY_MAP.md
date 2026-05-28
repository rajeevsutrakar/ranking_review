# Image Similarity Map

Simple, lightweight HashMap for storing review image similarity scores.

**Structure:** `{product_id: {image_url: similarity_score}}`

## Quick Start

```python
from app.services.image_similarity_loader import load_image_similarity_map

# Load all products with their review images (score initialized to 0.0)
similarity_map = load_image_similarity_map()

# Or load specific products for testing
similarity_map = load_image_similarity_map(product_ids=[123, 456])

# View structure
print(similarity_map.to_dict())
# Output:
# {
#   123: {
#     "https://example.com/rev1.jpg": 0.0,
#     "https://example.com/rev2.jpg": 0.0,
#   },
#   456: {...}
# }
```

## Update Similarity Scores

```python
# Update single image score
similarity_map.update_score(123, "https://example.com/rev1.jpg", 0.85)

# Batch update all images for a product
product_images = similarity_map.get_product_images(123)
for image_url in product_images.keys():
    score = your_model.compute(image_url)  # From your model
    similarity_map.update_score(123, image_url, score)
```

## Database Storage (JSONB)

Store directly in PostgreSQL JSONB:

```sql
CREATE TABLE product_image_similarity (
    product_id BIGINT PRIMARY KEY,
    images JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Insert/update:
```python
import json
from app.services.db_client import _build_conn, _get_psycopg2

similarity_map = load_image_similarity_map(product_ids=[123])
data = similarity_map.to_dict()

_, _ = _get_psycopg2()
with _build_conn() as conn:
    with conn.cursor() as cur:
        for product_id, images in data.items():
            cur.execute(
                """INSERT INTO product_image_similarity (product_id, images)
                   VALUES (%s, %s)
                   ON CONFLICT (product_id) DO UPDATE SET images = %s""",
                (product_id, json.dumps(images), json.dumps(images))
            )
    conn.commit()
```

## Example Data Structure

```json
{
  "123": {
    "https://example.com/review1.jpg": 0.85,
    "https://example.com/review2.jpg": 0.72,
    "https://example.com/review3.jpg": 0.91
  },
  "456": {
    "https://example.com/review4.jpg": 0.65,
    "https://example.com/review5.jpg": 0.78
  }
}
```

## Performance

- **Load**: O(R) where R = total review images
- **Update**: O(1) per image
- **Query**: O(1) per image
- **Memory**: O(P + R) where P = products, R = images
- **JSONB**: Direct serialization, no conversion needed

## Files

| File | Purpose |
|------|---------|
| `app/models/image_similarity.py` | Core HashMap structure |
| `app/services/image_similarity_loader.py` | DB loader |
| `app/examples/image_similarity_example.py` | Usage examples |

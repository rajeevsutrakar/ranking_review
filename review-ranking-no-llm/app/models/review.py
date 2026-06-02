import json
from datetime import datetime


class Review:
    def __init__(self, review_id, text, rating, timestamp, has_image, images=None):
        self.review_id = review_id
        self.text = text
        self.rating = rating
        self.timestamp = timestamp
        self.has_image = has_image
        self.images = images or []

        self.score = 0.0
        self.image_quality_score = 0.0
        self.clip_similarity_score = 0.0

        # OpenCV sub-scores (set by set_image_quality_scores)
        self.blur_score = 0.0
        self.brightness_score = 0.0
        self.resolution_score = 0.0

    @classmethod
    def from_record(cls, record):
        review_id = record.get("review_id", record.get("id"))
        text = record.get("text", record.get("review", "")) or ""
        rating = int(record.get("rating", 0) or 0)
        raw_images = record.get("images")
        if isinstance(raw_images, str) and raw_images.strip():
            try:
                raw_images = json.loads(raw_images)
            except json.JSONDecodeError:
                raw_images = []
        if isinstance(raw_images, list):
            images = raw_images
        elif raw_images:
            images = [raw_images]
        else:
            images = []

        raw_timestamp = record.get("timestamp", record.get("created_at"))
        if isinstance(raw_timestamp, datetime):
            timestamp = raw_timestamp.timestamp()
        elif isinstance(raw_timestamp, (int, float)):
            timestamp = float(raw_timestamp)
        elif isinstance(raw_timestamp, str):
            try:
                timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00")).timestamp()
            except ValueError:
                timestamp = datetime.now().timestamp()
        else:
            timestamp = datetime.now().timestamp()

        if "has_image" in record:
            has_image = bool(record.get("has_image"))
        else:
            has_image = bool(images)

        return cls(
            review_id=review_id,
            text=text,
            rating=rating,
            timestamp=timestamp,
            has_image=has_image,
            images=images,
        )

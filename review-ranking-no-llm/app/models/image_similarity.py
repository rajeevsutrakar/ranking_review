"""
Simplified image similarity data structure.
Maps productId → {image_url: similarity_score}
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ImageSimilarityMap:
    """
    Simple HashMap: productId → {image_url: similarity_score}
    JSONB-compatible for direct database storage.
    """
    products: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def add_product(self, product_id: int) -> None:
        """Create entry for product."""
        if product_id not in self.products:
            self.products[product_id] = {}

    def add_image(self, product_id: int, image_url: str, score: float = 0.0) -> None:
        """Add image with initial score."""
        self.add_product(product_id)
        self.products[product_id][image_url] = score

    def update_score(self, product_id: int, image_url: str, score: float) -> None:
        """Update similarity score for an image."""
        if product_id not in self.products:
            raise KeyError(f"Product {product_id} not in map")
        if image_url not in self.products[product_id]:
            raise KeyError(f"Image {image_url} not in product {product_id}")
        self.products[product_id][image_url] = score

    def get_score(self, product_id: int, image_url: str) -> float:
        """Get similarity score for an image."""
        return self.products.get(product_id, {}).get(image_url, 0.0)

    def get_product_images(self, product_id: int) -> Dict[str, float]:
        """Get all images and scores for a product."""
        return self.products.get(product_id, {}).copy()

    def to_dict(self) -> Dict[int, Dict[str, float]]:
        """Export as dict (JSONB-ready)."""
        return self.products.copy()

    def from_dict(self, data: Dict[int, Dict[str, float]]) -> None:
        """Load from dict."""
        self.products = data.copy()

    def summary(self) -> dict:
        """Summary stats."""
        return {
            "total_products": len(self.products),
            "total_images": sum(len(imgs) for imgs in self.products.values()),
            "products": {
                pid: len(imgs) for pid, imgs in self.products.items()
            },
        }

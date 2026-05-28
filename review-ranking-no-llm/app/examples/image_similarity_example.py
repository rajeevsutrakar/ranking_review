"""
Example usage of simplified ImageSimilarityMap.
Structure: {product_id: {image_url: similarity_score}}
"""
from app.models.image_similarity import ImageSimilarityMap
from app.services.image_similarity_loader import load_image_similarity_map


def example_basic_usage():
    """Basic example: create map, add images, update scores."""
    similarity_map = ImageSimilarityMap()

    # Add product 1 with images
    similarity_map.add_image(123, "https://example.com/rev1.jpg", 0.0)
    similarity_map.add_image(123, "https://example.com/rev2.jpg", 0.0)
    similarity_map.add_image(123, "https://example.com/rev3.jpg", 0.0)

    # Add product 2
    similarity_map.add_image(456, "https://example.com/rev4.jpg", 0.0)
    similarity_map.add_image(456, "https://example.com/rev5.jpg", 0.0)

    print("Initial state:")
    print(similarity_map.to_dict())
    # Output: {
    #   123: {
    #     "https://example.com/rev1.jpg": 0.0,
    #     "https://example.com/rev2.jpg": 0.0,
    #     "https://example.com/rev3.jpg": 0.0
    #   },
    #   456: { ... }
    # }

    # Update similarity scores
    similarity_map.update_score(123, "https://example.com/rev1.jpg", 0.85)
    similarity_map.update_score(123, "https://example.com/rev2.jpg", 0.72)
    similarity_map.update_score(123, "https://example.com/rev3.jpg", 0.91)

    print("\nAfter updating scores:")
    print(similarity_map.to_dict())

    # Get single score
    score = similarity_map.get_score(123, "https://example.com/rev1.jpg")
    print(f"\nScore for rev1.jpg: {score}")  # 0.85

    # Get all images for a product
    product_images = similarity_map.get_product_images(123)
    print(f"Product 123 images: {product_images}")

    # Summary
    print(f"\nSummary: {similarity_map.summary()}")


def example_load_from_db():
    """Load from database."""
    # Load all products
    similarity_map = load_image_similarity_map()

    print(f"Loaded {similarity_map.summary()['total_products']} products")
    print(f"Total images: {similarity_map.summary()['total_images']}")

    # Get first product
    if similarity_map.products:
        product_id = list(similarity_map.products.keys())[0]
        images = similarity_map.get_product_images(product_id)
        print(f"\nProduct {product_id} has {len(images)} review images")
        for url, score in list(images.items())[:3]:
            print(f"  {url}: {score}")

        # Simulate updating scores from a model
        for url in list(images.keys())[:3]:
            similarity_map.update_score(product_id, url, 0.75)
        print(f"\nUpdated first 3 images to score 0.75")


def example_load_specific_products():
    """Load specific products for testing."""
    product_ids = [8609857437975, 8609860321559]
    similarity_map = load_image_similarity_map(product_ids=product_ids)

    print(f"Loaded products: {list(similarity_map.products.keys())}")

    # View structure
    for product_id, images in similarity_map.products.items():
        print(f"\nProduct {product_id}:")
        print(f"  Images: {len(images)}")
        for url in list(images.keys())[:2]:
            print(f"    {url}: {images[url]}")


if __name__ == "__main__":
    print("=== Example 1: Basic Usage ===")
    example_basic_usage()

    print("\n=== Example 2: Load Specific Products ===")
    # example_load_specific_products()

    print("\n=== Example 3: Load from DB ===")
    # example_load_from_db()

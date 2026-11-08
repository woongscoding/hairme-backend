"""Image hashing helpers"""

import hashlib


def calculate_image_hash(image_data: bytes) -> str:
    """
    Calculate SHA256 hash of image data (analysis record key / log correlation)

    Args:
        image_data: Image binary data

    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(image_data).hexdigest()

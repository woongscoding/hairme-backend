"""Tests for image hashing"""

from core.hashing import calculate_image_hash


class TestImageHashing:
    """Test image hash calculation"""

    def test_calculate_image_hash(self, sample_image_bytes):
        """Test that image hash is calculated correctly"""
        hash1 = calculate_image_hash(sample_image_bytes.read())
        sample_image_bytes.seek(0)
        hash2 = calculate_image_hash(sample_image_bytes.read())

        # Same image should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_different_images_different_hashes(self):
        """Test that different images produce different hashes"""
        hash1 = calculate_image_hash(b"image_data_1")
        hash2 = calculate_image_hash(b"image_data_2")

        assert hash1 != hash2

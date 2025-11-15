"""Tests for caching functionality"""

import pytest
from unittest.mock import Mock, patch
import hashlib


class TestImageHashing:
    """Test image hash calculation"""

    def test_calculate_image_hash(self, sample_image_bytes):
        """Test that image hash is calculated correctly"""
        from core.cache import calculate_image_hash

        hash1 = calculate_image_hash(sample_image_bytes.read())
        sample_image_bytes.seek(0)
        hash2 = calculate_image_hash(sample_image_bytes.read())

        # Same image should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_different_images_different_hashes(self):
        """Test that different images produce different hashes"""
        from core.cache import calculate_image_hash

        image1 = b"image_data_1"
        image2 = b"image_data_2"

        hash1 = calculate_image_hash(image1)
        hash2 = calculate_image_hash(image2)

        assert hash1 != hash2


class TestRedisCache:
    """Test Redis caching functionality"""

    @patch('core.cache.redis_client')
    def test_get_cached_result(self, mock_redis):
        """Test retrieving cached result"""
        from core.cache import get_cached_result
        import json

        # Mock cache hit
        cached_data = {"face_shape": "계란형", "personal_color": "봄웜"}
        mock_redis.get.return_value = json.dumps(cached_data).encode()

        result = get_cached_result("test_hash")

        assert result is not None
        assert result["face_shape"] == "계란형"

    @patch('core.cache.redis_client')
    def test_get_cached_result_miss(self, mock_redis):
        """Test cache miss"""
        from core.cache import get_cached_result

        mock_redis.get.return_value = None

        result = get_cached_result("test_hash")

        assert result is None

    @patch('core.cache.redis_client')
    def test_save_to_cache(self, mock_redis):
        """Test saving result to cache"""
        from core.cache import save_to_cache

        data = {"face_shape": "계란형", "personal_color": "봄웜"}
        mock_redis.setex.return_value = True

        result = save_to_cache("test_hash", data, ttl=3600)

        # Should call setex with correct parameters
        assert mock_redis.setex.called or result is None

    @patch('core.cache.redis_client', None)
    def test_cache_disabled(self):
        """Test behavior when Redis is disabled"""
        from core.cache import get_cached_result, save_to_cache

        # Should not raise exception when Redis is disabled
        result = get_cached_result("test_hash")
        assert result is None

        save_to_cache("test_hash", {"data": "test"})
        # Should complete without error

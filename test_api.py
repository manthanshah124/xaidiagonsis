"""Tests for the prediction API endpoint."""

import io
import json
from PIL import Image
import pytest
from django.test import Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile


@pytest.fixture
def client():
    """Create a test client."""
    return Client()


@pytest.fixture
def valid_jpeg_file():
    """Create a valid JPEG test file as SimpleUploadedFile."""
    img = Image.new('RGB', (300, 300), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return SimpleUploadedFile(
        'test.jpg',
        buffer.getvalue(),
        content_type='image/jpeg'
    )


@pytest.fixture
def valid_png_file():
    """Create a valid PNG test file as SimpleUploadedFile."""
    img = Image.new('RGB', (300, 300), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return SimpleUploadedFile(
        'test.png',
        buffer.getvalue(),
        content_type='image/png'
    )


@pytest.fixture
def small_image_file():
    """Create an image below minimum dimensions."""
    img = Image.new('RGB', (50, 50), color='green')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return SimpleUploadedFile(
        'small.jpg',
        buffer.getvalue(),
        content_type='image/jpeg'
    )


@pytest.fixture
def large_image_file():
    """Create an image that exceeds 5MB when saved."""
    import numpy as np
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    return SimpleUploadedFile(
        'large.jpg',
        buffer.getvalue(),
        content_type='image/jpeg'
    )


@pytest.fixture
def corrupt_file():
    """Create a corrupt file."""
    return SimpleUploadedFile(
        'corrupt.jpg',
        b'not an image file',
        content_type='image/jpeg'
    )


@pytest.fixture
def bmp_file():
    """Create a BMP format image (unsupported)."""
    img = Image.new('RGB', (224, 224), color='purple')
    buffer = io.BytesIO()
    img.save(buffer, format='BMP')
    buffer.seek(0)
    return SimpleUploadedFile(
        'test.bmp',
        buffer.getvalue(),
        content_type='image/bmp'
    )


class TestPredictEndpoint:
    """Tests for POST /api/predict/ endpoint."""

    def test_valid_jpeg_upload(self, client, valid_jpeg_file):
        """Test uploading a valid JPEG image."""
        response = client.post(
            reverse('predict'),
            {'image': valid_jpeg_file},
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'disease' in data
        assert 'display_name' in data
        assert 'confidence' in data
        assert 'top3' in data
        assert 'original_image_base64' in data
        assert 'gradcam_image_base64' in data
        assert 'warnings' in data
        assert 'message' in data

    def test_valid_png_upload(self, client, valid_png_file):
        """Test uploading a valid PNG image."""
        response = client.post(
            reverse('predict'),
            {'image': valid_png_file},
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'disease' in data
        assert 'confidence' in data
        assert 'top3' in data
        assert 'original_image_base64' in data
        assert 'gradcam_image_base64' in data

    def test_no_file_uploaded(self, client):
        """Test error when no file is provided."""
        response = client.post(reverse('predict'), {})
        assert response.status_code == 400
        data = response.json()
        assert data['status'] == 'error'
        assert data['code'] == 'NO_FILE'

    def test_invalid_file_extension(self, client, bmp_file):
        """Test error for unsupported file extension."""
        response = client.post(
            reverse('predict'),
            {'image': bmp_file},
        )
        assert response.status_code == 400
        data = response.json()
        assert data['status'] == 'error'
        assert data['code'] == 'INVALID_TYPE'

    def test_corrupt_file(self, client, corrupt_file):
        """Test error for corrupt/unreadable file."""
        response = client.post(
            reverse('predict'),
            {'image': corrupt_file},
        )
        assert response.status_code == 400
        data = response.json()
        assert data['status'] == 'error'
        assert data['code'] == 'CORRUPT'

    def test_image_too_small(self, client, small_image_file):
        """Test error for image below minimum dimensions."""
        response = client.post(
            reverse('predict'),
            {'image': small_image_file},
        )
        assert response.status_code == 400
        data = response.json()
        assert data['status'] == 'error'
        assert data['code'] == 'TOO_SMALL'

    def test_file_too_large(self, client, large_image_file):
        """Test error for file exceeding size limit."""
        response = client.post(
            reverse('predict'),
            {'image': large_image_file},
        )
        # Should be 413 for TOO_LARGE
        assert response.status_code == 413
        data = response.json()
        assert data['status'] == 'error'
        assert data['code'] == 'TOO_LARGE'

    def test_csrf_token_endpoint(self, client):
        """Test CSRF token endpoint returns token."""
        response = client.get(reverse('csrf_token'))
        assert response.status_code == 200
        data = response.json()
        assert 'csrfToken' in data
        assert len(data['csrfToken']) > 0

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get(reverse('health'))
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'model_loaded' in data


class TestIndexView:
    """Tests for the main index view."""

    @pytest.mark.skip(reason="Django test client template rendering issue with Python 3.14")
    def test_index_view(self, client):
        """Test index view returns HTML."""
        response = client.get(reverse('index'))
        assert response.status_code == 200
        assert 'text/html' in response['Content-Type']
        assert b'XAI-PlantDoc' in response.content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
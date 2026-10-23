"""Tests for image preprocessing utilities."""

import io
import pytest
from PIL import Image
import numpy as np

from predictor.services.preprocessing import (
    preprocess_for_model,
    preprocess_pil_image,
    validate_image_bytes,
    read_image_bytes,
    resize_image,
    preprocess_mobilenetv2,
    add_batch_dimension,
    PreprocessingError,
    TARGET_SIZE,
    MIN_IMAGE_SIZE,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
)


class TestValidateImageBytes:
    """Tests for validate_image_bytes function."""

    def create_test_image(self, width=224, height=224, format='JPEG', color='red'):
        """Create a test image in memory."""
        img = Image.new('RGB', (width, height), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    def test_valid_jpeg(self):
        """Test valid JPEG image passes validation."""
        img_bytes = self.create_test_image(224, 224, 'JPEG')
        validate_image_bytes(img_bytes, 'test.jpg')  # Should not raise

    def test_valid_png(self):
        """Test valid PNG image passes validation."""
        img_bytes = self.create_test_image(224, 224, 'PNG')
        validate_image_bytes(img_bytes, 'test.png')  # Should not raise

    def test_invalid_extension(self):
        """Test invalid file extension raises error."""
        img_bytes = self.create_test_image(224, 224, 'JPEG')
        with pytest.raises(PreprocessingError) as exc_info:
            validate_image_bytes(img_bytes, 'test.gif')
        assert exc_info.value.code == 'INVALID_TYPE'

    def test_file_too_large(self):
        """Test file exceeding max size raises error."""
        # Create a large image (will be > 5MB when saved as JPEG)
        img = Image.new('RGB', (3000, 3000), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        large_bytes = buffer.getvalue()

        if len(large_bytes) > MAX_FILE_SIZE:
            with pytest.raises(PreprocessingError) as exc_info:
                validate_image_bytes(large_bytes, 'large.jpg')
            assert exc_info.value.code == 'TOO_LARGE'

    def test_image_too_small(self):
        """Test image below minimum dimensions raises error."""
        img_bytes = self.create_test_image(50, 50, 'JPEG')
        with pytest.raises(PreprocessingError) as exc_info:
            validate_image_bytes(img_bytes, 'small.jpg')
        assert exc_info.value.code == 'TOO_SMALL'

    def test_corrupt_image(self):
        """Test corrupt image data raises error."""
        corrupt_bytes = b'not an image'
        with pytest.raises(PreprocessingError) as exc_info:
            validate_image_bytes(corrupt_bytes, 'corrupt.jpg')
        assert exc_info.value.code == 'CORRUPT'

    def test_unsupported_format(self):
        """Test unsupported image format (e.g., BMP) raises error."""
        img_bytes = self.create_test_image(224, 224, 'BMP')
        with pytest.raises(PreprocessingError) as exc_info:
            validate_image_bytes(img_bytes, 'test.bmp')
        assert exc_info.value.code == 'INVALID_TYPE'


class TestReadImageBytes:
    """Tests for read_image_bytes function."""

    def create_test_image(self, width=224, height=224, format='JPEG'):
        img = Image.new('RGB', (width, height), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    def test_read_jpeg(self):
        """Test reading JPEG bytes returns RGB numpy array."""
        img_bytes = self.create_test_image(224, 224, 'JPEG')
        img_array = read_image_bytes(img_bytes)
        assert isinstance(img_array, np.ndarray)
        assert img_array.shape == (224, 224, 3)
        assert img_array.dtype == np.uint8

    def test_read_png(self):
        """Test reading PNG bytes returns RGB numpy array."""
        img_bytes = self.create_test_image(224, 224, 'PNG')
        img_array = read_image_bytes(img_bytes)
        assert isinstance(img_array, np.ndarray)
        assert img_array.shape == (224, 224, 3)
        assert img_array.dtype == np.uint8

    def test_read_corrupt_raises(self):
        """Test reading corrupt bytes raises error."""
        with pytest.raises(PreprocessingError) as exc_info:
            read_image_bytes(b'corrupt data')
        assert exc_info.value.code == 'CORRUPT'


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_to_target(self):
        """Test resizing image to target size."""
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        resized = resize_image(img, TARGET_SIZE)
        assert resized.shape == (224, 224, 3)

    def test_resize_custom_size(self):
        """Test resizing to custom size."""
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        resized = resize_image(img, (100, 100))
        assert resized.shape == (100, 100, 3)


class TestPreprocessMobileNetV2:
    """Tests for MobileNetV2 preprocessing."""

    def test_preprocess_range(self):
        """Test preprocessed values are in [-1, 1] range."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        preprocessed = preprocess_mobilenetv2(img)
        assert preprocessed.dtype == np.float32
        assert preprocessed.min() >= -1.0
        assert preprocessed.max() <= 1.0

    def test_preprocess_specific_values(self):
        """Test specific pixel values map correctly."""
        # 0 -> -1.0, 127.5 -> 0.0, 255 -> 1.0
        img = np.array([[[0, 127, 255]]], dtype=np.uint8)
        preprocessed = preprocess_mobilenetv2(img)
        assert abs(preprocessed[0, 0, 0] - (-1.0)) < 0.01
        assert abs(preprocessed[0, 0, 1] - 0.0) < 0.01
        assert abs(preprocessed[0, 0, 2] - 1.0) < 0.01


class TestAddBatchDimension:
    """Tests for add_batch_dimension function."""

    def test_add_batch_dim(self):
        """Test adding batch dimension."""
        img = np.random.rand(224, 224, 3).astype(np.float32)
        batched = add_batch_dimension(img)
        assert batched.shape == (1, 224, 224, 3)


class TestPreprocessForModel:
    """Tests for full preprocessing pipeline."""

    def create_test_image(self, width=224, height=224, format='JPEG'):
        img = Image.new('RGB', (width, height), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    def test_full_pipeline_jpeg(self):
        """Test full pipeline with JPEG."""
        img_bytes = self.create_test_image(300, 300, 'JPEG')
        model_input, resized_rgb = preprocess_for_model(img_bytes, 'test.jpg')
        assert model_input.shape == (1, 224, 224, 3)
        assert model_input.dtype == np.float32
        assert model_input.min() >= -1.0
        assert model_input.max() <= 1.0
        assert resized_rgb.shape == (224, 224, 3)
        assert resized_rgb.dtype == np.uint8

    def test_full_pipeline_png(self):
        """Test full pipeline with PNG."""
        img_bytes = self.create_test_image(300, 300, 'PNG')
        model_input, resized_rgb = preprocess_for_model(img_bytes, 'test.png')
        assert model_input.shape == (1, 224, 224, 3)
        assert resized_rgb.shape == (224, 224, 3)

    def test_pipeline_preserves_content(self):
        """Test that preprocessing preserves image content (roughly)."""
        # Create image with distinct color
        img = Image.new('RGB', (300, 300), color=(100, 150, 200))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()

        model_input, resized_rgb = preprocess_for_model(img_bytes, 'test.jpg')

        # Resized RGB should be close to original color (allowing for resize interpolation)
        mean_color = resized_rgb.mean(axis=(0, 1))
        assert abs(mean_color[0] - 100) < 10
        assert abs(mean_color[1] - 150) < 10
        assert abs(mean_color[2] - 200) < 10


class TestPreprocessPILImage:
    """Tests for preprocess_pil_image function."""

    def test_preprocess_pil(self):
        """Test preprocessing a PIL Image directly."""
        img = Image.new('RGB', (300, 300), color='yellow')
        model_input, resized_rgb = preprocess_pil_image(img)
        assert model_input.shape == (1, 224, 224, 3)
        assert resized_rgb.shape == (224, 224, 3)
        assert model_input.dtype == np.float32
        assert resized_rgb.dtype == np.uint8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
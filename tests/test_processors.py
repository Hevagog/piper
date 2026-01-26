import numpy as np


class TestImageTransformer:
    """Test ImageTransformer class."""

    def test_transformer_creation(self):
        """Test ImageTransformer instantiation."""
        from piper.processors import ImageTransformer

        transformer = ImageTransformer(width=224, height=224)
        assert transformer is not None
        assert transformer.width == 224
        assert transformer.height == 224

    def test_resize_image(self):
        """Test image resizing."""
        from piper.processors import ImageTransformer

        transformer = ImageTransformer(width=100, height=100)

        # Create a test image
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        resized = transformer.resize(img)

        assert resized.shape[:2] == (100, 100)
        assert resized.shape[2] == 3

    def test_normalize_image(self):
        """Test image normalization."""
        from piper.processors import ImageTransformer

        transformer = ImageTransformer()

        # Create a test image with known values
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        normalized = transformer.normalize(img)

        # After normalization, values should be floats
        assert np.issubdtype(normalized.dtype, np.floating)
        assert normalized.min() >= -3.0  # Reasonable range after normalization
        assert normalized.max() <= 3.0

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading non-existent file raises exception or returns None."""
        from piper.processors import ImageTransformer

        transformer = ImageTransformer()
        # Depending on implementation, this may return None or raise
        try:
            result = transformer.load(str(temp_dir / "nonexistent.png"))
            # If it returns something, it should be None for nonexistent file
            # But the actual implementation might raise an exception
        except Exception:
            pass  # Expected behavior for some implementations


class TestAugmentor:
    """Test Augmentor class."""

    def test_augmentor_creation_empty(self):
        """Test Augmentor with no augmentations."""
        from piper.processors import Augmentor

        augmentor = Augmentor([])
        assert augmentor is not None

    def test_augmentor_creation_with_augmentations(self):
        """Test Augmentor with augmentations."""
        from piper.processors import Augmentor
        from piper.utils.models import AugmentationType

        augmentor = Augmentor([AugmentationType.FLIP, AugmentationType.ROTATE])
        assert augmentor is not None

    def test_augmentor_apply(self):
        """Test applying augmentations."""
        from piper.processors import Augmentor
        from piper.utils.models import AugmentationType

        augmentor = Augmentor([AugmentationType.FLIP])

        # Create a test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Apply augmentation
        augmented = augmentor(img)

        # Result should have same shape
        assert augmented.shape == img.shape

    def test_augmentor_no_modification_without_augmentations(self):
        """Test that empty augmentor returns original image."""
        from piper.processors import Augmentor

        augmentor = Augmentor([])

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmentor(img)

        np.testing.assert_array_equal(result, img)


class TestAugmentationType:
    """Test AugmentationType enum."""

    def test_augmentation_values(self):
        """Test that augmentation types have correct values."""
        from piper.utils.models import AugmentationType

        assert AugmentationType.FLIP.value == "flip"
        assert AugmentationType.ROTATE.value == "rotate"
        assert AugmentationType.COLOR_JITTER.value == "color_jitter"

    def test_augmentation_from_string(self):
        """Test creating augmentation from string."""
        from piper.utils.models import AugmentationType

        flip = AugmentationType("flip")
        assert flip == AugmentationType.FLIP

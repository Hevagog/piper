import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample test image."""
    import numpy as np

    # Create a simple 100x100 RGB image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    img_path = temp_dir / "test_image.png"

    try:
        import cv2

        cv2.imwrite(str(img_path), img)
    except ImportError:
        from PIL import Image

        Image.fromarray(img).save(img_path)

    return img_path


@pytest.fixture
def sample_images_dir(temp_dir: Path) -> Path:
    """Create a directory with sample test images."""
    import numpy as np

    images_dir = temp_dir / "images"
    images_dir.mkdir()

    # Create a few test images
    for i in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = images_dir / f"image_{i}.png"

        try:
            import cv2

            cv2.imwrite(str(img_path), img)
        except ImportError:
            from PIL import Image

            Image.fromarray(img).save(img_path)

    return images_dir

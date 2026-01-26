import os

import cv2
import numpy as np


class ImageTransformer:
    """Stateless image transformation utilities.

    Provides methods for resizing, normalizing, and converting images.
    All methods are designed to work with numpy arrays (OpenCV format).

    Args:
        width: Target width for resize operations.
        height: Target height for resize operations.

    Example:
        >>> transformer = ImageTransformer(width=224, height=224)
        >>> img = cv2.imread("image.jpg")
        >>> resized = transformer.resize(img)
        >>> normalized = transformer.normalize(resized)
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, width: int = 224, height: int = 224):
        self.width = width
        self.height = height

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions.

        Uses INTER_AREA interpolation for downscaling (best quality)
        and INTER_LINEAR for upscaling.

        Args:
            img: Input image as numpy array (H, W, C).

        Returns:
            Resized image as numpy array.
        """

        h, w = img.shape[:2]
        if h > self.height or w > self.width:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        return cv2.resize(img, (self.width, self.height), interpolation=interpolation)

    def normalize(
        self,
        img: np.ndarray,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Normalize image using mean and standard deviation.

        Args:
            img: Input image as numpy array (H, W, C) in [0, 255] range.
            mean: Per-channel mean values. Defaults to ImageNet mean.
            std: Per-channel standard deviation. Defaults to ImageNet std.

        Returns:
            Normalized image as float32 numpy array.
        """
        mean = mean or self.IMAGENET_MEAN
        std = std or self.IMAGENET_STD

        img = img.astype(np.float32) / 255.0
        img = (img - np.array(mean)) / np.array(std)
        return img

    def to_rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB.

        Args:
            img: Input image in BGR format.

        Returns:
            Image in RGB format.
        """

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def to_bgr(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB image to BGR.

        Args:
            img: Input image in RGB format.

        Returns:
            Image in BGR format.
        """

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def to_chw(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (2, 0, 1))

    def to_hwc(self, img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (1, 2, 0))

    @staticmethod
    def load(path: str) -> np.ndarray:
        """Load an image from disk.

        Args:
            path: Path to image file.

        Returns:
            Image as numpy array in BGR format.

        Raises:
            ValueError: If image cannot be loaded.
        """

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return img

    @staticmethod
    def save(
        img: np.ndarray,
        path: str,
        quality: int = 95,
    ) -> bool:
        """Save an image to disk.

        Args:
            img: Image as numpy array.
            path: Output path.
            quality: JPEG quality (0-100). Only used for JPEG files.

        Returns:
            True if save was successful.
        """

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            return cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imwrite(path, img)

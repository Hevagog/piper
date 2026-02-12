from random import Random

import cv2
import numpy as np

from piper.utils import AugmentationType


class Augmentor:
    """Image augmentation pipeline.

    Args:
        augmentations: List of augmentation types to apply.
        seed: Random seed for reproducibility. If None, uses random state.

    Example:
        >>> augmentor = Augmentor([AugmentationType.FLIP, AugmentationType.ROTATE])
        >>> augmented = augmentor(img)
        >>>
        >>> # Or apply specific augmentations:
        >>> flipped = Augmentor.flip(img, mode="horizontal")
    """

    def __init__(
        self,
        augmentations: list[AugmentationType] | None = None,
        seed: int | None = None,
    ):
        self.augmentations = augmentations or []
        self.rng = Random(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply all configured augmentations to an image.

        Args:
            img: Input image as numpy array.

        Returns:
            Augmented image.
        """
        for aug in self.augmentations:
            img = self.apply(img, aug)
        return img

    def apply(self, img: np.ndarray, augmentation: AugmentationType) -> np.ndarray:
        """Apply a single augmentation to an image.

        Args:
            img: Input image as numpy array.
            augmentation: Type of augmentation to apply.

        Returns:
            Augmented image.
        """
        match augmentation:
            case AugmentationType.FLIP:
                return self.flip(
                    img, mode=self.rng.choice(["horizontal", "vertical", "both"])
                )
            case AugmentationType.ROTATE:
                return self.rotate(img, angle=self.rng.uniform(-15, 15))
            case AugmentationType.COLOR_JITTER:
                return self.color_jitter(
                    img,
                    alpha=self.rng.uniform(0.8, 1.2),
                    beta=self.rng.randint(-10, 10),
                )
            case AugmentationType.RANDOM_CROP:
                h, w = img.shape[:2]
                return self.random_crop(
                    img,
                    crop_width=int(w * 0.8),
                    crop_height=int(h * 0.8),
                    rng=self.rng,
                )
            case AugmentationType.CENTER_CROP:
                h, w = img.shape[:2]
                return self.center_crop(
                    img,
                    crop_width=int(w * 0.8),
                    crop_height=int(h * 0.8),
                )
            case _:
                return img

    @staticmethod
    def flip(img: np.ndarray, mode: str = "horizontal") -> np.ndarray:
        """Flip an image.

        Args:
            img: Input image.
            mode: Flip mode - "horizontal", "vertical", or "both".

        Returns:
            Flipped image.
        """

        if mode == "horizontal":
            return cv2.flip(img, 1)
        elif mode == "vertical":
            return cv2.flip(img, 0)
        elif mode == "both":
            return cv2.flip(img, -1)
        return img

    @staticmethod
    def rotate(img: np.ndarray, angle: float = 15.0) -> np.ndarray:
        """Rotate an image around its center.

        Args:
            img: Input image.
            angle: Rotation angle in degrees. Positive is counter-clockwise.

        Returns:
            Rotated image.
        """

        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))

    @staticmethod
    def color_jitter(
        img: np.ndarray,
        alpha: float = 1.2,
        beta: int = 10,
    ) -> np.ndarray:
        """Apply color jitter (brightness/contrast adjustment).

        Uses the formula: output = alpha * input + beta

        Args:
            img: Input image.
            alpha: Contrast control (1.0 = no change).
            beta: Brightness control (0 = no change).

        Returns:
            Color-adjusted image.
        """

        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def random_crop(
        img: np.ndarray,
        crop_width: int,
        crop_height: int,
        rng: Random | None = None,
    ) -> np.ndarray:
        """Randomly crop a region from the image.

        Args:
            img: Input image.
            crop_width: Width of the crop.
            crop_height: Height of the crop.
            rng: Random number generator for reproducibility.

        Returns:
            Cropped image.
        """
        rng = rng or Random()
        h, w = img.shape[:2]

        max_x = max(0, w - crop_width)
        max_y = max(0, h - crop_height)

        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0

        return img[y : y + crop_height, x : x + crop_width]

    @staticmethod
    def center_crop(
        img: np.ndarray,
        crop_width: int,
        crop_height: int,
    ) -> np.ndarray:
        """Crop the center region of an image.

        Args:
            img: Input image.
            crop_width: Width of the crop.
            crop_height: Height of the crop.

        Returns:
            Center-cropped image.
        """
        h, w = img.shape[:2]

        x = max(0, (w - crop_width) // 2)
        y = max(0, (h - crop_height) // 2)

        return img[y : y + crop_height, x : x + crop_width]

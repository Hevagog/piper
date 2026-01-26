from collections.abc import Iterator
from pathlib import Path

from piper.logger import get_logger
from piper.sources.base import BaseDataSource, ImageInfo


class HuggingFaceSource(BaseDataSource):
    """Data source adapter for HuggingFace Datasets.

    Downloads and prepares image datasets from the HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "cifar10", "imagenet-1k").
        split: Dataset split to use (default: "train").
        subset: Optional dataset subset/configuration name.
        image_column: Name of the column containing images (default: "image").
        label_column: Name of the column containing labels (default: "label").

    Example:
        >>> source = HuggingFaceSource("cifar10", split="train")
        >>> path = source.download(Path("/data/raw"))
        >>> for img in source.iter_images():
        ...     print(img.path, img.label)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        image_column: str = "image",
        label_column: str = "label",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.image_column = image_column
        self.label_column = label_column
        self._logger = get_logger(__name__)
        self._label_names: list[str] | None = None

    @property
    def name(self) -> str:
        """Return the dataset identifier."""
        name = self.dataset_name.replace("/", "_")
        if self.subset:
            name = f"{name}_{self.subset}"
        return f"{name}_{self.split}"

    def download(self, destination: Path) -> Path:
        """Download and save HuggingFace dataset as image files.

        Saves images maintaining a class-based directory structure
        compatible with ImageFolder-style loading.

        Args:
            destination: Directory where data should be stored.

        Returns:
            Path to the root of the saved data.

        Raises:
            RuntimeError: If download or processing fails.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets package not installed. "
                "Install with: pip install piper[huggingface]"
            ) from e

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        try:
            self._logger.info(
                f"Loading HuggingFace dataset: {self.dataset_name} "
                f"(subset={self.subset}, split={self.split})"
            )

            dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                trust_remote_code=True,
            )

            # Extract label names if available
            if self.label_column in dataset.features:
                feature = dataset.features[self.label_column]
                if hasattr(feature, "names"):
                    self._label_names = feature.names
                    if self._label_names:
                        self._logger.info(f"Found {len(self._label_names)} classes")

            self._logger.info(f"Saving {len(dataset)} images to {destination}")
            self._save_images(dataset, destination)
            self._logger.info("Dataset preparation complete.")

            self._set_last_download_path(destination)
            return destination

        except Exception as e:
            self._logger.error(f"Failed to load/save HuggingFace dataset: {e}")
            raise RuntimeError(f"HuggingFace download failed: {e}") from e

    def _save_images(self, dataset, destination: Path) -> None:
        """Save dataset images to disk with class-based directory structure."""
        for idx, sample in enumerate(dataset):
            # Get image
            image = sample.get(self.image_column)
            if image is None:
                self._logger.warning(
                    f"Sample {idx}: No image found in column '{self.image_column}'"
                )
                continue

            # Get label
            label_value = sample.get(self.label_column)
            if label_value is not None:
                if self._label_names and isinstance(label_value, int):
                    label = self._label_names[label_value]
                else:
                    label = str(label_value)
            else:
                label = "unlabeled"

            # Create output path
            output_dir = destination / label
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{idx:08d}.jpg"

            # Save image
            if hasattr(image, "save"):
                # PIL Image
                image.save(output_path, "JPEG", quality=95)
            else:
                # Fallback for other formats
                import cv2
                import numpy as np

                if isinstance(image, np.ndarray):
                    cv2.imwrite(str(output_path), image)
                else:
                    self._logger.warning(
                        f"Sample {idx}: Unknown image type {type(image)}"
                    )

            if (idx + 1) % 1000 == 0:
                self._logger.info(f"Saved {idx + 1} images...")

    def iter_images(self, root: Path | None = None) -> Iterator[ImageInfo]:
        """Iterate over saved images with their labels."""
        yield from super().iter_images(root)

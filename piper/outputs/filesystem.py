from pathlib import Path
from typing import TYPE_CHECKING

from piper.logger import get_logger
from piper.outputs.base import BaseOutputWriter

if TYPE_CHECKING:
    import numpy as np


class FilesystemWriter(BaseOutputWriter):
    """Write processed images to the filesystem.

    Maintains directory structure and supports various image formats.
    This is the default output format, compatible with ImageFolder-style
    dataset loaders in PyTorch, JAX, and Rust Burn.

    Args:
        output_dir: Base directory for output files.
        quality: JPEG quality for .jpg/.jpeg files (0-100).
        create_metadata: Whether to create a metadata.json file.

    Example:
        >>> writer = FilesystemWriter(Path("/data/processed"))
        >>> writer.write(img, "train/cats/image001.jpg", label="cats")
        >>> writer.finalize()
    """

    def __init__(
        self,
        output_dir: Path | str,
        quality: int = 95,
        create_metadata: bool = True,
    ):
        super().__init__(output_dir)
        self.quality = quality
        self.create_metadata = create_metadata
        self._logger = get_logger(__name__)
        self._image_count = 0
        self._labels: set[str] = set()

    def write(
        self,
        img: "np.ndarray",
        relative_path: Path | str,
        label: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Write an image to the filesystem.

        Args:
            img: Image data as numpy array (BGR format).
            relative_path: Relative path for the output file.
            label: Optional class label (for metadata tracking).
            metadata: Optional additional metadata (not used for filesystem).
        """
        import cv2

        relative_path = Path(relative_path)
        output_path = self.output_dir / relative_path

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write image
        ext = output_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            success = cv2.imwrite(
                str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
        elif ext == ".png":
            success = cv2.imwrite(
                str(output_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6]
            )
        else:
            success = cv2.imwrite(str(output_path), img)

        if not success:
            self._logger.warning(f"Failed to write image: {output_path}")
            return

        self._image_count += 1
        if label:
            self._labels.add(label)

        if self._image_count % 1000 == 0:
            self._logger.info(f"Written {self._image_count} images...")

    def finalize(self) -> None:
        """Write metadata file and log summary."""
        if self.create_metadata:
            self._write_metadata()

        self._logger.info(
            f"Finished writing {self._image_count} images "
            f"({len(self._labels)} classes) to {self.output_dir}"
        )

    def _write_metadata(self) -> None:
        """Write dataset metadata to a JSON file."""
        import json

        metadata = {
            "num_images": self._image_count,
            "num_classes": len(self._labels),
            "classes": sorted(self._labels),
            "format": "ImageFolder",
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._logger.info(f"Written metadata to {metadata_path}")

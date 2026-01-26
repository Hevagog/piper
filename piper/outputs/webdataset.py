import io
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from piper.logger import get_logger
from piper.outputs.base import BaseOutputWriter

if TYPE_CHECKING:
    import numpy as np


class WebDatasetWriter(BaseOutputWriter):
    """Write processed images to WebDataset format (tar shards).

    WebDataset is an efficient format for training at scale, supporting:
    - Streaming access (no need to download entire dataset)
    - Sharding for parallel loading
    - Compatible with PyTorch (native), JAX (via grain/webdataset)

    Args:
        output_dir: Base directory for shard files.
        shard_size: Number of samples per shard.
        shard_pattern: Pattern for shard filenames.

    Example:
        >>> writer = WebDatasetWriter(Path("/data/shards"), shard_size=1000)
        >>> writer.write(img, "image001.jpg", label="cats")
        >>> writer.finalize()  # Important: flushes final shard
    """

    def __init__(
        self,
        output_dir: Path | str,
        shard_size: int = 1000,
        shard_pattern: str = "shard-{:06d}.tar",
    ):
        super().__init__(output_dir)
        self.shard_size = shard_size
        self.shard_pattern = shard_pattern
        self._logger = get_logger(__name__)

        self._current_shard: list[dict] = []
        self._shard_index = 0
        self._total_samples = 0
        self._labels: set[str] = set()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        img: "np.ndarray",
        relative_path: Path | str,
        label: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Add an image to the current shard.

        Args:
            img: Image data as numpy array (BGR format).
            relative_path: Relative path (used for key generation).
            label: Optional class label.
            metadata: Optional additional metadata.
        """
        import cv2

        relative_path = Path(relative_path)

        # Encode image to bytes
        ext = relative_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == ".png":
            _, encoded = cv2.imencode(".png", img)
        else:
            _, encoded = cv2.imencode(".jpg", img)
            ext = ".jpg"

        # Generate sample key
        key = f"{self._total_samples:08d}"

        # Build sample data
        sample_data = {
            "key": key,
            "image_data": encoded.tobytes(),
            "image_ext": ext.lstrip("."),
            "label": label,
            "original_path": str(relative_path),
            "metadata": metadata or {},
        }

        self._current_shard.append(sample_data)
        self._total_samples += 1

        if label:
            self._labels.add(label)

        # Flush shard if full
        if len(self._current_shard) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self) -> None:
        """Write current shard to disk."""
        if not self._current_shard:
            return

        shard_name = self.shard_pattern.format(self._shard_index)
        shard_path = self.output_dir / shard_name

        with tarfile.open(shard_path, "w") as tar:
            for sample in self._current_shard:
                key = sample["key"]

                # Add image file
                img_name = f"{key}.{sample['image_ext']}"
                img_data = sample["image_data"]
                img_info = tarfile.TarInfo(name=img_name)
                img_info.size = len(img_data)
                tar.addfile(img_info, io.BytesIO(img_data))

                # Add label file (cls extension for webdataset)
                if sample["label"] is not None:
                    cls_name = f"{key}.cls"
                    cls_data = sample["label"].encode("utf-8")
                    cls_info = tarfile.TarInfo(name=cls_name)
                    cls_info.size = len(cls_data)
                    tar.addfile(cls_info, io.BytesIO(cls_data))

                # Add metadata file
                if sample["metadata"]:
                    json_name = f"{key}.json"
                    json_data = json.dumps(sample["metadata"]).encode("utf-8")
                    json_info = tarfile.TarInfo(name=json_name)
                    json_info.size = len(json_data)
                    tar.addfile(json_info, io.BytesIO(json_data))

        self._logger.info(
            f"Written shard {shard_name} with {len(self._current_shard)} samples"
        )

        self._current_shard = []
        self._shard_index += 1

    def finalize(self) -> None:
        """Flush remaining samples and write index file."""
        # Flush any remaining samples
        if self._current_shard:
            self._flush_shard()

        # Write index file
        index = {
            "num_shards": self._shard_index,
            "num_samples": self._total_samples,
            "num_classes": len(self._labels),
            "classes": sorted(self._labels),
            "shard_pattern": self.shard_pattern,
            "samples_per_shard": self.shard_size,
        }

        index_path = self.output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        self._logger.info(
            f"Finished writing {self._total_samples} samples "
            f"to {self._shard_index} shards"
        )

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class ImageInfo:
    """Metadata about an image in the dataset.

    Attributes:
        path: Path to the image file.
        label: Optional class/category label.
        split: Dataset split (train, val, test).
        metadata: Additional metadata as key-value pairs.
    """

    path: Path
    label: str | None = None
    split: str = "train"
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class DataSource(Protocol):
    """Protocol defining the interface for dataset sources.

    All data source implementations must provide these methods to be
    compatible with the piper pipeline.
    """

    @property
    def name(self) -> str:
        """Return the dataset identifier/name."""
        ...

    def download(self, destination: Path) -> Path:
        """Download/prepare data to the destination directory.

        Args:
            destination: Directory where data should be stored.

        Returns:
            Path to the root of the downloaded/prepared data.
        """
        ...

    def iter_images(self, root: Path | None = None) -> Iterator[ImageInfo]:
        """Iterate over all images in the dataset.

        Args:
            root: Optional root directory to search. If None, uses
                  the last download destination.

        Yields:
            ImageInfo objects for each image found.
        """
        ...


class BaseDataSource(ABC):
    """Abstract base class for data sources with common functionality.

    Provides shared implementation details for data source adapters.
    """

    _last_download_path: Path | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset identifier/name."""
        ...

    @abstractmethod
    def download(self, destination: Path) -> Path:
        """Download/prepare data to the destination directory."""
        ...

    def iter_images(self, root: Path | None = None) -> Iterator[ImageInfo]:
        """Iterate over all images in the dataset.

        Default implementation walks the directory tree and yields
        ImageInfo for each image file found.
        """
        from piper.utils.constants import IMAGE_FORMATS

        search_root = root or self._last_download_path
        if search_root is None:
            raise ValueError(
                "No root directory specified and no previous download path available. "
                "Call download() first or provide a root path."
            )

        for img_path in search_root.rglob("*"):
            if img_path.suffix.lower() in IMAGE_FORMATS:
                # Try to extract label from parent directory name
                label = img_path.parent.name if img_path.parent != search_root else None
                yield ImageInfo(path=img_path, label=label)

    def _set_last_download_path(self, path: Path) -> None:
        """Store the last download path for later use."""
        self._last_download_path = path

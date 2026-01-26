from collections.abc import Iterator
from pathlib import Path

from piper.logger import get_logger
from piper.sources.base import BaseDataSource, ImageInfo


class LocalSource(BaseDataSource):
    """Data source adapter for local filesystem directories.

    Use this when your images are already available locally and
    don't need to be downloaded.

    Args:
        path: Path to the root directory containing images.
        recursive: Whether to search subdirectories (default: True).

    Example:
        >>> source = LocalSource(Path("/path/to/images"))
        >>> for img in source.iter_images():
        ...     print(img.path, img.label)
    """

    def __init__(self, path: Path | str, recursive: bool = True):
        self.path = Path(path)
        self.recursive = recursive
        self._logger = get_logger(__name__)

    @property
    def name(self) -> str:
        """Return the directory name as identifier."""
        return self.path.name

    def download(self, destination: Path) -> Path:
        """For local sources, just validates the path exists.

        If destination differs from source, optionally copies or
        creates a symlink. By default, just returns the source path.

        Args:
            destination: Not used for local sources.

        Returns:
            Path to the source data.

        Raises:
            FileNotFoundError: If the source path doesn't exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Local source path does not exist: {self.path}")

        self._logger.info(f"Using local source: {self.path}")
        self._set_last_download_path(self.path)
        return self.path

    def iter_images(self, root: Path | None = None) -> Iterator[ImageInfo]:
        """Iterate over all images in the local directory.

        Args:
            root: Optional override for the search root.

        Yields:
            ImageInfo objects for each image found.
        """
        from piper.utils.constants import IMAGE_FORMATS

        search_root = root or self.path

        if not search_root.exists():
            raise FileNotFoundError(f"Path does not exist: {search_root}")

        if self.recursive:
            pattern_iter = search_root.rglob("*")
        else:
            pattern_iter = search_root.glob("*")

        for img_path in pattern_iter:
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_FORMATS:
                # Try to extract label from parent directory name
                label = img_path.parent.name if img_path.parent != search_root else None
                yield ImageInfo(path=img_path, label=label)

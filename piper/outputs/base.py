from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class OutputWriter(Protocol):
    """Protocol defining the interface for output format writers.

    Output writers handle the storage of processed images in various formats.
    """

    def write(
        self,
        img: "np.ndarray",
        relative_path: Path | str,
        label: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Write a processed image.

        Args:
            img: Image data as numpy array.
            relative_path: Relative path for the image (used for structure).
            label: Optional class label.
            metadata: Optional additional metadata.
        """
        ...

    def finalize(self) -> None:
        """Finalize the output (flush buffers, close files, etc.)."""
        ...


class BaseOutputWriter(ABC):
    """Abstract base class for output writers with common functionality."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    @abstractmethod
    def write(
        self,
        img: "np.ndarray",
        relative_path: Path | str,
        label: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Write a processed image."""
        ...

    def finalize(self) -> None:
        """Finalize the output. Override in subclasses if needed."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - calls finalize."""
        self.finalize()
        return False

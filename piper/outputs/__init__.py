from typing import TYPE_CHECKING

from piper.outputs.base import OutputWriter
from piper.outputs.filesystem import FilesystemWriter

if TYPE_CHECKING:
    from piper.outputs.webdataset import WebDatasetWriter


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "WebDatasetWriter":
        from piper.outputs.webdataset import WebDatasetWriter

        return WebDatasetWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OutputWriter",
    "FilesystemWriter",
    "WebDatasetWriter",
]

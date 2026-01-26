from typing import TYPE_CHECKING

from piper.sources.base import BaseDataSource, DataSource, ImageInfo
from piper.sources.local import LocalSource

if TYPE_CHECKING:
    from piper.sources.huggingface import HuggingFaceSource
    from piper.sources.kaggle import KaggleSource


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "KaggleSource":
        from piper.sources.kaggle import KaggleSource

        return KaggleSource
    if name == "HuggingFaceSource":
        from piper.sources.huggingface import HuggingFaceSource

        return HuggingFaceSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDataSource",
    "DataSource",
    "ImageInfo",
    "LocalSource",
    "KaggleSource",
    "HuggingFaceSource",
]

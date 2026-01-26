"""Piper - Distributed Image Processing Pipeline.

Example:
    >>> from piper import Pipeline, PiperConfig
    >>> from piper.utils.models import AugmentationType
    >>>
    >>> # Simple usage
    >>> Pipeline().from_kaggle("username/dataset").to_filesystem().run()
    >>>
    >>> # With configuration
    >>> config = PiperConfig(
    ...     target_width=512,
    ...     target_height=512,
    ...     augmentations=[AugmentationType.FLIP]
    ... )
    >>> Pipeline(config).from_local("./images").run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from piper import outputs, processors, sources
from piper.api import Pipeline
from piper.config import PiperConfig
from piper.utils.models import AugmentationType

if TYPE_CHECKING:
    from piper import luigi_tasks


def __getattr__(name: str):
    """Lazy loading for optional dependency submodules."""
    if name == "luigi_tasks":
        try:
            import importlib

            return importlib.import_module("piper.luigi_tasks")
        except ImportError as e:
            raise ImportError(
                "Luigi tasks require the 'luigi' package. "
                "Install with: pip install piper[luigi]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Pipeline",
    "PiperConfig",
    "AugmentationType",
    "sources",
    "processors",
    "outputs",
    "luigi_tasks",
]

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


def login():
    print("\n AUTHENTICATION MENU")
    print("1. Hugging Face")
    print("2. Kaggle")

    while True:
        choice = input("Select an option (1/2 or q to exit): ").strip()
        match choice:
            case "1":
                from huggingface_hub import login as hf_login

                hf_login()
                break
            case "2":
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi

                    KaggleApi().authenticate()
                    print("Kaggle authentication successful!")
                except Exception:
                    print("Authentication failed or credentials not found.")
                    print("Choose setup method:")
                    print("   [A] Manually enter Username & API Key")
                    print("   [B] Import an existing 'kaggle.json' file")
                    import json
                    import os
                    import shutil
                    from pathlib import Path

                    setup_choice = input("Select [A/B]: ").strip().upper()
                    kaggle_dir = Path.home() / ".config" / "kaggle"
                    target_file = kaggle_dir / "kaggle.json"

                    kaggle_dir.mkdir(exist_ok=True)
                    if setup_choice == "A":
                        k_user = input("Enter Kaggle Username: ").strip()
                        k_key = input("Enter Kaggle API Key: ").strip()

                        data = {"username": k_user, "key": k_key}

                        with open(target_file, "w") as f:
                            json.dump(data, f)
                        print(f"Credentials saved to {target_file}")

                    elif setup_choice == "B":
                        source_path = input(
                            "Enter path to your downloaded 'kaggle.json': "
                        ).strip()
                        source = Path(source_path)

                        if source.exists() and source.is_file():
                            shutil.copy(source, target_file)
                            print(f"File copied to {target_file}")
                        else:
                            print("File not found. Please try again.")
                            break

                    if os.name != "nt":  # If not Windows
                        target_file.chmod(0o600)

                    try:
                        api = KaggleApi()
                        api.authenticate()
                        break
                    except Exception as e:
                        print(f"Authentication Failed! {e}")
                        break
            case "q":
                break
            case _:
                ...


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
    "login",
]

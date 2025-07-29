_HARD_DEPENDENCIES = [
    ("luigi", "luigi"),
    ("kaggle.api.kaggle_api_extended", "kaggle"),
    ("PIL", "Pillow"),
]


def _check_dependencies():
    """Check that all required dependencies are available."""
    missing_dependencies = []

    for import_name, package_name in _HARD_DEPENDENCIES:
        try:
            __import__(import_name)
        except ImportError as e:
            missing_dependencies.append(
                f"Missing '{package_name}' (import as '{import_name}'): {e}"
            )

    if missing_dependencies:
        missing_packages = [
            pkg[1]
            for pkg in _HARD_DEPENDENCIES
            if any(pkg[0] in missing for missing in missing_dependencies)
        ]
        error_msg = (
            "Unable to import required dependencies:\n"
            + "\n".join(missing_dependencies)
            + "\n\nInstall missing packages with:\n"
            + f"  pip install {' '.join(missing_packages)}"
        )
        raise ImportError(error_msg)


# Check dependencies on import
_check_dependencies()

# Clean up module namespace
del _HARD_DEPENDENCIES, _check_dependencies

from .api import set_up_dataset  # noqa: E402

__version__ = "0.1.0"
__all__ = ["set_up_dataset"]

import shutil
import zipfile
from pathlib import Path

from piper.logger import get_logger
from piper.sources.base import BaseDataSource


class KaggleSource(BaseDataSource):
    """Data source adapter for Kaggle datasets.

    Downloads datasets from Kaggle using the Kaggle API.
    Requires authentication via ~/.kaggle/kaggle.json or
    KAGGLE_USERNAME/KAGGLE_KEY environment variables.

    Args:
        dataset_url: Kaggle dataset identifier in format "owner/dataset-name".

    Example:
        >>> source = KaggleSource("username/my-dataset")
        >>> path = source.download(Path("/data/raw"))
        >>> for img in source.iter_images():
        ...     print(img.path, img.label)
    """

    def __init__(self, dataset_url: str):
        self.dataset_url = dataset_url
        self._logger = get_logger(__name__)

    @property
    def name(self) -> str:
        """Return the dataset identifier."""
        return self.dataset_url.split("/")[-1]

    def download(self, destination: Path) -> Path:
        """Download and extract the Kaggle dataset.

        Handles single-root-directory zip archives by extracting
        contents directly, avoiding nested subdirectories.

        Args:
            destination: Directory where data should be stored.

        Returns:
            Path to the root of the extracted data.

        Raises:
            ImportError: If kaggle package is not installed.
            RuntimeError: If download or extraction fails.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise ImportError(
                "Kaggle package not installed. Install with: pip install piper[kaggle]"
            ) from e

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        zip_path = destination / f"{self.name}.zip"

        try:
            api = KaggleApi()
            api.authenticate()

            self._logger.info(f"Downloading dataset: {self.dataset_url} to {zip_path}")
            api.dataset_download_files(
                self.dataset_url, path=str(destination), unzip=False
            )
            self._logger.info("Download complete.")

            self._logger.info(f"Extracting {zip_path}...")
            self._extract_smart(zip_path, destination)
            self._logger.info("Extraction complete.")

            # Clean up zip file
            if zip_path.exists():
                zip_path.unlink()
                self._logger.info(f"Removed zip file: {zip_path}")

            self._set_last_download_path(destination)
            return destination

        except Exception as e:
            self._logger.error(f"Failed to download and extract dataset: {e}")
            # Clean up partial downloads
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"Kaggle download failed: {e}") from e

    def _extract_smart(self, zip_path: Path, destination: Path) -> None:
        """Extract zip file, handling single-root-directory archives.

        If the zip contains a single top-level directory, extracts its
        contents directly to avoid unnecessary nesting.
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.infolist()
            if not file_list:
                self._logger.warning(f"Zip file {zip_path} is empty.")
                return

            # Check if all members are in a single top-level directory
            first_member_parts = file_list[0].filename.split("/")
            root_dir: str | None = (
                first_member_parts[0] if len(first_member_parts) > 1 else None
            )

            if root_dir is None:
                self._logger.info("No single root directory. Extracting as-is.")
                zip_ref.extractall(destination)
                return

            is_single_root = all(
                m.filename.startswith(root_dir + "/") for m in file_list
            )

            if is_single_root:
                self._logger.info(
                    f"Single root directory '{root_dir}' detected. Stripping it."
                )
                prefix_len = len(root_dir) + 1
                for member in file_list:
                    # Skip the root directory entry itself
                    if member.filename == root_dir + "/":
                        continue
                    new_path = destination / member.filename[prefix_len:]
                    if member.is_dir():
                        new_path.mkdir(parents=True, exist_ok=True)
                    else:
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        with zip_ref.open(member, "r") as source:
                            with open(new_path, "wb") as target:
                                shutil.copyfileobj(source, target)
            else:
                self._logger.info("No single root directory. Extracting as-is.")
                zip_ref.extractall(destination)

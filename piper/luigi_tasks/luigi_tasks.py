import os
import shutil
import subprocess
from pathlib import Path

import luigi
from luigi.util import requires

from piper.config import PiperConfig
from piper.logger import get_logger


class GlobalConfig(luigi.Config):
    """Luigi configuration class for global pipeline settings."""

    target_width = luigi.IntParameter(default=224)
    target_height = luigi.IntParameter(default=224)
    shared_volume_dir = luigi.Parameter(default="/data")
    local_dir = luigi.Parameter(default="data")

    @property
    def raw_data_dir(self) -> str:
        return os.path.join(str(self.local_dir), "raw")

    @property
    def processed_data_dir(self) -> str:
        return os.path.join(str(self.shared_volume_dir), "processed")

    def to_piper_config(self) -> PiperConfig:
        """Convert to PiperConfig for use with new abstractions."""
        return PiperConfig(
            data_dir=Path(str(self.shared_volume_dir)),
            raw_subdir="raw",
            processed_subdir="processed",
            target_width=int(self.target_width),
            target_height=int(self.target_height),
        )


class DownloadDataset(luigi.Task):
    """Download a dataset from various sources.

    Supports Kaggle, HuggingFace, or local sources via the source_type parameter.

    Parameters:
        source_type: Type of data source ('kaggle', 'huggingface', 'local').
        source_id: Dataset identifier (Kaggle URL, HF dataset name, or path).
        hf_split: HuggingFace dataset split (default: 'train').
        hf_subset: HuggingFace dataset subset/configuration (optional).
    """

    source_type: str = luigi.ChoiceParameter(
        choices=["kaggle", "huggingface", "local"],
        default="kaggle",
    )
    source_id = luigi.Parameter()
    hf_split = luigi.Parameter(default="train")
    hf_subset = luigi.OptionalParameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                str(GlobalConfig().shared_volume_dir), ".dataset_download_complete"
            )
        )

    def run(self):
        """Download dataset using the appropriate source adapter."""
        raw_data_dir = Path(GlobalConfig().raw_data_dir)
        source = self._create_source()

        self._logger.info(f"Downloading from {self.source_type}: {self.source_id}")
        source.download(raw_data_dir)
        self._logger.info("Download complete.")

        with self.output().open("w") as f:
            f.write("download_complete\n")

    def _create_source(self):
        """Create the appropriate DataSource based on source_type."""
        match self.source_type:
            case "kaggle":
                from piper.sources.kaggle import KaggleSource

                return KaggleSource(str(self.source_id))
            case "huggingface":
                from piper.sources.huggingface import HuggingFaceSource

                return HuggingFaceSource(
                    str(self.source_id),
                    split=str(self.hf_split),
                    subset=str(self.hf_subset) if self.hf_subset else None,
                )
            case "local":
                from piper.sources.local import LocalSource

                return LocalSource(Path(str(self.source_id)))
            case _:
                raise ValueError(f"Unknown source type: {self.source_type}")


@requires(DownloadDataset)
class ListRawImages(luigi.Task):
    """Marker task indicating raw images are ready for processing.

    This task serves as a dependency checkpoint between download and processing.
    """

    source_type = luigi.ChoiceParameter(
        choices=["kaggle", "huggingface", "local"],
        default="kaggle",
    )
    source_id = luigi.Parameter()
    hf_split = luigi.Parameter(default="train")
    hf_subset = luigi.OptionalParameter(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                str(GlobalConfig().shared_volume_dir), ".raw_image_list_complete"
            )
        )

    def run(self):
        self._logger.info("Raw image list task completed - ready for processing")
        with self.output().open("w") as f:
            f.write("raw_image_list_generated\n")


@requires(ListRawImages)
class SparkProcessImages(luigi.Task):
    """Launch a PySpark job for distributed image processing.

    Processes images using Spark for parallel execution with configurable
    augmentations.

    Parameters:
        source_type: Type of data source.
        source_id: Dataset identifier.
        augmentations: Comma-separated list of augmentations (flip,rotate,color_jitter).
    """

    source_type = luigi.ChoiceParameter(
        choices=["kaggle", "huggingface", "local"],
        default="kaggle",
    )
    source_id = luigi.Parameter()
    hf_split = luigi.Parameter(default="train")
    hf_subset = luigi.OptionalParameter(default=None)
    augmentations = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(GlobalConfig().processed_data_dir, ".spark_augment_complete")
        )

    def run(self):
        """Launch the PySpark job for distributed image processing."""
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        package_name = os.path.basename(package_root)
        archive_path = os.path.join(
            os.path.dirname(package_root), f"{package_name}.zip"
        )

        self._logger.info(f"Creating archive of {package_root} at {archive_path}")
        shutil.make_archive(
            archive_path.replace(".zip", ""),
            "zip",
            root_dir=os.path.dirname(package_root),
            base_dir=package_name,
        )

        config = GlobalConfig()
        spark_script = os.path.join(package_root, "spark", "data_augment.py")
        cmd = [
            "spark-submit",
            "--py-files",
            archive_path,
            spark_script,
            "--input_dir",
            config.raw_data_dir,
            "--output_dir",
            config.processed_data_dir,
            "--width",
            str(config.target_width),
            "--height",
            str(config.target_height),
        ]

        if self.augmentations:
            cmd.extend(["--augmentations", str(self.augmentations)])

        self._logger.info(f"Launching Spark job: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._logger.info("Spark job completed successfully.")
            self._logger.debug(f"Spark job output: {result.stdout.strip()}")
            with self.output().open("w") as f:
                f.write("spark_augment_complete\n")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() if e.stderr else "Unknown error"
            self._logger.error(f"Spark job failed: {error_message}")
            self._logger.debug(f"Full stdout: {e.stdout.strip()}")
            raise RuntimeError(f"Spark job failed: {error_message}")
        finally:
            if os.path.exists(archive_path):
                os.remove(archive_path)
                self._logger.info(f"Removed temporary archive: {archive_path}")


class ProcessingPipeline(luigi.WrapperTask):
    """Run the complete image processing pipeline.

    Main entry point for Luigi-based orchestration. It runs:
    1. Dataset download (from Kaggle, HuggingFace, or local)
    2. Distributed image processing via Spark

    Parameters:
        source_type: Data source type ('kaggle', 'huggingface', 'local').
        source_id: Dataset identifier.
        hf_split: HuggingFace split (default: 'train').
        hf_subset: HuggingFace subset (optional).
        augmentations: Comma-separated augmentations.

    Example:
        luigi.build([ProcessingPipeline(
            source_type='kaggle',
            source_id='user/dataset',
            augmentations='flip,rotate',
        )])
    """

    source_type = luigi.ChoiceParameter(
        choices=["kaggle", "huggingface", "local"],
        default="kaggle",
    )
    source_id = luigi.Parameter()
    hf_split = luigi.Parameter(default="train")
    hf_subset = luigi.OptionalParameter(default=None)
    augmentations = luigi.Parameter(default="")

    def requires(self):
        yield SparkProcessImages(
            source_type=self.source_type,
            source_id=self.source_id,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            augmentations=self.augmentations,
        )

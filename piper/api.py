from __future__ import annotations

from pathlib import Path

from piper.config import PiperConfig
from piper.logger import get_logger
from piper.outputs.base import BaseOutputWriter
from piper.sources.base import DataSource


class Pipeline:
    """Main entry point for piper image processing pipelines.

    Args:
        config: Pipeline configuration. If None, uses default PiperConfig.

    Attributes:
        config: The pipeline configuration.

    Example:
        >>> pipeline = Pipeline()
        >>> pipeline.from_kaggle("user/dataset")
        >>> pipeline.to_filesystem("/data/processed")
        >>> pipeline.run()
    """

    def __init__(self, config: PiperConfig | None = None):
        self.config = config or PiperConfig()
        self._source: DataSource | None = None
        self._writer: BaseOutputWriter | None = None
        self._logger = get_logger(__name__)

    def from_kaggle(self, dataset_url: str) -> Pipeline:
        """Configure Kaggle as the data source.

        Args:
            dataset_url: Kaggle dataset identifier (e.g., "username/dataset-name").

        Returns:
            Self for method chaining.

        Raises:
            ImportError: If kaggle package is not installed.
        """
        from piper.sources.kaggle import KaggleSource

        self._source = KaggleSource(dataset_url)
        self._logger.info(f"Configured Kaggle source: {dataset_url}")
        return self

    def from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        image_column: str = "image",
        label_column: str = "label",
    ) -> Pipeline:
        """Configure HuggingFace Datasets as the data source.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split to use.
            subset: Optional subset/configuration name.
            image_column: Name of the image column.
            label_column: Name of the label column.

        Returns:
            Self for method chaining.

        Raises:
            ImportError: If datasets package is not installed.
        """
        from piper.sources.huggingface import HuggingFaceSource

        self._source = HuggingFaceSource(
            dataset_name,
            split=split,
            subset=subset,
            image_column=image_column,
            label_column=label_column,
        )
        self._logger.info(f"Configured HuggingFace source: {dataset_name} ({split})")
        return self

    def from_local(self, path: str | Path, recursive: bool = True) -> Pipeline:
        """Configure local filesystem as the data source.

        Args:
            path: Path to the directory containing images.
            recursive: Whether to search subdirectories.

        Returns:
            Self for method chaining.
        """
        from piper.sources.local import LocalSource

        self._source = LocalSource(Path(path), recursive=recursive)
        self._logger.info(f"Configured local source: {path}")
        return self

    def from_source(self, source: DataSource) -> Pipeline:
        """Configure a custom data source.

        Args:
            source: Any object implementing the DataSource protocol.

        Returns:
            Self for method chaining.
        """
        self._source = source
        self._logger.info(f"Configured custom source: {source.name}")
        return self

    def to_filesystem(self, output_dir: str | Path | None = None) -> Pipeline:
        """Configure filesystem output (ImageFolder format).

        Args:
            output_dir: Output directory. Defaults to config.processed_data_dir.

        Returns:
            Self for method chaining.
        """
        from piper.outputs.filesystem import FilesystemWriter

        output_path = Path(output_dir) if output_dir else self.config.processed_data_dir
        self._writer = FilesystemWriter(output_path)
        self._logger.info(f"Configured filesystem output: {output_path}")
        return self

    def to_webdataset(
        self,
        output_dir: str | Path,
        shard_size: int = 1000,
    ) -> Pipeline:
        """Configure WebDataset output (tar shards).

        Args:
            output_dir: Directory for shard files.
            shard_size: Number of samples per shard.

        Returns:
            Self for method chaining.
        """
        from piper.outputs.webdataset import WebDatasetWriter

        self._writer = WebDatasetWriter(Path(output_dir), shard_size=shard_size)
        self._logger.info(f"Configured WebDataset output: {output_dir}")
        return self

    def to_writer(self, writer: BaseOutputWriter) -> Pipeline:
        """Configure a custom output writer.

        Args:
            writer: Any object inheriting from BaseOutputWriter.

        Returns:
            Self for method chaining.
        """
        self._writer = writer
        self._logger.info("Configured custom output writer")
        return self

    def run(self, distributed: bool | None = None) -> None:
        """Execute the pipeline.

        Args:
            distributed: Whether to use Spark for distributed processing.
                        If None, uses config.use_spark setting.

        Raises:
            ValueError: If source or writer is not configured.
            RuntimeError: If processing fails.
        """
        if self._source is None:
            raise ValueError(
                "No data source configured. Use from_kaggle(), from_huggingface(), "
                "from_local(), or from_source() to configure a source."
            )

        if self._writer is None:
            # Default to filesystem writer
            self.to_filesystem()

        use_spark = distributed if distributed is not None else self.config.use_spark

        self._logger.info(
            f"Running pipeline: {self._source.name} -> "
            f"{'Spark' if use_spark else 'local'} processing"
        )

        # Download/prepare data
        try:
            raw_path = self._source.download(self.config.raw_data_dir)
        except OSError:
            self._logger.error(
                "Make sure to authenticate your kaggle account using piper.login() function!"
            )
            raise RuntimeError
        self._logger.info(f"Data prepared at: {raw_path}")

        # Process images
        if use_spark:
            self._run_spark(raw_path)
        else:
            self._run_local(raw_path)

    def _run_local(self, raw_path: Path) -> None:
        """Execute pipeline locally (single process)."""
        from piper.processors import Augmentor, ImageTransformer

        transformer = ImageTransformer(
            width=self.config.target_width,
            height=self.config.target_height,
        )
        augmentor = Augmentor(self.config.augmentations)

        processed_count = 0
        assert self._writer is not None
        assert self._source is not None

        with self._writer:
            for img_info in self._source.iter_images(raw_path):
                try:
                    img = transformer.load(str(img_info.path))
                    img = transformer.resize(img)

                    if self.config.augmentations:
                        img = augmentor(img)

                    try:
                        relative_path = img_info.path.relative_to(raw_path)
                    except ValueError:
                        relative_path = img_info.path.name

                    self._writer.write(
                        img,
                        relative_path,
                        label=img_info.label,
                        metadata=img_info.metadata,
                    )

                    processed_count += 1
                    if processed_count % 100 == 0:
                        self._logger.info(f"Processed {processed_count} images...")

                except Exception as e:
                    self._logger.warning(f"Failed to process {img_info.path}: {e}")

        self._logger.info(f"Pipeline complete: processed {processed_count} images")

    def _run_spark(self, raw_path: Path) -> None:
        """Execute pipeline using Spark for distributed processing."""
        import shutil
        import subprocess

        # Check if spark-submit is available
        if shutil.which("spark-submit") is None:
            raise RuntimeError(
                "Spark is not installed or 'spark-submit' is not in PATH. "
                "Either install Apache Spark, or use local processing by:\n"
                "  - Setting use_spark=False in PiperConfig\n"
                "  - Calling pipeline.run(distributed=False)\n"
                "  - Using --no-spark flag with CLI"
            )

        package_root = Path(__file__).parent
        package_name = package_root.name
        archive_path = package_root.parent / f"{package_name}.zip"

        self._logger.info(f"Creating Spark package archive: {archive_path}")
        shutil.make_archive(
            str(archive_path).replace(".zip", ""),
            "zip",
            root_dir=str(package_root.parent),
            base_dir=package_name,
        )

        spark_script = package_root / "spark" / "data_augment.py"
        cmd = [
            "spark-submit",
            "--py-files",
            str(archive_path),
            str(spark_script),
            "--input_dir",
            str(raw_path),
            "--output_dir",
            str(self.config.processed_data_dir),
            "--width",
            str(self.config.target_width),
            "--height",
            str(self.config.target_height),
        ]

        if self.config.spark_partitions > 0:
            cmd.extend(["--partition_count", str(self.config.spark_partitions)])

        if self.config.augmentations:
            aug_str = ",".join(a.value for a in self.config.augmentations)
            cmd.extend(["--augmentations", aug_str])

        self._logger.info(f"Launching Spark job: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._logger.info("Spark job completed successfully")
            self._logger.debug(f"Spark output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            self._logger.error(f"Spark job failed: {error_msg}")
            raise RuntimeError(f"Spark job failed: {error_msg}") from e
        finally:
            # Clean up archive
            if archive_path.exists():
                archive_path.unlink()
                self._logger.debug(f"Removed temporary archive: {archive_path}")

    def run_with_luigi(
        self,
        scheduler_host: str = "localhost",
        scheduler_port: int = 8082,
    ) -> bool:
        """Execute pipeline via Luigi scheduler.

        This method submits tasks to a Luigi scheduler for distributed
        orchestration. The scheduler must be running separately.

        Args:
            scheduler_host: Luigi scheduler hostname.
            scheduler_port: Luigi scheduler port.

        Returns:
            True if all tasks completed successfully.

        Raises:
            ImportError: If luigi package is not installed.
            ValueError: If source is not configured.
        """
        try:
            import luigi
        except ImportError as e:
            raise ImportError(
                "Luigi package not installed. Install with: pip install piper[luigi]"
            ) from e

        if self._source is None:
            raise ValueError("No data source configured")

        from piper.luigi_tasks import ProcessingPipeline

        # Determine source type and ID from the configured source
        source_type = type(self._source).__name__.lower().replace("source", "")
        source_id = getattr(
            self._source,
            "dataset_url",
            getattr(
                self._source, "dataset_name", str(getattr(self._source, "path", ""))
            ),
        )

        task = ProcessingPipeline(
            source_type=source_type,
            source_id=source_id,
        )

        success = luigi.build(
            [task],
            scheduler_host=scheduler_host,
            scheduler_port=scheduler_port,
        )

        return success

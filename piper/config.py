import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import platformdirs

from piper.utils.models import AugmentationType


@dataclass
class PiperConfig:
    """Central configuration for piper pipelines.

    Attributes:
        data_dir: Base directory for all data operations.
        raw_subdir: Subdirectory name for raw/downloaded data.
        processed_subdir: Subdirectory name for processed output.
        target_width: Target image width after processing.
        target_height: Target image height after processing.
        augmentations: List of augmentation types to apply.
        output_format: Output format for processed images.
        use_spark: Whether to use Spark for distributed processing.
        spark_partitions: Number of Spark partitions (0 = auto).
        preserve_structure: Whether to maintain directory structure in output.
    """

    data_dir: Path = field(
        default_factory=lambda: platformdirs.user_data_path(
            appname="piper", ensure_exists=True
        )
    )
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"

    target_width: int = 224
    target_height: int = 224
    augmentations: list[AugmentationType] = field(default_factory=list)

    output_format: Literal["filesystem", "webdataset", "parquet"] = "filesystem"

    use_spark: bool = False
    spark_partitions: int = 0  # 0 = auto

    preserve_structure: bool = True

    def __post_init__(self):
        """Ensure data_dir is a Path object."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw/downloaded data directory."""
        return self.data_dir / self.raw_subdir

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed output directory."""
        return self.data_dir / self.processed_subdir

    @classmethod
    def from_env(cls) -> "PiperConfig":
        """Load configuration from environment variables.

        Environment variables:
            PIPER_DATA_DIR: Base data directory (default: /data)
            PIPER_WIDTH: Target image width (default: 224)
            PIPER_HEIGHT: Target image height (default: 224)
            PIPER_OUTPUT_FORMAT: Output format (default: filesystem)
            PIPER_USE_SPARK: Whether to use Spark (default: false)
            PIPER_SPARK_PARTITIONS: Number of Spark partitions (default: 0)
        """
        augmentations = []
        aug_str = os.getenv("PIPER_AUGMENTATIONS", "")
        if aug_str:
            for aug_name in aug_str.split(","):
                aug_name = aug_name.strip().upper()
                try:
                    augmentations.append(AugmentationType[aug_name])
                except KeyError:
                    pass  # Ignore invalid augmentation names

        return cls(
            data_dir=Path(os.getenv("PIPER_DATA_DIR", "/data")),
            target_width=int(os.getenv("PIPER_WIDTH", "224")),
            target_height=int(os.getenv("PIPER_HEIGHT", "224")),
            output_format=os.getenv("PIPER_OUTPUT_FORMAT", "filesystem"),  # type: ignore
            use_spark=os.getenv("PIPER_USE_SPARK", "false").lower() == "true",
            spark_partitions=int(os.getenv("PIPER_SPARK_PARTITIONS", "0")),
            augmentations=augmentations,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "PiperConfig":
        """Load configuration from a dictionary.

        Args:
            d: Dictionary with configuration values. Keys should match
               the attribute names of PiperConfig.

        Returns:
            PiperConfig instance.
        """
        # Handle augmentations conversion if provided as strings
        if "augmentations" in d:
            augs = d["augmentations"]
            if augs and isinstance(augs[0], str):
                d = d.copy()
                d["augmentations"] = [
                    AugmentationType[a.upper()] if isinstance(a, str) else a
                    for a in augs
                ]

        # Handle data_dir conversion
        if "data_dir" in d and isinstance(d["data_dir"], str):
            d = d.copy()
            d["data_dir"] = Path(d["data_dir"])

        return cls(**d)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "data_dir": str(self.data_dir),
            "raw_subdir": self.raw_subdir,
            "processed_subdir": self.processed_subdir,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "augmentations": [a.value for a in self.augmentations],
            "output_format": self.output_format,
            "use_spark": self.use_spark,
            "spark_partitions": self.spark_partitions,
            "preserve_structure": self.preserve_structure,
        }

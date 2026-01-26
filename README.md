# Piper

Piper is a distributed image processing pipeline for machine learning datasets. It automates downloading image datasets from various sources (Kaggle, HuggingFace, local), processing them with configurable augmentations, and outputting results in ML-ready formats.

## Features

- **Multiple Data Sources**: Kaggle datasets, HuggingFace Datasets, or local directories
- **Distributed Processing**: PySpark integration for scalable image processing
- **Workflow Orchestration**: Luigi for task management and dependency tracking
- **Flexible Augmentations**: Flip, rotate, color jitter via OpenCV
- **Multiple Output Formats**: ImageFolder structure or WebDataset shards
- **Dockerized**: All components containerized for easy deployment

## Installation

```bash
# Core installation
pip install piper

# With optional dependencies
pip install piper[kaggle]      # Kaggle support
pip install piper[huggingface] # HuggingFace support
pip install piper[spark]       # Spark distributed processing
pip install piper[luigi]       # Luigi orchestration
pip install piper[webdataset]  # WebDataset output format
pip install piper[all]         # All optional dependencies
```

Or with uv:

```bash
uv add piper
uv add piper --extra all
```

## Quick Start

### Programmatic API:

```python
from piper import Pipeline, PiperConfig, AugmentationType

# Simple usage - process a Kaggle dataset
Pipeline().from_kaggle("username/dataset").to_filesystem("./output").run()

# With configuration
config = PiperConfig(
    target_width=512,
    target_height=512,
    augmentations=[AugmentationType.FLIP, AugmentationType.ROTATE],
)

Pipeline(config).from_huggingface("cifar10").to_filesystem("./processed").run()

# Process local images
Pipeline().from_local("./my_images").to_webdataset("./shards").run()
```


## Architecture

```
Pipeline API (piper/api.py)
    ├── from_kaggle() / from_huggingface() / from_local()
    │   └── Uses piper/sources/ adapters
    ├── to_filesystem() / to_webdataset()
    │   └── Uses piper/outputs/ writers
    └── run(distributed=True/False)
        ├── Local: Uses piper/processors/ directly
        └── Spark: Submits piper/spark/data_augment.py job

Luigi Orchestration (piper/luigi_tasks/)
    └── ProcessingPipeline → SparkProcessImages → ListRawImages → DownloadDataset
```

## Configuration

```python
from piper import PiperConfig, AugmentationType

config = PiperConfig(
    # Data directories
    data_dir="./data",
    raw_subdir="raw",
    processed_subdir="processed",
    
    # Image processing
    target_width=224,
    target_height=224,
    
    # Augmentations
    augmentations=[
        AugmentationType.FLIP,
        AugmentationType.ROTATE,
        AugmentationType.COLOR_JITTER,
    ],
    
    # Distributed processing
    use_spark=False,
    spark_partitions=0,  # Auto-detect
)
```

## Data Sources

### Kaggle

```python
from piper import Pipeline

Pipeline().from_kaggle("username/dataset-name").run()
```

Requires Kaggle API credentials in `~/.kaggle/kaggle.json`.

### HuggingFace Datasets

```python
from piper import Pipeline

Pipeline().from_huggingface(
    "cifar10",
    split="train",
    subset=None,  # Optional configuration name
    image_column="image",
    label_column="label",
).run()
```

### Local Directory

```python
from piper import Pipeline

Pipeline().from_local("./my_images", recursive=True).run()
```

## Output Formats

### Filesystem (ImageFolder)

```python
Pipeline().from_local("./input").to_filesystem("./output").run()

# Output structure:
# output/
#   ├── image_001.png
#   ├── image_002.png
#   └── metadata.json
```

### WebDataset (Tar Shards)

```python
Pipeline().from_local("./input").to_webdataset("./shards", shard_size=1000).run()

# Output structure:
# shards/
#   ├── shard-000000.tar
#   ├── shard-000001.tar
#   └── ...
```

## Docker Deployment

```bash
# Build all containers
docker buildx bake

# Run the full pipeline
docker compose up

# Start interactive shell
just shell luigi-worker
```

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/piper.git
cd piper

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linting
uv run ruff check piper/

# Run type checking
uv run mypy piper/
```

## Project Structure

```
piper/
├── api.py              # Pipeline class - main entry point
├── config.py           # PiperConfig dataclass
├── logger/             # Logging configuration
├── luigi_tasks/        # Luigi task definitions
├── outputs/            # Output writers (filesystem, webdataset)
├── processors/         # Image processing (transform, augment)
├── sources/            # Data sources (kaggle, huggingface, local)
├── spark/              # PySpark job for distributed processing
└── utils/              # Shared utilities and models
```


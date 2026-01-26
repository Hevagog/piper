"""Piper CLI entry point.

This script provides a command-line interface for running piper pipelines.
By default, it uses local processing. Use --use-luigi for Luigi scheduler orchestration.

Usage:
    python -m piper.main --source kaggle --source-id user/dataset

    # Or use the Pipeline API directly in your code (recommended)
    from piper import Pipeline
    Pipeline().from_kaggle("user/dataset").to_filesystem().run()
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Piper - Distributed Image Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a Kaggle dataset via Luigi scheduler
    python -m piper.main --source kaggle --source-id user/dataset

    # Process HuggingFace dataset
    python -m piper.main --source huggingface --source-id cifar10 --hf-split train

    # Process local images without Spark
    python -m piper.main --source local --source-id ./my_images --no-spark

    # Use programmatic API (recommended for new code)
    from piper import Pipeline
    Pipeline().from_kaggle("user/dataset").to_filesystem().run()
        """,
    )

    parser.add_argument(
        "--source",
        choices=["kaggle", "huggingface", "local"],
        default="kaggle",
        help="Data source type (default: kaggle)",
    )
    parser.add_argument(
        "--source-id",
        default=os.getenv("KAGGLE_URL", ""),
        help="Source identifier (Kaggle URL, HF dataset name, or local path)",
    )

    parser.add_argument("--hf-split", default="train", help="HuggingFace dataset split")
    parser.add_argument("--hf-subset", default=None, help="HuggingFace dataset subset")

    # Processing options
    parser.add_argument(
        "--width", type=int, default=224, help="Target image width (default: 224)"
    )
    parser.add_argument(
        "--height", type=int, default=224, help="Target image height (default: 224)"
    )
    parser.add_argument(
        "--augmentations",
        default="",
        help="Comma-separated augmentations (flip,rotate,color_jitter)",
    )
    parser.add_argument(
        "--no-spark",
        action="store_true",
        help="Disable Spark, use local processing",
    )

    # Luigi scheduler options
    parser.add_argument(
        "--scheduler-host",
        default=os.getenv("LUIGI_SCHEDULER_HOST", "luigi-scheduler"),
        help="Luigi scheduler host",
    )
    parser.add_argument(
        "--scheduler-port",
        type=int,
        default=int(os.getenv("LUIGI_SCHEDULER_PORT", "8082")),
        help="Luigi scheduler port",
    )

    # Execution mode
    parser.add_argument(
        "--use-luigi",
        action="store_true",
        help="Use Luigi scheduler (default: use Pipeline API)",
    )

    args = parser.parse_args()

    if not args.source_id:
        parser.error("--source-id is required")

    if args.use_luigi:
        _run_with_luigi(args)
    else:
        _run_with_pipeline(args)


def _run_with_pipeline(args):
    """Run using the programmatic Pipeline API."""
    from piper import Pipeline, PiperConfig
    from piper.utils.models import AugmentationType

    # Parse augmentations
    augmentations = []
    if args.augmentations:
        for name in args.augmentations.split(","):
            name = name.strip().upper()
            try:
                augmentations.append(AugmentationType[name])
            except KeyError:
                print(f"Warning: Unknown augmentation '{name}', skipping")

    # Create config
    config = PiperConfig(
        target_width=args.width,
        target_height=args.height,
        augmentations=augmentations,
        use_spark=not args.no_spark,
    )

    # Build pipeline
    pipeline = Pipeline(config)

    match args.source:
        case "kaggle":
            pipeline.from_kaggle(args.source_id)
        case "huggingface":
            pipeline.from_huggingface(
                args.source_id,
                split=args.hf_split,
                subset=args.hf_subset,
            )
        case "local":
            pipeline.from_local(args.source_id)

    # Configure output
    pipeline.to_filesystem()

    # Run
    pipeline.run()

    print("Pipeline completed successfully!")


def _run_with_luigi(args):
    """Run using Luigi scheduler."""
    try:
        import luigi
    except ImportError:
        print("Error: Luigi is not installed. Install with: pip install piper[luigi]")
        sys.exit(1)

    from piper.luigi_tasks import ProcessingPipeline

    scheduler_url = f"http://{args.scheduler_host}:{args.scheduler_port}"

    task = ProcessingPipeline(
        source_type=args.source,
        source_id=args.source_id,
        hf_split=args.hf_split,
        hf_subset=args.hf_subset,
        augmentations=args.augmentations,
    )

    success = luigi.build(
        [task],
        local_scheduler=False,
        scheduler_url=scheduler_url,
        workers=0,
        log_level="INFO",
    )

    if success:
        print("All tasks completed successfully!")
    else:
        print("Some tasks failed. Check Luigi scheduler for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

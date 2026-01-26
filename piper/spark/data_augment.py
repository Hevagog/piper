import argparse
import os

from pyspark.sql import SparkSession

from piper.logger import get_logger
from piper.processors import Augmentor, ImageTransformer
from piper.utils.models import AugmentationType


def process_image_row(
    row,
    output_dir: str,
    transformer: ImageTransformer,
    augmentor: Augmentor | None,
    input_dir: str,
) -> None:
    """Process an image row from Spark's image data source.

    Args:
        row: DataFrame row with image data.
        output_dir: Directory to save processed images.
        transformer: ImageTransformer for resizing.
        augmentor: Augmentor for applying augmentations (or None).
        input_dir: Input directory (for maintaining directory structure).
    """
    import cv2
    import numpy as np

    try:
        # Extract image data and convert to OpenCV format
        img_data = row.image
        nparr = np.frombuffer(img_data.data, dtype=np.uint8)

        # Check if the image has valid data
        if len(nparr) == 0:
            return

        # Decode the image based on the number of channels
        if img_data.nChannels == 3:
            img = nparr.reshape(img_data.height, img_data.width, img_data.nChannels)
        else:
            # Handle grayscale or other formats by decoding from binary data
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize using the transformer
        img = transformer.resize(img)

        # Apply augmentations if configured
        if augmentor is not None:
            img = augmentor(img)

        # Determine output path
        image_path = row.image.origin
        if image_path.startswith("file:"):
            image_path = image_path[5:]  # Remove 'file:' prefix

        rel_path = os.path.relpath(image_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Save using transformer
        ImageTransformer.save(img, out_path)

    except Exception as e:
        origin = getattr(row.image, "origin", "unknown")
        print(f"Error processing image {origin}: {e}")


def parse_augmentations(aug_string: str | None) -> list[AugmentationType]:
    """Parse augmentation string to list of AugmentationType."""
    if not aug_string:
        return []

    augmentations = []
    for name in aug_string.split(","):
        name = name.strip().upper()
        try:
            augmentations.append(AugmentationType[name])
        except KeyError:
            print(f"Warning: Unknown augmentation '{name}', skipping")
    return augmentations


def main():
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Distributed image processing with Spark"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with images"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for processed images"
    )
    parser.add_argument("--width", type=int, default=224, help="Target image width")
    parser.add_argument("--height", type=int, default=224, help="Target image height")
    parser.add_argument(
        "--augmentations",
        default="",
        help="Comma-separated list of augmentations (flip,rotate,color_jitter)",
    )
    parser.add_argument(
        "--partition_count",
        type=int,
        default=0,
        help="Number of partitions (0 for auto)",
    )
    args = parser.parse_args()

    logger.info("Starting distributed image processing")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Target size: {args.width}x{args.height}")

    augmentation_types = parse_augmentations(args.augmentations)
    if augmentation_types:
        logger.info(f"Augmentations: {[a.value for a in augmentation_types]}")
    else:
        logger.info("No augmentations configured")

    spark = (
        SparkSession.builder.appName("PiperImageProcessing")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    try:
        logger.info("Loading images from input directory")
        image_df = (
            spark.read.format("image")
            .option("dropInvalid", True)
            .option("recursiveFileLookup", True)
            .load(args.input_dir)
        )

        image_count = image_df.count()
        logger.info(f"Loaded {image_count} images")

        if args.partition_count > 0:
            logger.info(f"Repartitioning to {args.partition_count} partitions")
            image_df = image_df.repartition(args.partition_count)

        width = args.width
        height = args.height
        output_dir = args.output_dir
        input_dir = args.input_dir
        aug_types = augmentation_types

        def process_partition(partition):
            """Process a partition of images."""
            transformer = ImageTransformer(width=width, height=height)
            augmentor = Augmentor(aug_types) if aug_types else None

            for row in partition:
                process_image_row(row, output_dir, transformer, augmentor, input_dir)

        logger.info("Starting image processing")
        image_df.foreachPartition(process_partition)

        logger.info(f"Successfully processed {image_count} images")

    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()

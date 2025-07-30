import argparse
from pyspark.sql import SparkSession
import cv2
import os
import numpy as np
from random import randint, seed

from piper.logger import get_logger
from piper.utils.models import AugmentationType


def augment_image(img, augmentations: AugmentationType, seed_value: int = 42):
    seed(seed_value)
    match augmentations:
        case AugmentationType.FLIP | "flip":
            img = cv2.flip(img, randint(-1, 1))
        case AugmentationType.ROTATE | "rotate":
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        case AugmentationType.COLOR_JITTER | "color_jitter":
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        case _:
            pass
    return img


def process_image_row(row, output_dir, width, height, augmentations, input_dir):
    """
    Process an image row from Spark's image data source

    Args:
        row: DataFrame row with image data
        output_dir: Directory to save processed images
        width: Target width
        height: Target height
        augmentations: List of augmentation types to apply
        input_dir: Input directory (used for maintaining directory structure)
    """
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

        # Resize and augment the image
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = augment_image(img, augmentations)

        image_path = row.image.origin
        if image_path.startswith("file:"):
            image_path = image_path[5:]  # Remove 'file:' prefix

        rel_path = os.path.relpath(image_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        file_ext = os.path.splitext(out_path)[1].lower()
        if file_ext in [".jpg", ".jpeg"]:
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(out_path, img)

    except Exception as e:
        print(f"Error processing image {row.image.origin}: {e}")


def encode_image_to_binary(img):
    """Convert OpenCV image to binary for storage in DataFrame"""
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


if __name__ == "__main__":
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--augmentations")
    parser.add_argument(
        "--partition_count",
        type=int,
        default=0,
        help="Number of partitions to use (0 for auto)",
    )
    args = parser.parse_args()

    logger.info("Starting image augmentation process")

    spark = (
        SparkSession.builder.appName("ImageAugment")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    width = args.width
    height = args.height
    augmentations = [aug.name for aug in AugmentationType]

    logger.info("Reading images from input directory")
    image_df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .option("recursiveFileLookup", True)
        .load(args.input_dir)
    )

    print(f"Total images loaded: {image_df.count()}")

    # Repartition if specified
    if args.partition_count > 0:
        logger.info(f"Repartitioning DataFrame to {args.partition_count} partitions")
        image_df = image_df.repartition(args.partition_count)

    def process_partition(partition):
        for row in partition:
            process_image_row(
                row, args.output_dir, width, height, augmentations, args.input_dir
            )

    logger.info("Starting image processing")
    image_df.foreachPartition(process_partition)

    logger.info(f"Processed {image_df.count()} images")
    logger.info("Image augmentation process completed")
    spark.stop()

import luigi
from luigi.util import requires
import os
import subprocess
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import cv2

from piper.logger import get_logger
from piper.constants import IMAGE_FORMATS
import shutil


class GlobalConfig(luigi.Config):
    """
    Template class for global configuration.
    """

    target_width = luigi.IntParameter(default=224)
    target_height = luigi.IntParameter(default=224)

    base_data_dir = luigi.Parameter(default="data")

    kaggle_dataset_owner = luigi.Parameter(default="")
    kaggle_dataset_name = luigi.Parameter(default="")

    @property
    def raw_data_dir(self):
        return os.path.join(self.base_data_dir, "raw")

    @property
    def processed_data_dir(self):
        return os.path.join(self.base_data_dir, "processed")


def set_up_dataset(
    kaggle_url: str,
    target_width: int = 224,
    target_height: int = 224,
    base_data_dir: str = "data",
    num_workers: int = 8,
):
    """
    Sets up the global configuration and runs the Luigi pipeline.

    Args:
        kaggle_url (str): Kaggle dataset URL in the format "owner/dataset".
        target_width (int, optional): Target width for image resizing. Defaults to 224.
        target_height (int, optional): Target height for image resizing. Defaults to 224.
        base_data_dir (str, optional): Base directory for data storage. Defaults to "data".
        num_workers (int, optional): Number of workers for parallel processing. Defaults to 8.

    Raises:
        ValueError: If the Kaggle URL format is invalid.
    """
    try:
        owner, dataset_name = kaggle_url.split("/")
    except ValueError:
        raise ValueError("Invalid Kaggle URL format. Expected 'owner/dataset'.")

    GlobalConfig.target_width = target_width
    GlobalConfig.target_height = target_height
    GlobalConfig.base_data_dir = base_data_dir
    GlobalConfig.kaggle_dataset_owner = owner
    GlobalConfig.kaggle_dataset_name = dataset_name

    luigi.run(
        ["SparkAugmentImages", "--local-scheduler", "--workers", str(num_workers)]
    )


class DownloadKaggleDataset(luigi.Task):
    """
    Downloads the specified Kaggle dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        """
        Returns the target output for this task.
        """
        return luigi.LocalTarget(
            os.path.join(GlobalConfig().raw_data_dir, ".dataset_download_complete")
        )

    def run(self):
        """
        Executes the dataset download and unzipping efficiently.
        It detects if the zip archive contains a single root directory and, if so,
        extracts its contents directly into the target raw_data_dir, avoiding
        the creation of an unwanted subdirectory.
        """
        config = GlobalConfig()
        raw_data_dir = config.raw_data_dir
        try:
            os.makedirs(raw_data_dir, exist_ok=True)

            api = KaggleApi()
            api.authenticate()

            dataset_slug = f"{config.kaggle_dataset_owner}/{config.kaggle_dataset_name}"
            zip_path = os.path.join(raw_data_dir, f"{config.kaggle_dataset_name}.zip")

            self._logger.info(f"Downloading dataset: {dataset_slug} to {zip_path}")
            api.dataset_download_files(dataset_slug, path=raw_data_dir, unzip=False)
            self._logger.info("Download complete.")

            self._logger.info(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                file_list = zip_ref.infolist()
                if not file_list:
                    self._logger.warning(f"Zip file {zip_path} is empty.")
                else:
                    # Check if all members are in a single top-level directory
                    first_member_parts = file_list[0].filename.split("/")
                    root_dir = (
                        first_member_parts[0] if len(first_member_parts) > 1 else None
                    )

                    is_single_root = root_dir is not None and all(
                        m.filename.startswith(root_dir + "/") for m in file_list
                    )

                    if is_single_root:
                        self._logger.info(
                            f"Single root directory '{root_dir}' detected. Stripping it during extraction."
                        )
                        for member in file_list:
                            # Skip the root directory entry itself
                            if member.filename == root_dir + "/":
                                continue
                            new_path = os.path.join(
                                raw_data_dir, member.filename[len(root_dir) + 1 :]
                            )
                            if member.is_dir():
                                os.makedirs(new_path, exist_ok=True)
                            else:
                                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                                with (
                                    zip_ref.open(member, "r") as source,
                                    open(new_path, "wb") as target,
                                ):
                                    shutil.copyfileobj(source, target)
                    else:
                        self._logger.info(
                            "No single root directory detected. Extracting as is."
                        )
                        zip_ref.extractall(raw_data_dir)

            self._logger.info("Unzipping complete.")

            os.remove(zip_path)
            self._logger.info(f"Removed zip file: {zip_path}")

            with self.output().open("w") as f:
                f.write("download_complete\n")

        except Exception as e:
            self._logger.error(f"Failed to download and extract dataset: {e}")
            # Clean up partial downloads if something went wrong
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise


@requires(DownloadKaggleDataset)
class ListRawImages(luigi.Task):
    """
    Lists all image files in the raw data directory after download.
    This task doesn't create new files, but helps with dependency management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        """
        Returns a target indicating that the raw image list has been identified.
        """
        return luigi.LocalTarget(
            os.path.join(GlobalConfig().raw_data_dir, ".raw_image_list_complete")
        )

    def run(self):
        """
        Simply creates a marker file after ensuring the download is complete.
        The actual image paths will be passed down to subsequent tasks.
        """
        self._logger.info("Raw image list task completed - ready for image processing")
        with self.output().open("w") as f:
            f.write("raw_image_list_generated\n")


@requires(ListRawImages)
class ResizeImage(luigi.Task):
    """
    Resizes a single image to a specified dimension.
    """

    image_path = luigi.Parameter()
    target_width = luigi.IntParameter()
    target_height = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        """
        Returns the target output for this task (the resized image file).
        """
        relative_path = os.path.relpath(self.image_path, GlobalConfig().raw_data_dir)
        output_dir = os.path.join(
            GlobalConfig().processed_data_dir, os.path.dirname(relative_path)
        )
        output_filename = os.path.basename(self.image_path)

        return luigi.LocalTarget(os.path.join(output_dir, output_filename))

    def run(self):
        """
        Loads the image, resizes it, and saves it to the processed directory.
        Optimized for OpenCV performance and parallel execution.
        """
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        self._logger.debug(
            f"Resizing {self.image_path} to {self.target_width}x{self.target_height}"
        )

        try:
            img = cv2.imread(self.image_path)

            if img is None:
                raise ValueError(f"Could not read image: {self.image_path}")

            resized = cv2.resize(
                img,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA,
            )

            file_ext = os.path.splitext(self.output().path)[1].lower()
            if file_ext in [".jpg", ".jpeg"]:
                success = cv2.imwrite(
                    self.output().path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            else:
                success = cv2.imwrite(self.output().path, resized)

            if not success:
                raise ValueError(f"Could not save image: {self.output().path}")

            self._logger.debug(f"Resized image saved to: {self.output().path}")

        except Exception as e:
            self._logger.error(f"Error processing {self.image_path}: {e}")
            raise


@requires(ListRawImages)
class SparkAugmentImages(luigi.Task):
    """
    Luigi Task that launches a PySpark job to perform distributed image resizing and augmentation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(GlobalConfig().processed_data_dir, ".spark_augment_complete")
        )

    def run(self):
        """
        Launches a PySpark job for distributed image processing.
        """

        # Create a zip archive of the 'piper' package to be sent to Spark workers
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

        spark_script = os.path.join(os.path.dirname(__file__), "data_augment.py")
        cmd = [
            "spark-submit",
            "--py-files",
            archive_path,
            spark_script,
            "--input_dir",
            GlobalConfig().raw_data_dir,
            "--output_dir",
            GlobalConfig().processed_data_dir,
            "--width",
            str(GlobalConfig().target_width),
            "--height",
            str(GlobalConfig().target_height),
        ]
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


@requires(ListRawImages)
class ProcessAllImages(luigi.Task):
    """
    Orchestrates the resizing of all images in the dataset.
    This is a WrapperTask that requires ResizeImage for all images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(__name__)

    def run(self):
        """
        Yields a ResizeImage task for each image found in the raw data directory.
        Luigi will run these tasks in parallel when --workers > 1 is specified.
        """
        raw_images_root = GlobalConfig().raw_data_dir
        self._logger.info(f"Scanning for images in {raw_images_root}...")

        if not self.input().exists():
            error_msg = "Raw image list not complete. Run DownloadKaggleDataset and ListRawImages first."
            self._logger.error(error_msg)
            raise luigi.MissingDependencyException(error_msg)

        image_paths = []
        try:
            for root, _, files in os.walk(raw_images_root):
                valid_files = [
                    os.path.join(root, file)
                    for file in files
                    if file.lower().endswith(IMAGE_FORMATS)
                ]
                image_paths.extend(valid_files)
        except Exception as e:
            self._logger.error(f"Error scanning for images in {raw_images_root}: {e}")
            raise

        if not image_paths:
            self._logger.warning(
                f"No images found in {raw_images_root}. Please check the dataset structure."
            )
        else:
            self._logger.info(f"Found {len(image_paths)} images to process in parallel")

        for img_path in image_paths:
            yield ResizeImage(
                image_path=img_path,
                target_width=GlobalConfig().target_width,
                target_height=GlobalConfig().target_height,
            )

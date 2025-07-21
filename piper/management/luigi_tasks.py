import luigi
from luigi.util import requires
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
import numpy as np
import cv2

from piper.logger import get_logger


class GlobalConfig(luigi.Config):
    
    target_width = luigi.IntParameter(default=224)
    target_height = luigi.IntParameter(default=224)

    base_data_dir = luigi.Parameter(default="data")
    
    kaggle_dataset_owner = luigi.Parameter(default="jiayuanchengala")
    kaggle_dataset_name = luigi.Parameter(default="aid-scene-classification-datasets")
    
    @property
    def raw_data_dir(self):
        return os.path.join(self.base_data_dir, "raw")
    
    @property
    def processed_data_dir(self):
        return os.path.join(self.base_data_dir, "processed")


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
        In this case, it's a marker file indicating the dataset has been downloaded
        and unzipped to the raw data directory.
        """
        return luigi.LocalTarget(os.path.join(GlobalConfig().raw_data_dir, '.dataset_download_complete'))

    def run(self):
        """
        Executes the dataset download and unzipping.
        """
        try:
            os.makedirs(GlobalConfig().raw_data_dir, exist_ok=True)

            api = KaggleApi()
            api.authenticate()

            dataset_slug = f"{GlobalConfig().kaggle_dataset_owner}/{GlobalConfig().kaggle_dataset_name}"
            zip_path = os.path.join(GlobalConfig().raw_data_dir, f"{GlobalConfig().kaggle_dataset_name}.zip")

            self._logger.info(f"Downloading dataset: {dataset_slug} to {zip_path}")
            api.dataset_download_files(
                dataset_slug, 
                path=GlobalConfig().raw_data_dir, 
                unzip=False 
            )
            self._logger.info("Download complete.")

            self._logger.info(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(GlobalConfig().raw_data_dir)
            self._logger.info("Unzipping complete.")

            os.remove(zip_path)
            self._logger.info(f"Removed zip file: {zip_path}")

            with self.output().open('w') as f:
                f.write("download_complete\n")
                
        except Exception as e:
            self._logger.error(f"Failed to download and extract dataset: {e}")
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
        return luigi.LocalTarget(os.path.join(GlobalConfig().raw_data_dir, '.raw_image_list_complete'))

    def run(self):
        """
        Simply creates a marker file after ensuring the download is complete.
        The actual image paths will be passed down to subsequent tasks.
        """
        self._logger.info("Raw image list task completed - ready for image processing")
        with self.output().open('w') as f:
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
        output_dir = os.path.join(GlobalConfig().processed_data_dir, os.path.dirname(relative_path))
        output_filename = os.path.basename(self.image_path)
        
        
        return luigi.LocalTarget(os.path.join(output_dir, output_filename))

    def run(self):
        """
        Loads the image, resizes it, and saves it to the processed directory.
        Optimized for OpenCV performance and parallel execution.
        """
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        
        self._logger.debug(f"Resizing {self.image_path} to {self.target_width}x{self.target_height}")
        
        try:
            img = cv2.imread(self.image_path)
            
            if img is None:
                raise ValueError(f"Could not read image: {self.image_path}")
            
            resized = cv2.resize(
                img, 
                (self.target_width, self.target_height), 
                interpolation=cv2.INTER_AREA
            )
            
            file_ext = os.path.splitext(self.output().path)[1].lower()
            if file_ext in ['.jpg', '.jpeg']:
                success = cv2.imwrite(
                    self.output().path, 
                    resized, 
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
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
        image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
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
                    if file.lower().endswith(image_formats)
                ]
                image_paths.extend(valid_files)
        except Exception as e:
            self._logger.error(f"Error scanning for images in {raw_images_root}: {e}")
            raise
        
        if not image_paths:
            self._logger.warning(f"No images found in {raw_images_root}. Please check the dataset structure.")
        else:
            self._logger.info(f"Found {len(image_paths)} images to process in parallel")

        for img_path in image_paths:
            yield ResizeImage(
                image_path=img_path,
                target_width=GlobalConfig().target_width,
                target_height=GlobalConfig().target_height,
            )
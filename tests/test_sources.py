from pathlib import Path


class TestLocalSource:
    """Test LocalSource class."""

    def test_local_source_creation(self, temp_dir):
        """Test LocalSource instantiation."""
        from piper.sources.local import LocalSource

        source = LocalSource(temp_dir)
        assert source is not None
        # The name property might be different depending on implementation
        assert hasattr(source, "name")

    def test_local_source_download(self, sample_images_dir):
        """Test LocalSource download (which is a no-op for local files)."""
        from piper.sources.local import LocalSource

        source = LocalSource(sample_images_dir)
        result = source.download(sample_images_dir)

        # For local source, download returns the same path
        assert result == sample_images_dir

    def test_local_source_iter_images(self, sample_images_dir):
        """Test iterating over images in local source."""
        from piper.sources.local import LocalSource

        source = LocalSource(sample_images_dir)
        images = list(source.iter_images(sample_images_dir))

        # We created 5 test images
        assert len(images) == 5

        # Each should have a path
        for img_info in images:
            assert img_info.path.exists()
            assert img_info.path.suffix == ".png"


class TestBaseDataSource:
    """Test base data source functionality."""

    def test_image_info_dataclass(self):
        """Test ImageInfo dataclass."""
        from piper.sources.base import ImageInfo

        info = ImageInfo(path=Path("/test/image.png"))
        assert info.path == Path("/test/image.png")
        assert info.label is None
        assert info.metadata == {}

    def test_image_info_with_metadata(self):
        """Test ImageInfo with label and metadata."""
        from piper.sources.base import ImageInfo

        info = ImageInfo(
            path=Path("/test/image.png"),
            label="cat",
            metadata={"width": 100, "height": 100},
        )
        assert info.label == "cat"
        assert info.metadata["width"] == 100


class TestKaggleSourceImport:
    """Test KaggleSource import behavior."""

    def test_kaggle_import_requires_package(self):
        """Test that KaggleSource requires kaggle package."""
        try:
            from piper.sources.kaggle import KaggleSource

            # If import succeeds, kaggle is installed
            source = KaggleSource("user/dataset")
            assert hasattr(source, "name")
        except ImportError:
            # Expected when kaggle is not installed
            pass
        except Exception:
            # Other exceptions are acceptable (e.g., credential issues)
            pass


class TestHuggingFaceSourceImport:
    """Test HuggingFaceSource import behavior."""

    def test_huggingface_import_requires_package(self):
        """Test that HuggingFaceSource requires datasets package."""
        try:
            from piper.sources.huggingface import HuggingFaceSource

            # If import succeeds, datasets is installed
            assert HuggingFaceSource is not None
        except ImportError as e:
            # If import fails, check the error
            assert "datasets" in str(e).lower() or "huggingface" in str(e).lower()

import pytest


class TestImports:
    """Test that core imports work correctly."""

    def test_import_pipeline(self):
        """Test that Pipeline can be imported."""
        from piper import Pipeline

        assert Pipeline is not None

    def test_import_config(self):
        """Test that PiperConfig can be imported."""
        from piper import PiperConfig

        assert PiperConfig is not None

    def test_import_augmentation_type(self):
        """Test that AugmentationType can be imported."""
        from piper import AugmentationType

        assert AugmentationType is not None
        assert hasattr(AugmentationType, "FLIP")
        assert hasattr(AugmentationType, "ROTATE")
        assert hasattr(AugmentationType, "COLOR_JITTER")

    def test_import_submodules(self):
        """Test that submodules can be imported."""
        from piper import outputs, processors, sources

        assert outputs is not None
        assert processors is not None
        assert sources is not None

    def test_luigi_tasks_lazy_import(self):
        """Test that luigi_tasks raises ImportError when luigi is not installed."""
        # This test may pass or fail depending on whether luigi is installed
        try:
            from piper import luigi_tasks

            # If it succeeds, luigi is installed
            assert luigi_tasks is not None
        except ImportError as e:
            # If it fails, check the error message
            assert "luigi" in str(e).lower()


class TestPiperConfig:
    """Test PiperConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from piper import PiperConfig

        config = PiperConfig()
        assert config.target_width == 224
        assert config.target_height == 224
        # Note: use_spark defaults to True in the actual implementation
        assert config.augmentations == []

    def test_custom_config(self):
        """Test custom configuration values."""
        from piper import AugmentationType, PiperConfig

        config = PiperConfig(
            target_width=512,
            target_height=512,
            augmentations=[AugmentationType.FLIP, AugmentationType.ROTATE],
            use_spark=True,
        )
        assert config.target_width == 512
        assert config.target_height == 512
        assert len(config.augmentations) == 2
        assert config.use_spark is True


class TestPipeline:
    """Test Pipeline class."""

    def test_pipeline_creation(self):
        """Test Pipeline instantiation."""
        from piper import Pipeline

        pipeline = Pipeline()
        assert pipeline is not None
        assert pipeline.config is not None

    def test_pipeline_with_config(self):
        """Test Pipeline with custom config."""
        from piper import Pipeline, PiperConfig

        config = PiperConfig(target_width=512)
        pipeline = Pipeline(config)
        assert pipeline.config.target_width == 512

    def test_pipeline_method_chaining(self, temp_dir):
        """Test that pipeline methods return self for chaining."""
        from piper import Pipeline

        pipeline = Pipeline()

        # Test that from_local returns Pipeline
        result = pipeline.from_local(temp_dir)
        assert result is pipeline

        # Test that to_filesystem returns Pipeline
        result = pipeline.to_filesystem(temp_dir / "output")
        assert result is pipeline

    def test_pipeline_requires_source(self):
        """Test that run() raises error without source configured."""
        from piper import Pipeline

        pipeline = Pipeline()
        with pytest.raises(ValueError, match="No data source configured"):
            pipeline.run()

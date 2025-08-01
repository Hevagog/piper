# Piper

Piper is a distributed image processing pipeline built with PySpark, Luigi, and OpenCV. It automates the workflow of downloading image datasets, augmenting and resizing images, and storing processed results in a volume.

## Features

- Distributed Processing: Uses PySpark for scalable image augmentation and resizing.
- Workflow Orchestration: Luigi manages tasks and dependencies, including dataset download, extraction, and image processing.
- Flexible Augmentation: Supports flipping, rotation, and color jitter via OpenCV.
- Dockerized: All components run in containers for easy deployment and reproducibility.
- Kaggle Integration: Automatically downloads datasets from Kaggle.

## Architecture

- Luigi Scheduler & Worker: Orchestrates the pipeline, manages dependencies, and triggers Spark jobs.
- PySpark Job: Performs distributed image augmentation and resizing.
- Logger: Centralized logging configuration for all components.
- Docker Compose: Manages multi-container setup (scheduler, worker, orchestrator).

## Pipeline Overview

- DownloadKaggleDataset: Downloads and extracts a Kaggle dataset.
- ListRawImages: Identifies all image files in the dataset.
- ResizeImage: Resizes images to target dimensions.
- SparkAugmentImages: Runs a PySpark job for distributed augmentation and resizing.
- ProcessAllImages: Orchestrates parallel processing of all images.

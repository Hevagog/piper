import os
from piper import set_up_dataset


def main():
    # Get scheduler configuration from environment variables (for Docker)
    scheduler_host = os.getenv("LUIGI_SCHEDULER_HOST", "localhost")
    scheduler_port = int(os.getenv("LUIGI_SCHEDULER_PORT", "8082"))

    set_up_dataset(
        kaggle_url="jiayuanchengala/aid-scene-classification-datasets",
        scheduler_host=scheduler_host,
        scheduler_port=scheduler_port,
    )


if __name__ == "__main__":
    main()

import os

import luigi

from piper.luigi_tasks.luigi_tasks import AllTasks


def main():
    scheduler_host = os.getenv("LUIGI_SCHEDULER_HOST", "luigi-scheduler")
    scheduler_port = int(os.getenv("LUIGI_SCHEDULER_PORT", "8082"))
    kaggle_url = os.getenv(
        "KAGGLE_URL", "jiayuanchengala/aid-scene-classification-datasets"
    )
    scheduler_url = f"http://{scheduler_host}:{scheduler_port}"

    luigi.build(
        [AllTasks(kaggle_url=kaggle_url)],
        local_scheduler=False,
        scheduler_url=scheduler_url,
        workers=0,
        log_level="DEBUG",
    )


if __name__ == "__main__":
    main()

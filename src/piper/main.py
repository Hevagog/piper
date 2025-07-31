import os
import subprocess

import luigi

from piper.luigi_tasks.luigi_tasks import SparkAugmentImages


def main():
    scheduler_host = os.getenv("LUIGI_SCHEDULER_HOST", "luigi-scheduler")
    scheduler_port = int(os.getenv("LUIGI_SCHEDULER_PORT", "8082"))
    kaggle_url = os.getenv(
        "KAGGLE_URL", "jiayuanchengala/aid-scene-classification-datasets"
    )
    scheduler_url = f"http://{scheduler_host}:{scheduler_port}"

    luigi.build(
        [SparkAugmentImages(kaggle_url=kaggle_url)],
        local_scheduler=False,
        scheduler_url=scheduler_url,
        workers=0,
        log_level="DEBUG",
    )

    # luigi_cmd = [
    #     "luigi",
    #     "--module",
    #     "piper.luigi_tasks.luigi_tasks",
    #     "SparkAugmentImages",
    #     "--kaggle-url",
    #     kaggle_url,
    #     "--scheduler-host",
    #     scheduler_host,
    #     "--scheduler-port",
    #     str(scheduler_port),
    #     "--workers",
    #     "1",
    #     "--keep-alive",
    #     "--log-level",
    #     "DEBUG",
    # ]
    # # "luigi --module piper.luigi_tasks.luigi_tasks DownloadKaggleDataset --kaggle-url "jiayuanchengala/aid-scene-classification-datasets" --scheduler-host luigi-scheduler --scheduler-port 8082"
    # subprocess.run(luigi_cmd)


if __name__ == "__main__":
    main()

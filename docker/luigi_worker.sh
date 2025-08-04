#!/bin/bash
while true; do
  echo "Starting Luigi worker"
  luigi \
    --module piper.luigi_tasks.luigi_tasks \
    ${TASK_NAME} \
    --kaggle-url ${KAGGLE_URL} \
    --scheduler-host ${LUIGI_SCHEDULER_HOST:-luigi-scheduler} \
    --scheduler-port ${LUIGI_SCHEDULER_PORT:-8082} \
    --workers 1 \
    --log-level DEBUG
  echo "Luigi worker loop complete. Sleeping..."
  sleep 10
done

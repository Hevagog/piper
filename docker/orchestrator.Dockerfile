FROM piper-base:latest AS orchestrator

# Initiate the pipeline.
CMD [ "uv", "run", "piper/main.py"]
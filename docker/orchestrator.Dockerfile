FROM piper-base:latest AS orchestrator

CMD ["uv", "run", "piper/main.py"]
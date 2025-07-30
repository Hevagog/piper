FROM piper-base:latest AS worker

CMD ["luigid", "--port", "8082", "--address", "0.0.0.0"]

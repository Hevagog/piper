FROM piper-base:latest AS worker

COPY config/ ./

CMD ["luigid", "--port", "8082", "--address", "0.0.0.0"]

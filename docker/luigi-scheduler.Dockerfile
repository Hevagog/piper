FROM piper-base:latest AS scheduler

COPY config/ ./

CMD ["luigid", "--port", "8082", "--address", "0.0.0.0"]

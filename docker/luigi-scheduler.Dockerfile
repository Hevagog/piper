FROM piper-base:latest AS scheduler

# Expose the Luigi scheduler port
EXPOSE 8082
CMD ["luigid", "--port", "8082", "--address", "0.0.0.0"]

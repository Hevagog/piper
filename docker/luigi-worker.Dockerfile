FROM piper-base:latest AS worker

COPY docker/luigi_worker.sh /app/luigi_worker.sh
RUN chmod +x /app/luigi_worker.sh

CMD ["/app/luigi_worker.sh"]
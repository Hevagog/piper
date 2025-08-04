set shell := ["bash", "-euo", "pipefail", "-c"]

# Build and bake commands for Docker
build-bake-all: 
    docker buildx prune  && docker buildx bake 

# Start an interactive shell in a new container for a service
shell NAME:
    docker compose run --rm --service-ports {{NAME}} /bin/bash

run-piper:
    docker compose down && docker compose up --build

# clean up unused images
clean-images:
    docker image prune
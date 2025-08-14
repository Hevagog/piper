# Build bake commands for Docker
build-bake-all: 
    docker buildx prune  && docker buildx bake 

# Build bake commands for base
build-base-base:
    docker buildx bake base

# Build bake commands for nn-worker
build-nn-worker:
    docker buildx bake nn-worker

# Clean build a specific service without cache
build-clean NAME:
    docker buildx bake --pull --no-cache {{NAME}}

# Start an interactive shell in a new container for a service
shell NAME:
    docker compose run --rm --service-ports {{NAME}} /bin/bash

run-piper:
    docker compose down && docker compose up --build

# clean up unused images
clean-images:
    docker image prune

run-changelog:
    git-cliff -o CHANGELOG.md
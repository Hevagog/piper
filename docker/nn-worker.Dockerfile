FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS chef

RUN apt-get update && \
    apt-get install -y curl build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Install cargo-chef to speed up the container builds https://github.com/LukeMathWalker/cargo-chef
RUN cargo install cargo-chef

WORKDIR /app

FROM chef AS planner
COPY src/burn-nn/ .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies to cache them later on
RUN cargo chef cook --release --recipe-path recipe.json

COPY src/burn-nn/ .
RUN cargo build --release --bin burn-nn
RUN test -f /app/target/release/burn-nn || (echo "Error: /app/target/release/burn-nn not found!" && exit 1)

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS runtime

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /app/target/release/burn-nn /usr/local/bin/burn-nn

USER app
WORKDIR /home/app
CMD ["burn-nn"]
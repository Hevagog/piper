FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS base

RUN apt-get update && \
    apt-get install -y curl build-essential

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

FROM spark:4.0.0-scala2.13-java21-ubuntu AS base

WORKDIR /app

USER root

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        wget \
        curl \
        bash \
        git \
        procps \
        net-tools \
        vim \
        unzip \
        software-properties-common \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src/ ./

# Copy luigi configuration
COPY config/ ./

RUN pip3 install --no-cache-dir uv 
    
RUN uv venv && uv pip install .

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app"
ENV LUIGI_CONFIG_PATH="/app/luigi.cfg"

# Kaggle API configuration
RUN mkdir -p /root/.config/kaggle
RUN touch /root/.config/kaggle/kaggle.json && \
    chmod 600 /root/.config/kaggle/kaggle.json

RUN mkdir -p /root/.config/kaggle \
    && echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > /root/.config/kaggle/kaggle.json \
    && chmod 600 /root/.config/kaggle/kaggle.json
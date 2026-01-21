# Start from lightweight Python 3.11 image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Keep Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    build-essential \
    graphviz \
    wget \
    ngspice \
    libngspice0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV NGSPICE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libngspice.so.0


RUN poetry config virtualenvs.create false
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry lock
RUN poetry install --no-root --no-interaction
COPY . .
RUN poetry install --only-root --no-interaction

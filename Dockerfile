ARG WORKDIR=/app

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_PATH=/opt/poetry-venvs \
    NGSPICE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libngspice.so.0
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    build-essential \
    graphviz \
    wget \
    ngspice \
    libngspice0-dev \
    tree \
    sudo \
    && rm -rf /var/lib/apt/lists/*
ARG WORKDIR
WORKDIR ${WORKDIR}
RUN curl -sSL https://install.python-poetry.org | python3 -
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# PDK stage is isolated from dependencies so that a dep change does not
# re-trigger the volare download. In devcontainer use the bind mount at
# /app exposes the host-cached sky130_volare/ (gitignored); this stage is
# used for standalone/production Docker builds where no bind mount exists.
FROM base AS pdk
ARG WORKDIR
COPY scripts ./scripts/
RUN chmod +x scripts/setup_pdk.sh && scripts/setup_pdk.sh

FROM base AS dependencies
ARG WORKDIR
COPY pyproject.toml poetry.lock ./
RUN poetry lock
RUN poetry install --no-root --no-interaction
RUN ln -s "$(poetry env info --path)" /opt/venv

FROM dependencies AS development
ARG WORKDIR
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=pdk ${WORKDIR}/sky130_volare ${WORKDIR}/sky130_volare
COPY . .
RUN poetry install --only-root --no-interaction \
    && chown -R vscode:vscode ${WORKDIR} /opt/poetry-venvs
USER vscode

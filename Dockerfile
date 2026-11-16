# Linux image with both pinned environments, built by the same script as the native setup.
#
#   docker build -t emb3 .
#   docker run --rm -it -v "$PWD/data:/emb3/data" -v "$PWD/results:/emb3/results" emb3
#
# Inside the container, run code with scripts/run.sh <karateclub|main> ...
#
# The image is x86-64 only. The 2022 packages KarateClub needs (gensim 4.2.0 among them)
# have prebuilt wheels for x86-64 Linux but not for ARM Linux, and they no longer compile
# with current build tools. On an ARM host (Apple Silicon) Docker runs it under emulation.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.12.11 AS uv

FROM --platform=linux/amd64 debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PYTHONUNBUFFERED=1

WORKDIR /emb3

# Dependencies first, so that editing source code does not rebuild the environments.
COPY requirements/ requirements/
COPY scripts/setup_envs.sh scripts/setup_envs.sh
RUN scripts/setup_envs.sh && uv cache clean

COPY . .

CMD ["bash"]

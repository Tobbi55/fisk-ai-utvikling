FROM ghcr.io/prefix-dev/pixi:latest

# https://stackoverflow.com/questions/68036484/qt6-qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-thou#comment133288708_68058308
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -qy --no-install-recommends \
        libgl1 libxkbcommon0 libegl1 libdbus-1-3 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxkbcommon-x11-0 libxcb-cursor0 \
        libfontconfig1 libglib2.0-0

WORKDIR /app

COPY pyproject.toml pixi.lock ./
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --frozen

COPY app app

CMD ["pixi", "run", "start"]

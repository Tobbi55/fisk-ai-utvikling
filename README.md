# bachelor-oppgave-nina

Maintainers:
* [Francesco Frassinelli](https://github.com/frafra)

Original authors:
* [Lars Blütecher Holter](https://github.com/Firemines)
* [Benjamin Letnes Bjerken](https://github.com/beuss-git)
* [Lillian Alice Wangerud](https://github.com/Lilliaaw)
* [Daniel Hao Huynh](https://github.com/Mystodan)

[![CI][ci-badge]][ci]

[ci-badge]: https://github.com/beuss-git/bachelor-oppgave-nina/actions/workflows/code-quality.yml/badge.svg
[ci]: https://github.com/beuss-git/bachelor-oppgave-nina/actions/workflows/code-quality.yml

## Requirements
- [pixi](https://pixi.sh/latest/#installation)
- [pre-commit](https://pre-commit.com/#install)

## Installation

Download model weights:
```bash
pixi run fetch-models
```

Install pre-commit hooks:
```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

See the commitlint config:
https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional

## Run Locally

```bash
pixi run start
```

## Testing

Run unit tests:
```bash
pixi run pytest tests
```

Run VDI performance tests (requires GPU and test video data):
```bash
pixi run test
```

## Run using Docker

```
docker compose --profile prod build
docker compose --profile prod run --rm app
```

## Directory Structure

- `data/input/` - Place input video files (.mp4) here for processing
- `data/output/` - Processed video files and detection results are saved here
- `data/models/` - YOLO model weights (.pt files)

The performance test automatically picks the first .mp4 file from `data/input/` and saves results to `data/output/`.

## Notes

Model weights are automatically downloaded by the `fetch-models` task, or you can manually download them from [GitHub release v0.1.0](https://github.com/NINAnor/fisk-ai/releases/tag/v0.1.0).

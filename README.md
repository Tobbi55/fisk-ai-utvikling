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

### Run from VS Code (no manual commands)

You can also start the app using VS Code tasks:
- Open the Command Palette → **Tasks: Run Task**
- Run **Pixi: Install (frozen)** (first time)
- Optional: run **Pixi: Fetch Models**
- Run **Pixi: Start App (GUI)**

### Codespaces GUI (VNC)

Codespaces typically has no `DISPLAY`, so a normal Qt GUI can't open. To run the GUI anyway:
- Run the task **Pixi: Start App (Codespaces VNC)**
- In VS Code, open the **Ports** panel and forward port **5900**
- Connect with a VNC client to `localhost:5900` (VS Code will tunnel it to the codespace)

### Codespaces GUI (browser / noVNC)

If you don't have a VNC client, you can view the GUI in your browser via noVNC:
- Run **Codespaces: Install noVNC** (first time)
- Run **Codespaces: Start GUI in Browser (noVNC)**
- Forward port **6080** in the **Ports** panel and open it in the browser

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

## Export reviewed data to YOLO

After processing a video with the app and deleting false boxes in the **Review Processed Video** dialog,
you can export the corrected frames to a YOLO-style dataset (images + labels).

Example:
```bash
pixi run export-review-yolo -- --video data/output/<name>_processed.mp4 --out datasets/review_exports --val-ratio 0.1
```

This creates a new folder under `datasets/review_exports/` containing `images/`, `labels/`, `meta.jsonl` and `data.yaml`.

# poet-brain

A model trainer.

## Setup

direnv does it: `cd` in and `.envrc` runs `uv sync --frozen` and activates `.venv`
(and loads `.env` if there is one). Without direnv:

```bash
uv sync
```

`task` lists every task.

## Structure

```
poet-brain/
├── data/poems/        # Lauren Ireland's poems: 207 .txt plus SOURCES.tsv
├── src/poet_brain/
├── runs/              # gitignored: checkpoints and anything else a run writes
├── tests/
└── pyproject.toml
```

`data/poems/` is a copy of `sloppy-feelings/data/poems/` at commit `e25cec5`, not a link,
so the training set doesn't move when sloppy-feelings does. The poems are published;
`SOURCES.tsv` says where.

Training runs on the laptop. `modal` is in the dev group for trying remote GPUs; nothing
under `src/` imports it.

## Running

```bash
uv run poet-brain           # console script
uv run python -m poet_brain
```

## Checks

`task ci` runs every check in order, and it's the only thing CI
(`.github/workflows/ci.yml`) runs:

```
lock-check → fmt-check → lint → typecheck → test → build
```

Each stage is a task of its own. `task fmt` applies ruff's fixes and formatting.

ruff and ty aren't in `.venv`. The Taskfile runs them with `uvx` at the versions pinned at
its top, so local runs and CI use the same binaries. ty reads packages from `.venv`, so
`typecheck` syncs it first. uv itself is pinned by `required-version` in `pyproject.toml`.

## Dependencies

```bash
uv add httpx               # runtime
uv add --dev hypothesis    # dev group
```

Commit `uv.lock`.

## Zed

Global Zed settings run ty (with `experimental.useUv`) and Ruff for Python.
`.zed/tasks.json` calls the same tasks (`task: spawn`); `.zed/debug.json` has debugpy
configs (F4).

## Python version

Pinned in `.python-version` (3.15).

---

Ian Furst (ian.furst@gmail.com)

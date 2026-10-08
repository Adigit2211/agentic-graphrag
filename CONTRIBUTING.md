# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install          # editable install + dev + embeddings extras
pre-commit install    # optional; runs ruff, black, mypy on commit
```

## Before opening a PR

```bash
make format
make lint
make typecheck
make test
```

All four must pass; CI runs the same commands on Python 3.11 and 3.12.

## Ground rules

- **No fabricated results.** Benchmark numbers go into the README only after they
  were produced by the benchmark script, with the command, seed and model version.
- **No stub logic in core modules.** If something is out of scope, document it
  under *Limitations* in the README instead.
- Every module is typed (`mypy --strict`), has docstrings and has tests.
- Tests must run offline: mock the LLM (see `tests/conftest.py::FakeLLM`).

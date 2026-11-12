# Contributing

Thanks for helping with `to-weaviate`. Report bugs and ask for new sources in
[GitHub issues](https://github.com/weaviate/to-weaviate/issues).

## Set up

You need Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/weaviate/to-weaviate
cd to-weaviate
uv venv --python 3.10 .venv
uv pip install -e ".[test]" pre-commit
.venv/bin/pre-commit install
```

The `test` extra does not include `pre-commit`, so the `uv pip install`
command names both.

## Before you open a pull request

```bash
.venv/bin/python -m pytest tests/unit
.venv/bin/pre-commit run --all-files
```

The pre-commit hooks run ruff, ruff-format and mypy. CI runs them too, with
the unit tests and the acceptance tests. [Testing](docs/development/testing.md)
explains how to run the acceptance tests on your machine.

- Open the pull request against `main`.
- A behavior change comes with a unit test that fails without it. See
  [Writing tests](docs/development/testing.md#writing-tests).
- A change to what a user sees updates the docs: the
  [README](README.md), [the commands](docs/commands.md), or the guide of the
  source it affects.

## Where to start

- [Architecture](docs/development/architecture.md): the layers and the
  modules.
- [Adding a source](docs/development/adding-a-source.md): the steps for a new
  source database.
- [Testing](docs/development/testing.md): test levels, local servers and CI.
- [Releasing](docs/development/releasing.md): how a version reaches GitHub
  Releases and PyPI.

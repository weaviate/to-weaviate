"""The mypy hook installs its own packages, apart from uv.lock."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# mypy reads types-PyYAML instead of pyyaml.
STUBBED = {"pyyaml"}


def _project_dependencies() -> list[str]:
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^dependencies = \[\n(.*?)^\]", text, re.M | re.S)
    assert block is not None
    return re.findall(r'^\s*"([^"]+)",?\s*$', block.group(1), re.M)


def _mypy_hook_dependencies() -> list[str]:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    (mypy,) = [hook for hook in hooks if hook["id"] == "mypy"]
    return list(mypy["additional_dependencies"])


def test_mypy_hook_uses_the_bounds_of_pyproject() -> None:
    dependencies = _project_dependencies()
    assert len(dependencies) == 6
    expected = [d for d in dependencies if re.split(r"[<>=]", d)[0] not in STUBBED]
    hook = _mypy_hook_dependencies()
    assert [d for d in expected if d not in hook] == []

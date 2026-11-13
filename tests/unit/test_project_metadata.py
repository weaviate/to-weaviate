"""The operating systems that pyproject.toml declares on PyPI."""

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_the_classifiers_name_only_the_tested_operating_systems() -> None:
    # Windows is untested and has no fcntl, so copy and sync run without the
    # lock there (docs/commands.md).
    declared = re.findall(
        r'^\s*"(Operating System :: [^"]+)",$',
        PYPROJECT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )

    assert declared == [
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ]

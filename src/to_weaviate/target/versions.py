import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from to_weaviate.plan import IndexPlan

Version = tuple[int, ...]

# The Weaviate client refuses to connect to an older server.
MINIMUM_VERSION: Version = (1, 27, 0)

_VERSION = re.compile(r"v?(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(text: Any) -> Version | None:
    """(major, minor, patch) of a version text such as "1.39.16"; None otherwise.

    A missing patch is 0. A suffix after the numbers, as in "1.40.0-rc.1", is
    ignored.
    """
    if not isinstance(text, str):
        return None
    match = _VERSION.match(text.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def version_text(version: Version) -> str:
    return ".".join(map(str, version))


def _rq(index: IndexPlan) -> bool:
    return index.quantizer.type == "rq"


@dataclass(frozen=True)
class SettingMinimum:
    """A plan setting of one vector index and the oldest server that applies it."""

    setting: str
    minimum: Version
    applies: Callable[[IndexPlan], bool]


# An older server ignores the setting without an error, or refuses the
# collection. docs/weaviate.md lists the same rows.
SETTING_MINIMUMS: tuple[SettingMinimum, ...] = (
    SettingMinimum("quantizer rq", (1, 32), _rq),
    SettingMinimum(
        "rq with bits 1", (1, 33), lambda i: _rq(i) and i.quantizer.bits == 1
    ),
    SettingMinimum(
        "rq on a flat or dynamic index",
        (1, 34),
        lambda i: _rq(i) and i.type in ("flat", "dynamic"),
    ),
    SettingMinimum("index type hfresh", (1, 36), lambda i: i.type == "hfresh"),
    SettingMinimum(
        "rq with bits 4", (1, 39), lambda i: _rq(i) and i.quantizer.bits == 4
    ),
    SettingMinimum(
        "training_limit of rq",
        (1, 39),
        lambda i: _rq(i) and i.quantizer.training_limit is not None,
    ),
)


def versioned_settings(index: IndexPlan) -> list[SettingMinimum]:
    """The rows of SETTING_MINIMUMS that `index` uses."""
    return [row for row in SETTING_MINIMUMS if row.applies(index)]


def unmet_settings(index: IndexPlan, server: Version) -> list[SettingMinimum]:
    """The settings of `index` that need a newer server than `server`."""
    return [row for row in versioned_settings(index) if server < row.minimum]

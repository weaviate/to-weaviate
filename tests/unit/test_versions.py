import pytest

from to_weaviate import plan as p
from to_weaviate.target.versions import (
    MINIMUM_VERSION,
    parse_version,
    unmet_settings,
    version_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.27.0", (1, 27, 0)),
        ("1.39.16", (1, 39, 16)),
        ("v1.40.4", (1, 40, 4)),
        ("1.34", (1, 34, 0)),
        ("1.40.0-rc.1", (1, 40, 0)),
        ("1.36.24-a1b2c3d", (1, 36, 24)),
        ("", None),
        ("latest", None),
        ("1", None),
        ("x1.30.0", None),
        (None, None),
        (130, None),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_versions_compare_as_numbers():
    older, newer = parse_version("1.9.20"), parse_version("1.10.0")
    assert older is not None and newer is not None and newer > older


def test_version_text():
    assert version_text(MINIMUM_VERSION) == "1.27.0"
    assert version_text((1, 32)) == "1.32"


def index(type_: str, **quantizer: object) -> p.IndexPlan:
    return p.IndexPlan.model_validate({"type": type_, "quantizer": quantizer})


@pytest.mark.parametrize(
    ("planned", "version", "expected"),
    [
        (index("hnsw", type="rq"), "1.31.9", [("quantizer rq", "1.32")]),
        (index("hnsw", type="rq"), "1.32.0", []),
        (index("hnsw", type="rq", bits=8), "1.32.0", []),
        (index("hnsw", type="rq", bits=1), "1.32.27", [("rq with bits 1", "1.33")]),
        (index("hnsw", type="rq", bits=1), "1.33.0", []),
        (
            index("flat", type="rq"),
            "1.33.18",
            [("rq on a flat or dynamic index", "1.34")],
        ),
        (
            index("dynamic", type="rq"),
            "1.33.18",
            [("rq on a flat or dynamic index", "1.34")],
        ),
        (index("dynamic", type="rq"), "1.34.0", []),
        (index("flat", type="rq", bits=1), "1.34.0", []),
        (index("hfresh"), "1.35.23", [("index type hfresh", "1.36")]),
        (index("hfresh"), "1.36.0", []),
        (index("hfresh", type="rq", bits=1), "1.36.0", []),
        (index("hnsw", type="rq", bits=4), "1.38.25", [("rq with bits 4", "1.39")]),
        (index("hnsw", type="rq", bits=4, centering=True), "1.39.0", []),
        (
            index("hnsw", type="rq", training_limit=100),
            "1.38.25",
            [("training_limit of rq", "1.39")],
        ),
        (index("hnsw", type="rq", training_limit=100), "1.39.16", []),
        (
            index("hnsw", type="rq", bits=4),
            "1.27.0",
            [("quantizer rq", "1.32"), ("rq with bits 4", "1.39")],
        ),
        (index("default"), "1.27.0", []),
        (index("hnsw", type="none"), "1.27.0", []),
        (index("hnsw", type="sq", training_limit=100), "1.27.0", []),
        (index("hnsw", type="pq"), "1.27.0", []),
        (index("flat", type="bq"), "1.27.0", []),
        (index("dynamic"), "1.27.0", []),
        (index("hnsw", type="rq", bits=4), "1.100.0", []),
        (index("hnsw", type="rq", bits=4), "2.0.0", []),
    ],
)
def test_unmet_settings(planned, version, expected):
    parsed = parse_version(version)
    assert parsed is not None
    found = [
        (s.setting, version_text(s.minimum)) for s in unmet_settings(planned, parsed)
    ]
    assert found == expected

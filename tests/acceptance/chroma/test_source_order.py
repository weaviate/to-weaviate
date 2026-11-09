"""The order of `get` with offset and limit, which copy relies on.

copy reads a collection in offset partitions, resumes at an offset, and finds
a shift of the source by the ids at the end of each partition. That needs
`get` to return the records in insertion order: a deletion moves each later
record one offset earlier, and an addition goes last. On a distributed Chroma
the case waits until each write is compacted, as the seed does.
"""

from chroma.seed import (
    ADD_BATCH,
    ORDER,
    ORDER_COUNT,
    add_records,
    count,
    delete_collection,
    delete_records,
    make_collection,
    order_records,
    page_ids,
    wait_indexed,
)
from conftest import chroma_kind

from to_weaviate.sources.chroma.client import CHROMA_MAX_PAGE

# The positions in ORDER of the records that the case deletes.
DELETED = (400, 450, 500, 550, 600)
ADDED = 10


def _written(name: str) -> None:
    """On a distributed Chroma, wait until the writes to `name` are compacted."""
    if chroma_kind() == "distributed":
        wait_indexed([name])


def _ids(name: str) -> list[str]:
    """The ids of `name`, read with `get` from offset 0 in pages of CHROMA_MAX_PAGE."""
    found: list[str] = []
    for offset in range(0, count(name), CHROMA_MAX_PAGE):
        found.extend(page_ids(name, offset, CHROMA_MAX_PAGE))
    return found


def test_get_pages_follow_the_insertion_order():
    records = order_records(0, ORDER_COUNT)
    expected = [r.id for r in records]
    # Otherwise an order by id would pass too.
    assert expected != sorted(expected)
    kind = chroma_kind()
    try:
        make_collection(ORDER, records, ADD_BATCH)
        _written(ORDER)
        assert _ids(ORDER) == expected, f"{ORDER} on {kind} Chroma after its adds"

        deleted = [expected[i] for i in DELETED]
        delete_records(ORDER, deleted)
        _written(ORDER)
        expected = [i for i in expected if i not in deleted]
        assert (
            _ids(ORDER) == expected
        ), f"{ORDER} on {kind} Chroma after the deletion of {deleted}"

        added = order_records(ORDER_COUNT, ADDED)
        add_records(ORDER, added)
        _written(ORDER)
        expected += [r.id for r in added]
        assert (
            _ids(ORDER) == expected
        ), f"{ORDER} on {kind} Chroma after the addition of {ADDED} records"
    finally:
        delete_collection(ORDER)

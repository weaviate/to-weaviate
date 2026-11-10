import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from weaviate.classes.config import ConsistencyLevel
from weaviate.exceptions import WeaviateConnectionError

from to_weaviate import plan as p
from to_weaviate.target.deleter import DELETE_BATCH, DeleteFailed, WeaviateDeleter


def make_entry(tenant: str | None = None) -> p.CollectionPlan:
    return p.CollectionPlan(
        source=p.SourceRef(
            ref="r",
            namespace="n",
            name="docs",
            count=1,
            dimension=2,
            distance=None,
            embedding=None,
        ),
        target=p.TargetPlan(
            name="Docs",
            tenant=tenant,
            vectors=[p.VectorPlan(name="default", mode="keep")],
            properties=[
                p.PropertyPlan(source_key="#id", name="source_id", type="text")
            ],
        ),
    )


ENTRY = make_entry()
UUIDS = [str(uuid.UUID(int=i)) for i in range(250)]


class FakeDeleteCollection:
    def __init__(self, client: "FakeDeleteClient") -> None:
        self._client = client
        self.data = SimpleNamespace(delete_many=self._delete_many)

    def with_tenant(self, tenant: str) -> "FakeDeleteCollection":
        self._client.tenants.append(tenant)
        return self

    def with_consistency_level(self, level: Any) -> "FakeDeleteCollection":
        self._client.levels.append(level)
        return self

    def _delete_many(self, where: Any) -> SimpleNamespace:
        self._client.calls.append(where)
        if len(self._client.calls) == self._client.fail_at:
            raise WeaviateConnectionError("connection lost")
        ids = list(where.value)
        failed = 2 if len(self._client.calls) == self._client.partial_at else 0
        return SimpleNamespace(
            successful=len(ids) - failed, failed=failed, matches=len(ids)
        )


class FakeDeleteClient:
    """The part of a WeaviateClient that WeaviateDeleter uses.

    Call `fail_at` (1-based) of `delete_many` raises a connection error once.
    Call `partial_at` reports two of its objects as failed.
    """

    def __init__(
        self, fail_at: int | None = None, partial_at: int | None = None
    ) -> None:
        self.fail_at = fail_at
        self.partial_at = partial_at
        self.calls: list[Any] = []
        self.tenants: list[str] = []
        self.levels: list[Any] = []
        self.collections = SimpleNamespace(get=lambda name: FakeDeleteCollection(self))


def deleter(
    client: FakeDeleteClient, consistency: str | None = None
) -> tuple[WeaviateDeleter, list[float]]:
    sleeps: list[float] = []
    made = WeaviateDeleter(client, consistency, sleep=sleeps.append)  # type: ignore[arg-type]
    return made, sleeps


def test_delete_sends_batches_of_100_and_returns_the_deleted_count():
    client = FakeDeleteClient()
    made, _ = deleter(client)
    assert made.delete(ENTRY, UUIDS) == 250
    assert DELETE_BATCH == 100
    assert [list(call.value) for call in client.calls] == [
        UUIDS[:100],
        UUIDS[100:200],
        UUIDS[200:],
    ]
    assert all(call.target == "_id" for call in client.calls)
    assert all(call.operator.name == "CONTAINS_ANY" for call in client.calls)


def test_delete_applies_the_tenant_and_the_consistency():
    client = FakeDeleteClient()
    made, _ = deleter(client, "QUORUM")
    made.delete(make_entry(tenant="t1"), UUIDS[:5])
    assert client.tenants == ["t1"]
    assert client.levels == [ConsistencyLevel.QUORUM]
    plain = FakeDeleteClient()
    made, _ = deleter(plain)
    made.delete(ENTRY, UUIDS[:5])
    assert plain.tenants == [] and plain.levels == []


def test_delete_retries_a_transient_error():
    client = FakeDeleteClient(fail_at=1)
    made, sleeps = deleter(client)
    assert made.delete(ENTRY, UUIDS) == 250
    assert len(client.calls) == 4 and len(sleeps) == 1


def test_a_batch_that_deletes_only_some_objects_raises_delete_failed():
    client = FakeDeleteClient(partial_at=2)
    made, _ = deleter(client)
    with pytest.raises(DeleteFailed) as info:
        made.delete(ENTRY, UUIDS)
    assert (info.value.collection, info.value.failed, info.value.matched) == (
        "Docs",
        2,
        100,
    )
    assert len(client.calls) == 2


def test_delete_of_nothing_sends_nothing():
    client = FakeDeleteClient()
    made, _ = deleter(client)
    assert made.delete(ENTRY, []) == 0
    assert client.calls == []

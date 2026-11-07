# to-weaviate

`to-weaviate` copies collections from Chroma into Weaviate. It reads each
record's id, vector, document, URI and metadata from Chroma and writes the
records as Weaviate objects. The tool never writes to or deletes from the
source.

## Chroma Cloud dates

Chroma Cloud stops accepting writes on 2026-12-07 and stops serving reads on
2026-12-22. Run the migration from Chroma Cloud before 2026-12-22. After that
date you can migrate only from a Chroma server that you run yourself, for
example one that serves a local Chroma directory.

## Install

From a clone of this repository:

```bash
pip install .
```

## Quickstart

Put the credentials in a `.env` file in the directory where you run the
tool, or export them:

```
CHROMA_API_KEY=<Chroma Cloud API key>
WEAVIATE_URL=<Weaviate Cloud cluster URL>
WEAVIATE_API_KEY=<Weaviate API key>
```

Then run the commands in this order:

```bash
to-weaviate inspect chroma   # writes .to-weaviate/inspect.json and .to-weaviate/plan.yaml
                             # review and edit .to-weaviate/plan.yaml
to-weaviate check            # checks the plan against Weaviate
to-weaviate copy             # creates the collections and copies the records
to-weaviate verify           # compares Weaviate with Chroma
to-weaviate status           # shows the copy progress at any time
```

## Commands

### `inspect chroma`

Reads the Chroma collections and writes `inspect.json` and a draft
`plan.yaml`. The plan names each target collection, its properties and its
vector settings. Review it before you run `check`.

- `--database NAME`, `--collection NAME`: limit the inspection. Both repeat.
- `--sample-pages N`: metadata pages to read per collection. Default: all.
- `--vectors keep|embed|both`: keep the Chroma vectors (default), let a
  Weaviate vectorizer embed the documents, or both.
- `--vectorizer MODULE`, `--vectorizer-option KEY=VALUE`: the Weaviate
  vectorizer module and its options. `plan.yaml` holds each value as text;
  `copy` sends it as a number or a boolean where the module takes one.
- `--index`, `--quantizer`, `--quantizer-bits`: the vector index type, the
  quantizer and the bits for `rq` (1, 4 or 8). `default` leaves the choice
  to the server. `hnsw` takes every quantizer, `flat` and `dynamic` take
  `none`, `rq` and `bq`, and `hfresh` takes `rq` only, as Weaviate 1.40.4
  does. `rq` takes 1, 4 or 8 bits with `hnsw`, 1 or 8 with `flat` and
  `dynamic`, and 1 with `hfresh`; its `centering` needs `hnsw` and 4 bits.
  A quantizer field in `plan.yaml` that the index type does not apply is
  refused too: `cache` of `rq` and `bq`, and `rescore_limit` of `bq`, with
  `hnsw`; `training_limit` of `rq` with `flat`; `cache` and
  `training_limit` of `rq` with `hfresh`. `inspect`, `check` and `copy`
  refuse any other setting. `check` and `copy` also refuse `pq` `segments`
  that do not divide the dimension of the source vectors.
- `--force`: replace an existing `plan.yaml`.

### `check`

Checks `plan.yaml` against the Weaviate server: the plan itself, the
vectorizer modules, and existing collections with other settings. It
creates nothing. Plan entries that write into one collection, each into its
own tenant, must have the same vectors and the same settings for each
property that more than one of them has.

### `copy`

Creates the target collections, then copies the records. Progress goes to
`state.sqlite`, so a second run continues where the first one stopped.
Each run also copies the records added to the source since the last run.
Each record that the tool cannot convert or that Weaviate refuses goes to
`rejects.jsonl` with the reason. When Weaviate refuses every object of a
batch of more than one record, `copy` retries the batch as it retries a
network error. When the retries are used up, it stops with
`target_unreachable` and stores no reject. The next run starts at that page.

A quantizer that Weaviate trains on the vectors (`sq`, `pq`, and `rq` with
`centering`) is enabled again after the last page of each plan entry, so
it compresses the vectors with and without `ASYNC_INDEXING` on the server.
The entry is done only after that update; a run that stops before it sends
the update again and copies no page again.

`state.sqlite` also records the target of each source collection: the
Weaviate host and port, the collection and the tenant. Once records of a
source collection are copied, `copy` stops with `state_mismatch` when
`WEAVIATE_URL` or the target in `plan.yaml` names another target, or when
the target collection no longer exists. To copy to another target, use
another `--dir`. Before any record is copied, a run takes the new target.

One `copy` at a time uses a directory. It holds a lock on `copy.lock` while
it runs, and a second `copy` on that directory exits with `usage`. The
operating system releases the lock when the process ends, so a stopped run
leaves no lock behind. On a system without `fcntl`, such as Windows, `copy`
runs without the lock.

- `--collection NAME`: copy only this plan entry. Repeatable.
- `--parallelism N`: reads per collection, 1 to 8. Default: 5.
- `--consistency ONE|QUORUM|ALL`: the write consistency level.

### `verify`

Compares Weaviate with Chroma: the counts, the full list of ids, and a
sample of records with their properties and vectors. It writes `report.md`
and `report.json`.

`copy` reads a live source by position. A record deleted from the source
during a copy can make `copy` skip other records. Run a full `verify` after
writes to the source have stopped. `verify --quick` skips the id comparison,
so when the counts match it finds a missing record only if the sample holds
that record.

- `--sample N`: records to compare per collection, 1 to 10000. Default: 300.
- `--quick`: skip the comparison of the full id lists.

### `status`

Shows the copy progress and the recorded target of each collection from
`state.sqlite`. It needs no credentials. The total of a collection is the
highest source count that a `copy` read.

## Migration directory

Every command takes `--dir`. The default is `./.to-weaviate`. The directory
holds `inspect.json`, `plan.yaml`, `state.sqlite`, `copy.lock`,
`events.jsonl`, `rejects.jsonl`, `report.md` and `report.json`.

## JSON output

Every command accepts `--json`. With it, stdout holds exactly one JSON
document, and progress lines and warnings go to stderr. Every document has
`status`, which is `success` or `error`. An error document also has `code`,
`message` and `hint`:

```json
{
  "status": "error",
  "code": "auth_failed",
  "message": "Weaviate refused the request: the API key is missing or wrong.",
  "hint": "Check WEAVIATE_API_KEY in the environment or in .env."
}
```

The exit code is the same with and without `--json`:

| Exit code | Meaning | Error codes |
|---|---|---|
| 0 | success | |
| 1 | rejected records or a failed verification | `rejects`, `verification_failed` |
| 2 | usage error, invalid plan, or a state of another target | `usage`, `invalid_plan`, `plan_mismatch`, `target_rejected_schema`, `state_mismatch` |
| 3 | missing credential or authentication failure | `missing_credential`, `auth_failed` |
| 4 | source or target unreachable | `source_unreachable`, `target_unreachable` |
| 5 | internal error | `internal_error` |

## Environment variables

The tool reads its settings from the environment and from `.env` in the
working directory. A value in the environment wins over `.env`. No option
takes a secret, and the tool prints no secret and writes none to a file.

| Variable | Use |
|---|---|
| `CHROMA_URL` | Chroma server. Default: `https://api.trychroma.com` (Chroma Cloud). |
| `CHROMA_API_KEY` | Chroma API key. Required for Chroma Cloud. |
| `CHROMA_TENANT` | Chroma tenant. Default: the tenant of the API key. |
| `CHROMA_DATABASE` | The database that `inspect chroma` reads when you give no `--database`. `--database` wins over it. With neither, `inspect chroma` reads every database of the tenant. |
| `CHROMA_TOKEN_HEADER` | Header that carries the Chroma key. Default: `x-chroma-token`. With `Authorization`, the tool sends `Bearer <key>`. |
| `WEAVIATE_URL` | Weaviate server. Required. |
| `WEAVIATE_API_KEY` | Weaviate API key. |
| `WEAVIATE_GRPC_URL` | Weaviate gRPC endpoint. Set it only for a server outside Weaviate Cloud. |
| `OPENAI_API_KEY`, `COHERE_API_KEY`, ... | Model provider keys. The tool sends them to Weaviate as headers for the vectorizer modules. `API_KEY_MAP` in `src/to_weaviate/settings.py` lists every variable. |
| `TO_WEAVIATE_DEBUG` | `1` prints the traceback when a command fails. |

For a Weaviate server outside Weaviate Cloud, set both URLs, for example
`WEAVIATE_URL=http://localhost:8080` and
`WEAVIATE_GRPC_URL=http://localhost:50051`. Without `WEAVIATE_GRPC_URL`
the tool connects to port 443 of the `WEAVIATE_URL` host over TLS, so it
refuses a `WEAVIATE_URL` that names another port.

## Migrate from a local Chroma directory

You can serve a Chroma directory on disk, for example one that
`chromadb.PersistentClient(path=...)` wrote, with the `chroma` command from
the `chromadb` package:

```bash
pip install chromadb
chroma run --path /path/to/chroma-directory --port 8000
```

Then point the tool at that server. A server without authentication needs
no `CHROMA_API_KEY`:

```bash
export CHROMA_URL=http://localhost:8000
to-weaviate inspect chroma
```

## Development

The `test` extra does not install `pre-commit`, so install it next to it:

```bash
uv venv --python 3.10 .venv
uv pip install -e ".[test]" pre-commit
.venv/bin/python -m pytest tests/unit
.venv/bin/pre-commit run --all-files
```

### Acceptance suite

`tests/acceptance` runs the installed `to-weaviate` command against a real
Chroma server and a real Weaviate server. The tests start no server. They
run only against the test endpoints, so the four variables must have
exactly these values:

```bash
WEAVIATE_URL=http://localhost:8099
WEAVIATE_GRPC_URL=http://localhost:50099
WEAVIATE_API_KEY=admin-key
CHROMA_URL=http://localhost:8001
```

When a variable is missing or has another value, the suite skips every test
and names that variable. With `TO_WEAVIATE_REQUIRE_ACCEPTANCE=1`, as in CI,
the suite fails with the same text instead. It connects to no server before this check passes.
The commands it starts get empty values for `CHROMA_DATABASE`,
`CHROMA_TENANT`, `CHROMA_API_KEY` and `CHROMA_TOKEN_HEADER`, so values from
your shell do not reach them.

The suite deletes and recreates the Chroma collections that `COLLECTIONS`
in `tests/acceptance/chroma/seed.py` lists. One test adds 10 records to
`tiny.l2` and deletes them when it ends. Before and after each test the
suite deletes the Weaviate collections that `SUITE_COLLECTIONS` in
`tests/acceptance/conftest.py` lists. It writes to no other Weaviate
collection, also when the test Chroma holds collections that the seed does
not make.

One test is marked `xfail(strict=True)`: the copy of a dynamic index. It
describes a finding that is not fixed and expects a Weaviate server without
`ASYNC_INDEXING`, as in CI. Against a server with `ASYNC_INDEXING=true` it
can pass, and strict mode then reports it as a failure.

The `Makefile` starts the servers and sets the variables. `make` alone
prints the targets. `make acceptance-up` and `make acceptance-down` call
`tests/acceptance/cluster.sh up` and `down`. `make acceptance-up` needs a
clone of [weaviate-local-k8s](https://github.com/weaviate/weaviate-local-k8s)
at `LOCAL_K8S` (default: `../weaviate-local-k8s`), Docker, and every program
that `local-k8s.sh` uses: `kind`, `helm`, `kubectl`, `curl`, `nohup`, `jq` and
`lsof`, plus `gtimeout` on macOS (from coreutils) or `timeout` on Linux.

```bash
make acceptance-up     # Chroma on port 8001, Weaviate on ports 8099 and 50099 in a kind cluster
source .venv/bin/activate
make acceptance        # pytest tests/acceptance -v
make acceptance-down   # stops Chroma, then asks before it deletes the kind cluster
```

`make acceptance-up` creates a kind cluster named `weaviate-k8s` and
switches the kubectl context to `kind-weaviate-k8s`. Before it starts
anything, it checks that `kind` is on `PATH`, that
`$LOCAL_K8S/local-k8s.sh` is executable, and that `kind get clusters`
succeeds and lists no cluster named `weaviate-k8s`, because
`local-k8s.sh setup` would reuse that name. After `setup` succeeds, it
writes the id of the container `weaviate-k8s-control-plane` to
`.acceptance-kind-cluster`, which git ignores. When `setup` fails, it writes
no marker, so `make acceptance-down` will not remove a partly created
cluster. It then prints the command that removes the cluster by hand:
`kind delete cluster --name weaviate-k8s`.

`make acceptance-down` always stops Chroma. It deletes the cluster only when
`.acceptance-kind-cluster` exists and holds the id of the
`weaviate-k8s-control-plane` container that exists now. When that container
is not found or has another id, it removes the stale marker and leaves the
cluster alone. Otherwise it lists every effect of `local-k8s.sh clean` and
runs it only after you type `y` on a terminal. With stdin from a pipe or a
file, it does not run `clean` and prints the command to run by hand. `clean` deletes the kind cluster
`weaviate-k8s`, switches the kubectl context to `kind-weaviate-k8s` first,
and kills every `kubectl-relay` process and every
`kubectl get pods -n weaviate -w` process on the machine. After it, no
kubectl context is current, so select your previous context again with
`kubectl config use-context NAME`.

`make acceptance` works with any servers on those ports: Chroma from
`tests/acceptance/chroma/docker-compose.yml`, and a Weaviate server that
accepts the API key `admin-key`. The Weaviate version of CI is in
`tests/acceptance/weaviate-version`.

### Cloud suite and releases

`tests/cloud` runs the installed `to-weaviate` command against Chroma Cloud
and Weaviate Cloud. It reads the Chroma collection `to-weaviate-ci` and
writes nothing to Chroma. In Weaviate it deletes the collection
`To_weaviate_ci` before and after its test, and no other collection.

The suite runs whenever `WEAVIATE_URL`, `WEAVIATE_API_KEY` and either
`CHROMA_API_KEY` or `CHROMA_URL` are set. It does not check which servers
they name, so a bare `pytest` in a shell with these variables runs it
against those servers. The commands it starts get your environment
unchanged. When a variable is missing, the suite skips and names it. With
`TO_WEAVIATE_REQUIRE_CLOUD=1`, as in the release workflow, it fails
instead.

To run it against the local test servers, seed the local Chroma with
`CHROMA_URL=http://localhost:8001`, `CHROMA_TENANT=default_tenant` and
`CHROMA_DATABASE=default_database`, then also set `WEAVIATE_URL`,
`WEAVIATE_GRPC_URL` and `WEAVIATE_API_KEY` to the acceptance values.

`.github/workflows/release.yml` runs when a tag that matches
`v[0-9]+.[0-9]+.[0-9]+*` is pushed. The `build` job stops unless the tag
points at a commit on `main` and the built wheel has the tag's version
without the `v`. The `cloud` job installs that wheel and runs
`tests/cloud`. The `publish` job uploads the wheel and the sdist to PyPI
through trusted publishing. It uses no token.

A person with access to the GitHub repository and to PyPI sets up three
things once:

- The GitHub environment `cloud-tests` holds five secrets. The `cloud` job
  passes each one to the suite as the variable in the last column.

  | Secret | Meaning | Variable |
  |---|---|---|
  | `CHROMA_CLOUD_API_KEY` | API key for the Chroma Cloud database that holds `to-weaviate-ci` | `CHROMA_API_KEY` |
  | `CHROMA_CLOUD_TENANT` | Chroma Cloud tenant of that database | `CHROMA_TENANT` |
  | `CHROMA_CLOUD_DATABASE` | Name of that database | `CHROMA_DATABASE` |
  | `WEAVIATE_CLOUD_URL` | URL of the Weaviate Cloud cluster that the job copies into | `WEAVIATE_URL` |
  | `WEAVIATE_CLOUD_API_KEY` | API key of that cluster | `WEAVIATE_API_KEY` |

- PyPI has a pending trusted publisher for the project `to-weaviate`:
  repository `weaviate/to-weaviate`, workflow `release.yml`, environment
  `pypi`.
- Both environments, `cloud-tests` and `pypi`, allow deployments from tags
  that match `v*`. GitHub rejects the `cloud` or `publish` job of a tag
  that its environment does not allow.

The Chroma collection is created once, by hand, before 2026-12-07, when
Chroma Cloud stops accepting writes. Export `CHROMA_API_KEY`,
`CHROMA_TENANT` and `CHROMA_DATABASE` with the values of the three Chroma
secrets, then run the seed from the development environment:

```bash
.venv/bin/python tests/cloud/chroma/seed.py
```

It creates `to-weaviate-ci` with 2,000 records and prints
`Created to-weaviate-ci with 2000 records.` It reads the environment, not
`.env`. When the collection exists, it stops and changes nothing. It
deletes nothing. When it stops after it created the collection, for
example because an `add` call failed, the collection is incomplete. Delete
`to-weaviate-ci` by hand in Chroma Cloud, then run the seed again.

A release is a tag on a commit of `main`. The `build` job fails for a tag
on any other commit:

```bash
git fetch origin
git tag v0.1.0 origin/main
git push origin v0.1.0
```

When the three jobs pass, `uvx to-weaviate --version` prints
`to-weaviate 0.1.0`.

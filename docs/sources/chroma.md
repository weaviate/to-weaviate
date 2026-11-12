# Chroma

`to-weaviate inspect chroma` reads collections from Chroma. For each record
the tool copies its id, document, URI and metadata, and its embedding when
the plan keeps the source vector. It never writes to Chroma.

> [!IMPORTANT]
> Chroma Cloud stops accepting writes on 2026-12-07 and stops serving reads
> on 2026-12-22. Migrate from Chroma Cloud before 2026-12-22. After that you
> can migrate only from a Chroma server you run yourself, for example one
> that serves a [Chroma directory on disk](#a-chroma-directory-on-disk).

## Deployments

| Deployment | Notes |
|---|---|
| Chroma Cloud | The default `CHROMA_URL`. Needs `CHROMA_API_KEY`. |
| Distributed Chroma | The open-source cluster. `copy`, `sync` and `verify` wait for its compaction; see [Compaction](#compaction). |
| Single-node Chroma | A `chroma run` server or the `chromadb/chroma` container. |
| A Chroma directory on disk | Serve it with `chroma run`, then read it as a single-node server. |

## Connection settings

| Variable | Required | Use |
|---|---|---|
| `CHROMA_URL` | no | The Chroma server. Default: `https://api.trychroma.com` (Chroma Cloud). |
| `CHROMA_API_KEY` | for Chroma Cloud | The Chroma API key. A server without authentication needs none. |
| `CHROMA_TENANT` | no | The Chroma tenant. Default: the tenant of the API key. |
| `CHROMA_DATABASE` | no | The database that `inspect chroma` reads when you pass no `--database`. With neither, it reads every database of the tenant. |
| `CHROMA_TOKEN_HEADER` | no | The header that carries the key. Default: `x-chroma-token`. With `Authorization`, the tool sends `Bearer <key>`. |

`inspect` records `CHROMA_URL`, without its userinfo, query and fragment,
and `CHROMA_TENANT`, or null when it is unset, in `plan.yaml`. `copy` and
`verify` compare both with the current values as text, and stop with
`invalid_plan` before they connect when either differs. Set both as you set
them for `inspect`: a trailing `/`, another letter case in the host, or a
tenant that was unset then also counts as a difference.

The target settings are on [Weaviate target](../weaviate.md).

## `inspect chroma` options

| Option | Effect |
|---|---|
| `--database NAME` | Read this database. Repeatable. Default: `CHROMA_DATABASE`, else every database. |
| `--collection NAME` | Read this collection. Repeatable. Default: every collection. |
| `--sample-pages N` | Read at most N pages of 300 records per collection to find the metadata keys and their types. Default: every page. |

`inspect chroma` also takes the vector options that every source shares:
`--vectors`, `--vectorizer`, `--vectorizer-option`, `--index`, `--quantizer`,
`--quantizer-bits` and `--force`. See [Commands](../commands.md#inspect-source).

On a large collection, `--sample-pages` makes `inspect` faster. `inspect`
spreads the sampled pages over the whole collection. A metadata key that
appears only in pages it did not sample is missing from the plan, and `copy`
rejects each record that has it.

## Examples

### Chroma Cloud

```dotenv
CHROMA_API_KEY=<Chroma Cloud API key>
WEAVIATE_URL=https://my-cluster.weaviate.cloud
WEAVIATE_API_KEY=<Weaviate API key>
```

```bash
to-weaviate inspect chroma --database my-database
```

When the key can read several tenants, set `CHROMA_TENANT` too.

### A Chroma server you run

```dotenv
CHROMA_URL=http://localhost:8000
CHROMA_API_KEY=<token, if the server has authentication>
CHROMA_TOKEN_HEADER=Authorization
```

Leave out `CHROMA_TOKEN_HEADER` when the server reads the token from
`x-chroma-token`.

### A Chroma directory on disk

The `chroma` command of the `chromadb` package serves a directory that
`chromadb.PersistentClient(path=...)` wrote:

```bash
pip install chromadb
chroma run --path /path/to/chroma-directory --port 8000
```

Then point the tool at that server. It needs no API key:

```bash
export CHROMA_URL=http://localhost:8000
to-weaviate inspect chroma
```

## How the data maps to Weaviate

| Chroma | Weaviate |
|---|---|
| Collection | A collection. `inspect` turns the name into one Weaviate accepts: `docs-cosine` becomes `Docs_cosine`, and a name that does not start with a letter gets the prefix `C_`. |
| Record id | The object UUID. A canonical lowercase UUID stays as it is; any other id becomes a UUIDv5 of the id. The `source_id` property holds the original id. |
| Embedding | The vector `default`, or `source` with `--vectors both`. Not copied with `--vectors embed`. With `embed` or `both`, a collection whose sampled documents are empty keeps the embedding as `default`, and the plan notes it. |
| Document | The `document` property, `text` with `word` tokenization. |
| URI | The `uri` property. |
| Metadata key | A property. See [types and conversion](../plan-file.md#types-and-conversion). |
| Sparse-vector metadata key | Not migrated. The plan lists it under `dropped`. |

### Distances

The distance comes from the collection's `hnsw` or `spann` configuration,
or from the `hnsw:space` metadata of collections that Chroma 0.5 created.
Without one, Chroma uses `l2`.

| Chroma | Weaviate |
|---|---|
| `l2` | `l2-squared` |
| `cosine` | `cosine` |
| `ip` | `dot` |

### Embedding models

When a collection stores its embedding function, Weaviate has a module for
that model, and you run `inspect` without `--vectorizer`, `inspect` sets the
module as the vectorizer of the kept vector `default`. Weaviate can then
embed query text with the same model. With `--vectors both`, the vector
`source` gets no vectorizer.

| Chroma embedding function | Weaviate module |
|---|---|
| `openai` | `text2vec-openai` |
| `cohere` | `text2vec-cohere` |
| `jina` | `text2vec-jinaai` |
| `voyageai` | `text2vec-voyageai` |
| `mistral` | `text2vec-mistral` |
| `huggingface` | `text2vec-huggingface` |
| `google_generative_ai`, `google_gemini`, `google_vertex` | `text2vec-google` |
| `ollama` | `text2vec-ollama` |

The plan carries the model name and, where the module takes them, the
dimensions, the base URL or endpoint, the project and the location. When a
function gets no vectorizer, the plan notes why:

- `default`: Chroma's `all-MiniLM-L6-v2` has no Weaviate module. Embed
  queries in your application, use `text2vec-huggingface` with that model,
  or embed the documents again with `--vectors embed`.
- `chroma-cloud-qwen`: no Weaviate module serves this model.
- `openai` on Azure, a function that lacks a setting the module needs (such
  as the model name of `voyageai` or the project of `google_vertex`), or any
  other function: no mapping.

Chroma stores some embedding functions without a name (type `legacy`). For
such a collection `inspect` stores no embedding, sets no vectorizer and
writes no note.

`inspect.json` and `plan.yaml` store only the configuration keys that the
mapping reads. They never store `default_headers`, because a header can
hold a token.

## Compaction

Distributed Chroma and Chroma Cloud keep new writes in a log until a
compaction moves them into the collection. A compaction of more than 5,000
records gives them new positions, so a read by position during it skips
records. To prevent that, `copy`, `sync` and `verify` do two things on these
deployments:

- Before the first page of each collection, they ask Chroma every 5 s how
  many writes are left to compact, until none is left. They give up at the
  first answer after 600 s that still counts one. Each time the number changes
  they print `Waiting for Chroma to compact <name>: <N> of <M> operations not
  compacted`, and `copy` and `sync` write the event `compaction_wait`.
- After the last read, they check whether Chroma compacted the collection
  during the read. When it did, they stop with `source_compacting`, exit 4.
  The next `copy` or `sync` reads that collection again from the start, and
  `verify` writes no report.

A collection that receives writes all the time does not reach zero within
600 s, so `copy` and `sync` stop with `source_compacting` before their first
page. Pause the writes to the source while they run.

Single-node Chroma has no compaction. The tool sends one `indexing_status`
request and does not wait.
A Chroma server older than 1.4.1 has no `indexing_status` route, and the
tool reads it without the wait, as a single-node server.

## Writes during a copy

On Chroma Cloud and distributed Chroma, pause the writes while `copy` and
`sync` run: a collection that keeps receiving writes stops them with
`source_compacting` (see [Compaction](#compaction)). Writes made between two
runs are copied by the next `sync`.

On single-node Chroma and a Chroma directory on disk, the application can
keep writing while `copy` and `sync` run. Stop the writes before the last
`sync --prune`.

## Limits

- Records come in pages of 300, the most that Chroma Cloud returns per
  request.
- The tool tries a Chroma request up to 7 times when it fails with a network
  error, a timeout, or HTTP 429, 500, 502, 503 or 504, and waits 15.75 s to 31.5 s in total between the
  attempts. A `Retry-After` header raises each wait to its value, up to 30 s
  per wait, so these retries can take longer.
- A record deleted from Chroma during a copy can make `copy` skip another
  record; the next `sync` writes it. See [`verify`](../commands.md#verify).
  The same deletion during the id listing of `sync --prune` makes the
  listing skip a record; `sync` reads each stale record by id before it
  deletes the object, so that object stays.
- Sparse vectors are not migrated.

## Implementation

This section is for developers. The general design is in
[Architecture](../development/architecture.md).

### Modules

| Module | Holds |
|---|---|
| `src/to_weaviate/sources/chroma/client.py` | `ChromaClient`: the HTTP client for Chroma's `/api/v2` REST API, with retries, redaction and `ChromaError` |
| `src/to_weaviate/sources/chroma/source.py` | `ChromaSource`: the [`Source` protocol](../development/architecture.md#the-source-protocol) for Chroma |
| `src/to_weaviate/sources/chroma/config.py` | The distance, the embedding function and the sparse-vector keys of a collection's configuration |
| `src/to_weaviate/sources/chroma/mapping.py` | Embedding function to Weaviate vectorizer, and the secret-free form that `inspect` stores |

`chroma_settings` in `src/to_weaviate/settings.py` reads the `CHROMA_*`
variables.

### How it reads

- `inspect` lists the tenant's databases when neither `--database` nor
  `CHROMA_DATABASE` names one, then the collections of each database in
  pages of 1,000. For each collection it reads the count and pages of
  metadata to collect each key's types. It then reads the first page and up
  to 10 spread pages of documents to estimate the document size.
- `read` uses `/get` with an offset and a limit of at most 300. Chroma
  returns records in insertion order, so `partitions` splits a collection
  into offset ranges, one per thread. A deletion moves the later records one
  position down, and an addition goes after the last record. The copy engine
  relies on that order to resume and to find new records.
- `ids` and `ids_at` read ids in the same order, for `verify` and for the
  copy engine's search for its place after deletions.
- `get` uses `/get` with up to 300 ids, for `sync`. A server can return fewer
  rows than asked, so `get` asks again for the ids a response left out,
  until a response holds none of them.
- `wait_compacted` polls `/indexing_status` until `num_unindexed_ops` is 0
  and returns the collection's `version` and `log_position`.
  `compaction_state` reads them again after the last page, and the engine
  compares the two. Single-node Chroma answers `/indexing_status` with HTTP
  500, and a server older than 1.4.1 with 404. In both cases the source
  reports no compaction and the engine compares nothing.

### Tests

- Unit tests: `tests/unit/test_chroma_client.py`, `test_chroma_config.py`,
  `test_chroma_mapping.py`, `test_chroma_source.py` and `test_cli_chroma.py`.
  `tests/unit/chroma_server.py` serves a fake Chroma through
  `httpx.MockTransport`, so the CLI tests run the real client.
- Acceptance tests: `tests/acceptance/chroma/`. `seed.py` creates the test
  collections. Locally and in the CI job `acceptance (single)` they read the
  single-node Chroma of `tests/acceptance/chroma/docker-compose.yml`. The CI
  job `acceptance (cluster)` reads distributed Chroma, which
  `tests/acceptance/chroma-k8s.sh` deploys from Chroma's Helm chart.
  `test_source_order.py` checks the page order in both jobs, and its
  compaction case runs in `acceptance (cluster)` only.

[Testing](../development/testing.md) explains how to run them.

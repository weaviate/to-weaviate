# Weaviate target

The target of every migration is one Weaviate instance: a Weaviate Cloud
cluster or a server you run yourself. The tests run against the Weaviate
version in
[`tests/acceptance/weaviate-version`](../tests/acceptance/weaviate-version).

## Supported Weaviate versions

| Weaviate server | State |
|---|---|
| 1.39.16 and later | Supported. Every acceptance test passes on 1.39.16 and on the version in [`tests/acceptance/weaviate-version`](../tests/acceptance/weaviate-version), which CI runs. |
| 1.27.0 to 1.39.15 | Supported, except for the plan settings in the next table, which need a newer server. |
| Older than 1.27.0 | Not supported. The Weaviate Python client refuses the server, and every command that connects stops with `target_version_unsupported`, exit 2. The message names the server version. |

The acceptance tests were run on 1.27.27, 1.30.23, 1.32.27, 1.33.18,
1.35.23, 1.37.18, 1.39.16 and 1.40.4. A selection of them was run on 1.27.0,
1.34.20, 1.36.24 and 1.38.25. The versions 1.39.0 to 1.39.15 were not run.

The server must report a release version such as `1.39.16`. The Weaviate
Python client cannot read a version text such as `nightly` or
`preview-1.39.0`. For such a server, and for a server that reports no
version, every command that connects stops with `target_version_unsupported`,
exit 2. Check that `WEAVIATE_URL` is the server you mean, and use a server
that reports a release version, 1.27.0 or later.

### Minimum version per plan setting

| Plan setting | Minimum server | How the minimum is known |
|---|---|---|
| Quantizer `rq` | 1.32 | Read in the server source. Run: 1.27.27 and 1.30.23 ignore `rq` without an error and create the collection uncompressed. |
| `rq` with `bits: 1` | 1.33 | Read in the server source, not run. |
| `rq` with the index type `flat` or `dynamic` | 1.34 | Run for `dynamic`: 1.33.18 leaves the flat part uncompressed without an error, 1.34.20 applies it. Read in the server source for `flat`, not run. |
| Index type `hfresh` | 1.36 | Run: 1.35.23 refuses it, 1.36.24 accepts it. |
| `rq` with `bits: 4`, which `centering` needs | 1.39 | Run: 1.38.25 refuses it, 1.39.16 accepts it. |
| `training_limit` of `rq` | 1.39 | Run: 1.38.25 does not store it, 1.39.16 does. |

`check` reads the server version and reports each setting of the plan that
the server is too old for as an error, `server_too_old`. The error names the
vector, the setting, the minimum and the server version. When the plan has
settings with different minimums, each error for a lower minimum also names
the highest one, which is the version to upgrade to. `copy` and `sync`
make the same check for the collections they write, and stop with
`invalid_plan`, exit 2, before they create a collection. To go on, upgrade
Weaviate or change the setting in `plan.yaml`. For a collection that exists,
a changed index or quantizer stops `sync` with `plan_mismatch`; see
[Edits you can make](plan-file.md#edits-you-can-make).

The tool compares the server version with the first release of the minimum:
1.39.0 for 1.39. The tool reads the version again after the client has
connected. When that answer has no version text that it can compare, it
refuses no setting, and `check`, `copy` and `sync` warn with
`server_version_unknown` for each collection that uses a setting of the
table.

Two more differences were seen in the runs on servers older than 1.33:

- The server does not enable the vectorizer modules that call a model
  provider, such as `text2vec-openai`, by default. `check` reports a module
  that is not enabled as `module_not_enabled`.
- With `--vectors both`, a record without a document gets the embedded
  vector `default` next to its `source` vector. On 1.33.18 and later it gets
  only `source`.

## Connection settings

Set these variables in the environment or in `.env`:

| Variable | Required | Use |
|---|---|---|
| `WEAVIATE_URL` | yes | The Weaviate server, for example `https://my-cluster.weaviate.cloud` or `http://localhost:8080`. |
| `WEAVIATE_API_KEY` | no | The API key. Leave it unset for a server without authentication. |
| `WEAVIATE_GRPC_URL` | for a server outside Weaviate Cloud | The gRPC endpoint. Leave it unset for Weaviate Cloud. |

### Weaviate Cloud

Set `WEAVIATE_URL` and `WEAVIATE_API_KEY`, and leave `WEAVIATE_GRPC_URL`
unset. The tool then connects over TLS on port 443: REST to the
`WEAVIATE_URL` host and gRPC to the host `grpc-<WEAVIATE_URL host>`. It
refuses a `WEAVIATE_URL` with another port.

A free (sandbox) cluster accepts only the index type `hfresh`. It refuses
`hnsw`, `flat`, `dynamic` and their quantizer settings with a message that
`copy` prints. For a free cluster, draft the plan with `--index hfresh`.

### A server you run yourself

Set both URLs, for example:

```dotenv
WEAVIATE_URL=http://localhost:8080
WEAVIATE_GRPC_URL=http://localhost:50051
```

## Permissions

`check` reads the module list and the schema. `copy` reads the module list,
the schema and the tenants, creates collections and tenants, adds
properties, updates quantizer settings and writes objects. `sync` does the
same and reads the objects; with `--prune` it also deletes objects. `verify`
reads the schema and the tenants, counts the objects and reads them. A key
without the permission for one of these steps makes the command stop with
`auth_failed`, exit 3.

`check` reads the nodes of the cluster, to check the
[replication factor](#replication). `copy` and `sync` read them only when
they create a collection with `replication_factor: auto`. With role-based
access control, that is the permission to read the nodes with minimal
verbosity (`read_nodes`). A key without it does not stop them.

`copy` and `sync` create every target collection before they write the
first object. When the key may not create a collection, they stop before
they write any object. A collection that they created earlier in the same
run stays.

## Replication

`replication_factor` of a plan entry sets how many nodes hold each object of
a collection that `copy` or `sync` creates. `inspect` drafts `auto`.

| Value | Effect |
|---|---|
| `auto` | `copy` reads the node count and creates the collection with a factor of 3, or with the node count on a cluster of fewer than 3 nodes. |
| A number from 1 | `copy` creates the collection with that factor. |

`copy` and `sync` print the factor of each collection they create, for
example `Created Docs with replication factor 3 (3 nodes)`. With `--json`,
the line goes to stderr and the document lists the collection under
`created_collections`.

Weaviate cannot raise the factor of a collection through its schema once
the collection exists, and the tool never changes it. Choose the factor
before the first `copy`.

With `auto`:

- A key that may not read the nodes makes `copy` send no factor. The server
  then applies its own: its `REPLICATION_MINIMUM_FACTOR`, which is 1 unless
  the server sets it. `check` warns before the collection exists, and `copy`
  prints the factor that the server chose and warns
  (`unknown_node_count`). To choose the factor, set `replication_factor` to
  a number, or give the key the permission to read the nodes.
- When the server refuses the factor, `copy` creates the collection once
  more without a factor and prints the factor that the server chose. A
  server refuses a factor below its `REPLICATION_MINIMUM_FACTOR`. `copy`
  warns when the server chose a lower factor than the refused one
  (`low_replication_factor`).
- An existing collection is accepted with the factor it has. `check` warns
  when that factor is 1 and the cluster has more than one node
  (`not_replicated`).

To change the factor of a collection that exists, delete the collection and
copy it again with another `--dir`. `copy` then creates it with the factor of
the plan.

With a number:

- `check` reports `too_few_nodes` when the cluster has fewer nodes than the
  factor and the collection does not exist yet. When the key may not read
  the nodes, it warns instead (`unknown_node_count`).
- A factor that the server refuses stops `copy` with
  `target_rejected_schema`, exit 2.
- An existing collection with another factor does not match the plan.
  `check` reports `plan_mismatch`, and `copy` stops with `plan_mismatch`,
  exit 2, before it writes an object.

Entries that share a collection, one tenant each, need the same
`replication_factor`.

A Weaviate Cloud cluster with high availability has 3 nodes and sets its
minimum factor to 3. It creates every collection with factor 3 at least,
also when the request sends no factor, and it refuses a lower one. `auto`
sends 3 there.

## Vectorizer keys

When the plan uses a vectorizer module that calls a model provider, such as
`text2vec-openai`, set the provider's key in the environment. The tool sends
it to Weaviate as a request header and never prints it.

| Provider | Variables |
|---|---|
| OpenAI | `OPENAI_APIKEY`, `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_APIKEY`, `AZURE_API_KEY` |
| Cohere | `COHERE_APIKEY`, `COHERE_API_KEY` |
| Google (Vertex AI, AI Studio) | `VERTEX_APIKEY`, `VERTEX_API_KEY`, `STUDIO_APIKEY`, `STUDIO_API_KEY` |
| Hugging Face | `HUGGINGFACE_APIKEY`, `HUGGINGFACE_API_KEY` |
| Jina AI | `JINAAI_APIKEY`, `JINAAI_API_KEY` |
| Mistral | `MISTRAL_APIKEY`, `MISTRAL_API_KEY` |
| Voyage AI | `VOYAGE_APIKEY`, `VOYAGE_API_KEY` |
| Others | `API_KEY_MAP` in [`settings.py`](../src/to_weaviate/settings.py) lists every variable. |

When two variables name the same header, the first one in `API_KEY_MAP` that
is set wins. The table lists them in that order.

`check`, `copy` and `sync` stop with `missing_credential`, exit 3, before
they connect, when a planned `text2vec-openai`, `text2vec-cohere`,
`text2vec-jinaai`, `text2vec-voyageai`, `text2vec-mistral`,
`text2vec-huggingface` or `text2vec-google` vectorizer has no key. They do
not check the keys of other modules. A provider that refuses the key during
`copy` or `sync` stops it with `auth_failed`, exit 3. The message names the provider,
and the hint names its variables.

## Async indexing

Weaviate Cloud runs with `ASYNC_INDEXING=true`. `ASYNC_INDEXING` changes two
things for a migration:

- A `dynamic` index needs it. `check` warns when the plan has a `dynamic`
  index (`needs_async_indexing`).
- A trained quantizer compresses a shard only once the shard holds
  `training_limit` vectors. See [trained quantizers](plan-file.md#trained-quantizers).

## Compression

Weaviate Cloud compresses the vectors of a new collection by default. That
includes the vectors that a migration keeps from the source (mode `keep`).
This is the intended setup. Compression lowers the memory that the cluster
needs, and with it the cost. Weaviate
[rescores](https://docs.weaviate.io/weaviate/concepts/vector-quantization)
the results of a search with the original vectors, so the results stay close
to those of an uncompressed search. The
[RQ guide](https://docs.weaviate.io/weaviate/configuration/compression/rq-compression)
describes the recall of each bit width.

The tool leaves this choice to the server. `inspect` drafts the index type
`default` and the quantizer `default`, unless you pass `--index` or
`--quantizer`. `copy` sends no index settings for the index type `default`
and no quantizer settings for the quantizer `default`. The server then
applies its default quantization. A server you run yourself has one only
when its `DEFAULT_QUANTIZATION` variable is set.

### Keep the vectors uncompressed

Set the quantizer of each vector to `none`. `none` needs the index type
`hnsw`, `flat` or `dynamic`. `inspect --index hnsw --quantizer none` drafts
every vector that way. In `plan.yaml`, a vector with `none` looks like this:

```yaml
    vectors:
    - name: default
      mode: keep
      vectorizer: null
      distance: cosine
      index:
        type: hnsw
        quantizer:
          type: none
```

`copy` sends `none` as `skipDefaultQuantization: true`, and the server does
not apply its default to that vector.

- `hfresh` does not take `none`. It always compresses, with 1-bit `rq`. A
  free cluster accepts only `hfresh`, so it cannot keep the vectors
  uncompressed.
- A server can limit the compression types it accepts
  (`ALLOWED_COMPRESSION_TYPES`). When the list leaves out `none`, the server
  refuses the collection, and `copy` stops with `target_rejected_schema`.
  `check` does not see this limit.

### Compression cannot be turned off

Weaviate cannot turn compression off on an existing collection. To undo it,
copy the source again into a new collection: draft a plan with `none` in a
new migration directory (`--dir`), give the target a name that does not
exist yet, and run `copy`.

A plan with `none` for a collection that the server already compressed does
not match that collection. `check` reports `plan_mismatch`, and `copy` stops
with `plan_mismatch`, exit 2, before it writes an object.

`verify` is not affected by compression. It compares the vectors that
Weaviate returns, and Weaviate returns the original vectors, which it stores
next to the compressed ones.

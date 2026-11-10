# Weaviate target

The target of every migration is one Weaviate instance: a Weaviate Cloud
cluster or a server you run yourself. The tests run against the Weaviate
version in
[`tests/acceptance/weaviate-version`](../tests/acceptance/weaviate-version).

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

`copy` and `sync` create every target collection before they write the
first object. When the key may not create a collection, they stop before
they write any object. A collection that they created earlier in the same
run stays.

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

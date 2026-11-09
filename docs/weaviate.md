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
| `WEAVIATE_GRPC_URL` | no | The gRPC endpoint. Set it only for a server outside Weaviate Cloud. |

### Weaviate Cloud

Set `WEAVIATE_URL` and `WEAVIATE_API_KEY`, and leave `WEAVIATE_GRPC_URL`
unset. The tool then connects to port 443 of the `WEAVIATE_URL` host over
TLS, so it refuses a `WEAVIATE_URL` with another port.

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

`copy` reads the schema, creates collections and tenants, adds properties,
updates quantizer settings and writes objects. `verify` reads objects. A key
without the permission for one of these steps makes the command stop with
`auth_failed`, exit 3. When the key may not create a collection, `copy`
stops before it writes anything.

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
is set wins. The table lists them in that order. `check` stops with
`missing_credential`, exit 3, when the key of a planned vectorizer is
missing. A provider that refuses the key during
`copy` stops it with `auth_failed`, exit 3. The message names the provider,
and the hint names its variables.

## Async indexing

Weaviate Cloud runs with `ASYNC_INDEXING=true`. `ASYNC_INDEXING` changes two
things for a migration:

- A `dynamic` index needs it. `check` warns when the plan has a `dynamic`
  index (`needs_async_indexing`).
- A trained quantizer compresses a shard only once the shard holds
  `training_limit` vectors. See [trained quantizers](plan-file.md#trained-quantizers).

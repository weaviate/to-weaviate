# <Source name>

<!--
Copy this file to docs/sources/<name>.md for a new source and fill every
section. Keep the headings: docs/sources/README.md lists them for readers.
Add the source to the table in README.md and in docs/sources/README.md.
-->

`to-weaviate inspect <name>` reads collections from <Source name>. For each
record the tool copies <the fields it reads>. It never writes to the source.

<!-- A callout for a deadline or a deprecation of the source, if any. -->

## Deployments

| Deployment | Notes |
|---|---|
| <Managed service> | <URL default, key needed> |
| <Self-hosted server> | <version range the tool reads> |

## Connection settings

| Variable | Required | Use |
|---|---|---|
| `<NAME>_URL` | <yes/no> | <The server. Default: ...> |
| `<NAME>_API_KEY` | <yes/no> | <The key.> |

<!-- Which settings inspect records in plan.yaml, and what copy and verify
do when they change. -->

The target settings are on [Weaviate target](../weaviate.md).

## `inspect <name>` options

| Option | Effect |
|---|---|
| `--<namespace> NAME` | <The source's namespace: database, index, project. Repeatable.> |
| `--collection NAME` | <Read this collection. Repeatable.> |
| `--sample-pages N` | <Pages read to find the metadata keys and their types.> |

`inspect <name>` also takes the vector options that every source shares:
`--vectors`, `--vectorizer`, `--vectorizer-option`, `--index`, `--quantizer`,
`--quantizer-bits` and `--force`. See [Commands](../commands.md#inspect-source).

## Examples

### <Managed service>

```dotenv
<NAME>_API_KEY=<key>
WEAVIATE_URL=https://my-cluster.weaviate.cloud
WEAVIATE_API_KEY=<Weaviate API key>
```

```bash
to-weaviate inspect <name>
```

### <Self-hosted server>

## How the data maps to Weaviate

| <Source name> | Weaviate |
|---|---|
| Collection | <name rules> |
| Record id | <UUID rule; the source_id property> |
| Vector | <named vector> |
| Text | <document property> |
| Metadata | <properties; types> |
| <Fields not migrated> | <dropped, with the reason> |

### Distances

| <Source name> | Weaviate |
|---|---|
| <metric> | <distance> |

### Embedding models

<!-- Which stored embedding configurations map to a Weaviate vectorizer
module, and what inspect stores of them. -->

## Writes during a copy

<!-- Whether the application can keep writing to the source while copy and
sync run, and what the tool does when it cannot. The README and
docs/commands.md point here for it. -->

## Limits

- <Page size, rate limits, retries.>
- <What happens when the source changes during a copy.>
- <What the tool does not migrate.>

## Implementation

This section is for developers. The general design is in
[Architecture](../development/architecture.md).

### Modules

| Module | Holds |
|---|---|
| `src/to_weaviate/sources/<name>/client.py` | <The client, its errors and retries> |
| `src/to_weaviate/sources/<name>/source.py` | <The `Source` protocol for this source> |

### How it reads

<!-- Paging, the order of records, how partitions split a collection, and
any wait or consistency check before a read. -->

### Tests

- Unit tests: <modules>.
- Acceptance tests: `tests/acceptance/<name>/`, against <the test server>.

# The plan file

`inspect` writes a draft `plan.yaml` in the migration directory. The plan
says what each source collection becomes in Weaviate: the collection name,
the properties and their types, and the vector settings. You review and edit
it before `check`. `check` and `copy` refuse an invalid plan before they
write anything, and the message names the field.

`inspect --force` drafts the plan again and discards your edits.

## Example

One entry of the plan that `inspect chroma` drafted for a collection of 700
records:

```yaml
version: 1
source:
  type: chroma
  url: http://localhost:8001
  tenant: null
collections:
- source:
    ref: 451fdd2c-b3ba-4224-8117-97d6a271e002
    namespace: default_database
    name: docs-cosine
    count: 700
    dimension: 8
    distance: cosine
    embedding: null
  target:
    name: Docs_cosine
    tenant: null
    vectors:
    - name: default
      mode: keep
      vectorizer: null
      distance: cosine
      index:
        type: default
        options: {}
        quantizer:
          type: default
    properties:
    - source_key: '#document'
      name: document
      type: text
      tokenization: word
      searchable: true
      filterable: false
      range_filters: false
      coerced_from: []
    - source_key: code
      name: code
      type: text
      tokenization: field
      searchable: false
      filterable: true
      range_filters: false
      coerced_from:
      - int
    - source_key: year
      name: year
      type: int
      tokenization: null
      searchable: false
      filterable: true
      range_filters: true
      coerced_from: []
    dropped: []
  notes: []
```

The excerpt leaves out the other quantizer fields, all `null` in a draft,
and the properties `source_id`, `uri` and the other metadata keys.

## Fields

| Field | Written by | Meaning |
|---|---|---|
| `version` | `inspect` | The plan format. Always `1`. |
| `source` | `inspect` | The source type, its URL without user information, and its tenant. See [the source check](commands.md#existing-collections-and-the-recorded-target). |
| `collections[].source` | `inspect` | The source collection: its id, namespace, name, count, dimension, distance and embedding configuration. Do not edit it. |
| `collections[].target.name` | you may edit | The Weaviate collection. |
| `collections[].target.tenant` | you may edit | A tenant of the collection, or `null` for a collection without multi-tenancy. |
| `collections[].target.vectors` | you may edit | The named vectors. See [vector modes](#vector-modes) and [index and quantizer](#index-and-quantizer). |
| `collections[].target.properties` | you may edit | One entry per source field. See [properties](#properties). |
| `collections[].target.dropped` | you may edit | Source keys that are not copied, each with a reason. |
| `collections[].notes` | `inspect` | What `inspect` decided and why. |

## Edits you can make

- Rename a target collection: change `target.name`.
- Merge several source collections into one collection, one tenant each:
  give the entries the same `target.name` and a different `target.tenant`.
  The entries must have the same vectors, the same settings for each property
  that more than one of them has, and no two property names that differ only
  in case.
- Rename a property, change its type, its tokenization or its index flags.
- Drop a metadata key: remove its property and list the key under
  `target.dropped` with a reason. Each metadata key must be a property or
  listed under `dropped`; `check` refuses a key that is neither
  (`unmapped_key`).
- Change the vector mode, the vectorizer, the index type or the quantizer.

## Properties

`inspect` drafts one property per source field. Three come from the record
itself rather than from its metadata:

| `source_key` | Property | Holds |
|---|---|---|
| `#document` | `document` | The record's text |
| `#id` | `source_id` | The record's id in the source |
| `#uri` | `uri` | The record's URI |

Every other `source_key` is a metadata key. A property has these fields:

| Field | Values |
|---|---|
| `name` | The Weaviate property name. |
| `type` | `text`, `int`, `number`, `boolean`, or an array of one of them: `text[]`, `int[]`, `number[]`, `boolean[]`. |
| `tokenization` | `word` or `field`, for `text` types only. |
| `searchable` | Whether keyword search indexes the property. |
| `filterable` | Whether filters can use the property. |
| `range_filters` | Whether range filters can use the property, for `int` and `number`. |
| `coerced_from` | The other types the source holds for this key. `inspect` fills it in. |

### Types and conversion

`inspect` picks each property type from the values it read:

- A key with one type keeps it.
- A key with `int` and `number` values becomes `number`.
- A key with any other mix becomes `text`, and `coerced_from` lists the other
  types. `check` and `copy` show the warning `coerced_keys`.
- A key with both single values and lists becomes an array of the merged
  type, for example `text[]` for `text` and `text[]` values.
- A list with mixed element types becomes `text[]`.

`copy` converts each value to the property type:

| Property type | Accepts |
|---|---|
| `text` | Any scalar. A boolean becomes `true` or `false`. |
| `number` | An int or a number. |
| `int` | An int, or a number without a fractional part. |
| `boolean` | A boolean. |

Any other value is a reject with the reason "cannot convert ... without
losing data". An array type converts each element the same way.

Weaviate stores an `int` property as a double. An int above 2**53 in
magnitude is stored rounded, and `verify` compares the rounded value. An int
that rounds to 2**63 or above is a reject, because whether the server takes
it depends on its CPU. To keep every digit of such values, set the type to
`text`.

## Vector modes

`inspect --vectors` sets the mode. The plan holds one entry per named vector.

| Mode | Vectors in Weaviate | A record without a vector or a document |
|---|---|---|
| `keep` (default) | `default`: the source vectors, with the source distance. When the source's embedding model has a Weaviate module, the plan sets it as the vectorizer, so Weaviate can embed queries. | No vector: a reject, "no vector". |
| `embed` | `default`: embedded by the Weaviate vectorizer from the `document` property. The source distance is not carried over. | No document: a reject, "no document to embed". |
| `both` | `source`: the source vectors. `default`: embedded by the Weaviate vectorizer. | No document: written without the embedded vector. |

A collection without documents keeps its source vectors in every mode, and
the plan notes it.

## Index and quantizer

`index.type` is `default`, `hnsw`, `flat`, `dynamic` or `hfresh`. With
`default`, the server chooses. A quantizer, or the distance `l2-squared` or
`dot`, needs a named index type; for those distances `inspect` sets `hnsw`.

The rules below are those of the Weaviate version in
[`tests/acceptance/weaviate-version`](../tests/acceptance/weaviate-version).
`inspect`, `check` and `copy` refuse any other combination.

| Index type | `index.options` | Quantizer types | `rq` bits |
|---|---|---|---|
| `hnsw` | `ef`, `ef_construction`, `max_connections`, `dynamic_ef_min`, `dynamic_ef_max`, `dynamic_ef_factor`, `flat_search_cutoff`, `vector_cache_max_objects` | `none`, `rq`, `sq`, `pq`, `bq` | 1, 4, 8 |
| `flat` | `vector_cache_max_objects` | `none`, `rq`, `bq` | 1, 8 |
| `dynamic` | `threshold` | `none`, `rq`, `bq` | 1, 8 |
| `hfresh` | `max_posting_size_kb`, `replicas`, `search_probe` | `rq` | 1 |

The quantizer fields:

| Quantizer | Fields |
|---|---|
| `rq` | `bits`, `rescore_limit`, `training_limit`, `centering`, `cache` |
| `sq` | `rescore_limit`, `training_limit` |
| `pq` | `training_limit`, `segments`, `centroids` |
| `bq` | `rescore_limit`, `cache` |

Some combinations have extra rules:

- `rq` `centering` needs `hnsw` and 4 bits.
- `inspect`, `check` and `copy` refuse a field that the index type ignores:
  `cache` of `rq` and `bq`, and `rescore_limit` of `bq`, with `hnsw`;
  `training_limit` of `rq` with `flat`; `cache` and `training_limit` of `rq`
  with `hfresh`.
- `pq` `segments` must divide the dimension of the source vectors.

### Trained quantizers

Weaviate trains `sq`, `pq`, and `rq` with `centering` on the vectors of the
index. `copy` enables such a quantizer after the last page of each plan
entry. A server without `ASYNC_INDEXING` then compresses the vectors at once.
A server with `ASYNC_INDEXING=true` compresses a shard only once it holds
`training_limit` vectors: 100000 for `sq` and `pq`, 10000 for `rq`, unless
the plan sets another value. A smaller shard stays uncompressed, and `check`
and `copy` warn with `quantizer_training`.

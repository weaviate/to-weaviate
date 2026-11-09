# Sources

`to-weaviate` reads from one source per migration. Each source has its own
`inspect` command and its own guide. After `inspect`, the commands are the
same for every source: see [Commands](../commands.md).

| Source | Deployments | Inspect command | Acceptance tests | Guide |
|---|---|---|---|---|
| Chroma | Chroma Cloud, distributed Chroma, single-node Chroma, a Chroma directory on disk | `inspect chroma` | yes | [chroma.md](chroma.md) |

To ask for another source, [open an issue](https://github.com/weaviate/to-weaviate/issues/new).
To add one, follow [Adding a source](../development/adding-a-source.md).

## What each guide covers

Every source guide has the same sections, from
[`_template.md`](_template.md):

1. **Deployments**: which servers and services of the source the tool reads.
2. **Connection settings**: the environment variables of the source.
3. **`inspect <source>` options**: the options that select what to read.
4. **Examples**: the settings and commands for each deployment.
5. **How the data maps to Weaviate**: ids, vectors, text, metadata, distances
   and embedding models.
6. **Limits**: what the tool does not migrate, and source behavior that
   affects a copy.
7. **Implementation**: for developers. The modules, how the source reads
   pages, and how its tests run.

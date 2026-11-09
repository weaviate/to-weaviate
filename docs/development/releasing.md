# Releasing

A release is a tag `vX.Y.Z` on a commit of `main`.
`.github/workflows/release.yml` builds the wheel and the sdist and publishes
them to PyPI through trusted publishing, with no token.

## Before the tag

The release workflow runs no tests. Check two runs of the commit you will
tag, and record each result in the release notes (passed, skipped with the
reasons, or failed):

- The `CI` workflow of that commit on `main`.
- `make acceptance-cloud` against a dedicated serverless Weaviate Cloud
  cluster, from your machine. See [Testing](testing.md#against-weaviate-cloud).

## Tag

```bash
git fetch origin
git tag v0.1.0 origin/main
git push origin v0.1.0
```

The `build` job stops unless the tag points at a commit on `main` and the
built wheel has the tag's version without the `v`. When both jobs pass,
`uvx to-weaviate --version` prints `to-weaviate 0.1.0`.

## One-time setup

Someone with admin access to the GitHub repository and to PyPI sets up two
things once:

- On PyPI, a trusted publisher for the project `to-weaviate`: repository
  `weaviate/to-weaviate`, workflow `release.yml`, environment `pypi`.
- On GitHub, the environment `pypi`, which allows deployments from tags that
  match `v*`. GitHub refuses the `publish` job of any other tag.

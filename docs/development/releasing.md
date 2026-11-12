# Releasing

A release is a tag `vX.Y.Z` on a commit of `main`.
`.github/workflows/release.yml` builds the wheel and the sdist, attaches them
to a GitHub release, and publishes them to PyPI through trusted publishing,
with no token.

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

The workflow runs three jobs in order:

1. `build` stops unless the tag points at a commit on `main` and the built
   wheel has the tag's version without the `v`.
2. `github-release` creates the GitHub release for the tag, with generated
   notes and the wheel and the sdist as assets.
3. `publish` uploads the same files to PyPI. It waits until a member of
   `weaviate/core` approves the deployment to the `pypi` environment, under
   "Review deployments" on the workflow run.

PyPI keeps a version forever, so `publish` runs last: a failed release can
be deleted and the tag moved, a failed upload cannot. When the three jobs
pass, `uvx to-weaviate --version` prints `to-weaviate 0.1.0`.

## One-time setup

Someone with admin access to the GitHub repository and a PyPI account set
up two things once:

- On PyPI, a trusted publisher for the project `to-weaviate`: repository
  `weaviate/to-weaviate`, workflow `release.yml`, environment `pypi`. It was
  registered as a pending publisher, before the project existed. The first
  upload creates the project, owned by the PyPI account that registered the
  publisher; further owners are added under Manage → Collaborators. A pending
  publisher does not reserve the name.
- On GitHub, the environment `pypi`: deployments only from tags that match
  `v*`, and `weaviate/core` as required reviewers. GitHub refuses the
  `publish` job of any other ref.

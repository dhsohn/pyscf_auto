# Contributing to pyscf_auto

Thanks for helping improve pyscf_auto. This repo targets conda-only distribution and
the core CLI workflow.

## Quick start

1. Fork the repo and create a topic branch.
2. Make changes with clear commit messages.
3. Open a PR with a concise description and screenshots/logs if applicable.

## Coding guidelines

- Prefer small, focused changes.
- Keep user-facing text consistent with conda-only installation guidance.
- Avoid adding pip-specific instructions or dependencies.
- Add tests when behavior changes are not obvious.

## CI and packaging

- CI uses `pip install ".[dev]"` to install test/lint tooling only.
- Do not imply pip installs are supported for end users; production installs are conda-only.

## Testing

Run unit tests:

```bash
pytest -q
```

If you cannot run tests, note the reason in the PR.

## Reporting issues

Please use the issue templates and include:
- Steps to reproduce
- Logs or error output
- OS and conda environment details

## Release notes

Add user-facing changes to `CHANGELOG.md` under the **Unreleased** section.

# Changelog

All notable changes to pyscf_auto will be documented in this file.

Format follows Keep a Changelog, and versioning follows SemVer with a beta
stability note (APIs and workflows may change before 1.0).

## [Unreleased]

- Smoke-test reliability improvements: progress tracking, heartbeat-based watch, and status auto-recovery.
- Added `--smoke-mode` (default quick) to control smoke-test matrix size.
- Queue runner now auto-prunes entries older than 3 days on startup.
- SMD now errors when unavailable instead of falling back to PCM/vacuum.
- Added dependency install profiles (`engine`, `dispersion`, `full`) for leaner default setup.
- Added runner/execution boundary entrypoint: `execution.entrypoint.execute_attempt`.
- Added lazy plugin loading for optional stage workflows and QCSchema export.
- Added architecture guardrail tests to prevent direct stage imports from `execution/__init__.py`.

## [0.1.0] - TBD

- Initial beta release with CLI and desktop GUI.
- SMD-enabled PySCF distribution via conda channel.

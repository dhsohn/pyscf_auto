# pyscf_auto

**Status: Beta**

pyscf_auto is a local PySCF/ASE retry runner for `.inp`-based reaction directories.
The user-facing CLI is intentionally aligned with `orca_auto`:

- CLI entry point: `pyscf_auto`
- Main commands: `run-inp`, `status`, `organize`
- App config: `~/.pyscf_auto/config.yaml` (override with `PYSCF_AUTO_CONFIG`)
- Default roots: `~/pyscf_runs` (inputs/runs), `~/pyscf_outputs` (organized outputs)

## Documentation

- GitHub Pages manual (KR/EN): https://dhsohn.github.io/pyscf_auto/
- Source docs: `docs/`

## Quickstart Guide

### 1) Installation

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto
```

### 2) Run a Reaction Directory

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/my_reaction
# (repo checkout) ./bin/pyscf_auto run-inp --reaction-dir ~/pyscf_runs/my_reaction
```

### 3) Check Status

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/my_reaction
# (repo checkout) ./bin/pyscf_auto status --reaction-dir ~/pyscf_runs/my_reaction
```

### Utility Scripts (replacing legacy `doctor/validate` CLI)

```bash
# environment preflight
./scripts/preflight_check.sh

# validate one .inp file without execution
./scripts/validate_inp.py path/to/input.inp

# validate runtime config
./scripts/validate_runtime_config.py --config ~/.pyscf_auto/config.yaml
```

## Development Quality Checks

```bash
pip install -r requirements-dev.txt

pytest -q
pytest -q --cov=src --cov-report=term-missing
ruff check src tests
mypy src
```

## Scope

pyscf_auto is a local workstation tool. It is not a distributed job orchestrator,
and it does not provide remote scheduling/orchestration features.

# pyscf_auto

**Status: Beta**

pyscf_auto is a local PySCF/ASE retry runner for `.inp`-based reaction directories.
It executes calculations, applies conservative retry patches on failure, and records
state/metadata for reproducible reruns.

- CLI entry point: `pyscf_auto`
- Main commands: `run-inp`, `status`, `organize`, `doctor`, `validate`
- App config: `~/.pyscf_auto/config.yaml` (override with `PYSCF_AUTO_CONFIG`)
- Default roots: `~/pyscf_runs` (inputs/runs), `~/pyscf_outputs` (organized outputs)

## Documentation

- GitHub Pages manual (KR/EN): https://dhsohn.github.io/pyscf_auto/
- Source docs: `docs/`

## Quickstart

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

pyscf_auto run-inp --reaction-dir ~/pyscf_runs/my_reaction
pyscf_auto status --reaction-dir ~/pyscf_runs/my_reaction
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

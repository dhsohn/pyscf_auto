# pyscf_auto

**Status: Beta**

pyscf_auto is a lightweight workflow wrapper around PySCF (SCF/DFT/gradients/Hessians)
and ASE (optimization driver). It runs geometry optimization (min/TS), single-point
energy, frequency, IRC, and scans with consistent logging and reproducible metadata.

- CLI entry point: `pyscf_auto`
- Default config template: `run_config.json`
- Runs are stored under `~/pyscf_auto/runs/YYYY-MM-DD_HHMMSS/` (override with `PYSCF_AUTO_BASE_DIR`)

## Documentation

- GitHub Pages manual (KR/EN): https://dhsohn.github.io/pyscf_auto/
- Source docs: `docs/`

## Quickstart

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

pyscf_auto run path/to/input.xyz --config run_config.yaml
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

pyscf_auto is a PySCF/ASE-centered local workflow tool for workstation runs. It is
not a general-purpose distributed workflow engine; background queueing is local
to the machine where runs are submitted.

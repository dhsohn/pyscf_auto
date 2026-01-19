# DFTFlow

**Status: Beta**

DFTFlow is a lightweight workflow wrapper around PySCF (SCF/DFT/gradients/Hessians)
and ASE (optimization driver). It runs geometry optimization (min/TS), single-point
energy, frequency, IRC, and scans with consistent logging and reproducible metadata.

- CLI entry point: `dftflow`
- Default config template: `run_config.json`
- Runs are stored under `~/DFTFlow/runs/YYYY-MM-DD_HHMMSS/` (override with `DFTFLOW_BASE_DIR`)

## Documentation

- GitHub Pages manual (KR/EN): https://dhsohn.github.io/DFTFlow/
- Source docs: `docs/`

## Quickstart

```bash
conda create -n dftflow -c daehyupsohn -c conda-forge dftflow
conda activate dftflow

dftflow run path/to/input.xyz --config run_config.yaml
```

GUI:

- Desktop GUI is distributed separately (`dftflow_gui`).

## Scope

DFTFlow is a PySCF/ASE-centered local workflow tool for workstation runs. It is
not a general-purpose distributed workflow engine; background queueing is local
to the machine where runs are submitted.

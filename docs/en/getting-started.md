# Install & Quickstart

## Installation

pyscf_auto is distributed via conda.

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto
```

- `pip install pyscf_auto` is not supported.

## Environment Check

```bash
pyscf_auto doctor
```

## App Configuration

pyscf_auto reads runtime/app settings from:

1. `--config` option
2. `PYSCF_AUTO_CONFIG` environment variable
3. `~/.pyscf_auto/config.yaml` (default)

Minimal example (`~/.pyscf_auto/config.yaml`):

```yaml
runtime:
  allowed_root: ~/pyscf_runs
  organized_root: ~/pyscf_outputs
  default_max_retries: 5
```

## Prepare a Reaction Directory

Place at least one `.inp` file in a reaction directory under `runtime.allowed_root`.

```bash
mkdir -p ~/pyscf_runs/water_opt
cp input/water_opt.inp ~/pyscf_runs/water_opt/
```

## First Run

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/water_opt
```

## Check Status

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt
```

Run artifacts are written inside the reaction directory, including:

- `run_state.json`
- `run_report.json`
- `run_report.md`

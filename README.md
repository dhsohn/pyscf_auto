# DFTFlow

**Status: Beta**

DFTFlow is a lightweight workflow wrapper around PySCF (SCF/DFT/gradients/Hessians) and ASE (optimization driver). It runs geometry optimization (min/TS), single-point energy, frequency, IRC, and scans with consistent logging and reproducible metadata.

- CLI entry point: `dftflow` (implementation in `run_opt.py`, core logic under `src/`)
- Default config template: `run_config.json`
- Outputs are organized under `~/DFTFlow/runs/YYYY-MM-DD_HHMMSS/`

## Highlights

- **Reproducible runs**: config, environment, and git metadata captured per run.
- **Desktop GUI + CLI backend**: GUI for submit/monitor/view, CLI for automation.
- **Chained workflows**: optimization → frequency → single-point in one execution.
- **Solvent + dispersion support**: PCM/SMD, D3/D4 with guardrails.
- **Queue and status tooling**: background runs and quick status views.

## Capabilities

### Geometry optimization
- Minimum or transition-state (TS) optimization.
- Optimizers: ASE (BFGS, etc.) and Sella for TS (order=1).
- Optional follow-up: frequency, SP, IRC.

### Single-point energy
- Compute energy at the current geometry (standalone or after optimization).

### Frequency analysis
- Hessian via PySCF; harmonic analysis.
- Default: **no dispersion in Hessian** (frequency dispersion mode = `none`).

### IRC
- IRC path from a TS; forward/reverse trajectories with energy profile.

### Scans (1D/2D)
- Scan bond/angle/dihedral grids with optimization or single-point at each point.

## Solvent models

- `vacuum` (default): no solvent.
- `pcm`: dielectric epsilon from `solvent_dielectric.json`.
- `smd`: requires PySCF built with SMD support (see SMD packaging below).

If SMD is not available, DFTFlow falls back to PCM or vacuum depending on context.

## Dispersion

- Supported: `d3bj`, `d3zero`, `d4`.
- Duplicated dispersion is avoided when the XC already embeds it.
- Some XC + D3 parameter combos are unsupported by dftd3; smoke tests skip them.

## Smoke test

`dftflow smoke-test` runs a broad matrix of quick checks (1 SCF cycle and 1-step optimizations).

Default behavior:
- Each case runs in a **separate subprocess** (isolation to avoid cascading crashes).
- Capability check is **skipped** during smoke tests only.
- Unsupported D3/XC combos are marked **skipped** instead of failing.

Useful flags:
- `--resume`: continue in an existing run directory.
- `--stop-on-error`: stop immediately on the first failure.
- `--watch`: monitor and auto-resume when logs stall.
- `--watch-timeout <sec>`: inactivity timeout before restart.
- `--watch-interval <sec>`: polling interval.
- `--watch-max-restarts <n>`: limit restarts (0 = unlimited).
- `--no-isolate`: run all cases in the same process (not recommended).

Smoke-test artifacts (per case):
- `run.log`, `log/run_events.jsonl`
- `smoke_subprocess.out` / `smoke_subprocess.err`
- `smoke_subprocess.status`

## Output layout

Per run directory (example):

```
run.log
log/run_events.jsonl
metadata.json
config_used.json
optimized.xyz
frequency_result.json
irc_result.json
scan_result.json
```

## Installation (SMD-enabled build)

Python: **3.12**

Install DFTFlow from the SMD-enabled conda channel:

```bash
conda create -n dftflow -c daehyupsohn -c conda-forge dftflow
```

This installation includes the SMD-enabled PySCF build required for solvent modeling.
Keep `daehyupsohn` first so the SMD-enabled PySCF build is preferred.

Launch the desktop app:

```bash
dftflow-gui
```

## Usage

### Desktop GUI (default)

```bash
dftflow-gui
```

### CLI run

```bash
dftflow run path/to/input.xyz --config run_config.json
```

### Resume a run

```bash
dftflow run --resume ~/DFTFlow/runs/2026-01-03_100104/0147_optimization_...
```

### Status and diagnostics

```bash
dftflow status runs/2026-01-03_100104/0147_optimization_...
dftflow status --recent 5
dftflow doctor
dftflow validate-config run_config.json
```

### Queue

```bash
dftflow queue status
dftflow queue cancel <RUN_ID>
dftflow queue retry <RUN_ID>
dftflow queue requeue-failed
dftflow queue prune --keep-days 30
```

## Configuration notes

- Set charge/multiplicity in XYZ comment line:
  - Example: `charge=0 multiplicity=1`
- If omitted, multiplicity is inferred from electron parity.
- `solvent_dielectric.json` provides PCM epsilon map.
- `frequency_dispersion_mode` defaults to `none`.

## Repository structure

```
run_opt.py
src/
  run_opt.py
  run_opt_engine.py
  run_opt_dispersion.py
  run_opt_config.py
  run_opt_logging.py
  run_opt_metadata.py
  run_opt_paths.py
  run_opt_resources.py
  gui_app.py
packaging/
  pyscf-smd/
run_config.json
solvent_dielectric.json
~/DFTFlow/runs/
```

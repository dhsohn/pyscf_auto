# DFTFlow

**Status: Beta**

DFTFlow is a lightweight workflow wrapper around PySCF (SCF/DFT/gradients/Hessians) and ASE (optimization driver). It runs geometry optimization (min/TS), single-point energy, frequency, IRC, and scans with consistent logging and reproducible metadata.

- CLI entry point: `dftflow` (implementation in `run_opt.py`, core logic under `src/`)
- Default config template: `run_config.json`
- Outputs are organized under `~/DFTFlow/runs/YYYY-MM-DD_HHMMSS/`

## Scope

DFTFlow is a PySCF/ASE-centered local workflow tool for workstation runs. It is
not a general-purpose distributed workflow engine; background queueing is local
to the machine where runs are submitted.

## Highlights

- **Reproducible runs**: config, environment, and git metadata captured per run.
- **Desktop GUI (separate app)**: install `dftflow_gui` for submit/monitor/view.
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

If SMD is not available, DFTFlow stops with a clear error message.

## Dispersion

- Supported: `d3bj`, `d3zero`, `d4`.
- Duplicated dispersion is avoided when the XC already embeds it.
- Some XC + D3 parameter combos are unsupported by dftd3; smoke tests skip them.

## Smoke test

`dftflow smoke-test` runs a matrix of quick checks (1 SCF cycle and 1-step optimizations).

Default behavior:
- Each case runs in a **separate subprocess** (isolation to avoid cascading crashes).
- Capability check is **skipped** during smoke tests only.
- Unsupported D3/XC combos are marked **skipped** instead of failing.
- Defaults to `--smoke-mode quick` (representative subset). Use `--smoke-mode full` for the full matrix.
- Progress is tracked in `smoke_progress.json` for reliable resume.
- Heartbeat updates are written to `smoke_heartbeat.txt` for stall detection.

Useful flags:
- `--resume`: continue in an existing run directory.
- `--stop-on-error`: stop immediately on the first failure.
- `--watch`: monitor and auto-resume when logs stall.
- `--watch-timeout <sec>`: inactivity timeout before restart.
- `--watch-interval <sec>`: polling interval.
- `--watch-max-restarts <n>`: limit restarts (0 = unlimited).
- `--no-isolate`: run all cases in the same process (not recommended).
- `--smoke-mode <quick|full>`: choose the test matrix size.

Smoke-test artifacts (per case):
- `run.log`, `log/run_events.jsonl`
- `smoke_subprocess.out` / `smoke_subprocess.err`
- `smoke_subprocess.status`
Smoke-test summary:
- `smoke_progress.json`

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
conda activate dftflow
```

Note: DFTFlow is distributed via conda only. `pip install dftflow` (or
`pip install .`) is unsupported and will not provide the SMD-enabled PySCF build.

This installation includes the SMD-enabled PySCF build required for solvent modeling.
Keep `daehyupsohn` first so the SMD-enabled PySCF build is preferred.

Desktop GUI is distributed separately (see the `dftflow_gui` repository).

## Usage

### Desktop GUI

Install and launch the separate `dftflow_gui` app.

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

## Reproducibility and operations

### Reproducibility checklist

- Keep your `run_config.json` under version control; run `dftflow validate-config` before production runs.
- Each run writes `config_used.json` and `metadata.json` in the run directory for exact inputs and run context.
- Use `--run-dir` and `--run-id` to make runs traceable in notebooks, tickets, or LIMS.
- Capture the conda environment (`conda list --explicit` or `conda env export --no-builds`) and store it alongside the run.
- To reproduce results elsewhere, copy the full run directory (see Output layout above).

### Operations and monitoring

- Use `--background` to enqueue runs; the queue runner auto-starts on first use.
- `dftflow queue status` shows queued and running jobs; `dftflow status --recent N` is a quick health check.
- Use `--queue-max-runtime` to cap wall time and `dftflow queue retry` to re-run failed entries.
- Prefer `--resume` for restarts so metadata and logs stay linked to the original run.
- Primary logs: `run.log`, `log/run_events.jsonl`, and `metadata.json` inside each run directory.

## Troubleshooting / FAQ

### SMD missing or unavailable

- Error: "SMD is unavailable in this PySCF build."
- Fix: Install from the SMD-enabled conda channel and keep `daehyupsohn` first.
- Check: `python -c "import pyscf; from pyscf.solvent import smd; print(smd.libsolvent is not None)"`

### Queue stalls or jobs never start

- Check the queue runner log: `~/DFTFlow/log/queue_runner.log` (or `$DFTFLOW_BASE_DIR/log/queue_runner.log`).
- Use `dftflow queue status` and `dftflow status --recent 5` to see if runs are stuck.
- Retry failed entries with `dftflow queue retry <RUN_ID>` or requeue with `dftflow queue requeue-failed`.
- If a run is stuck in "running" with no log updates, check `run.log` and `log/run_events.jsonl`.

### Resume errors

- If the run is marked completed/failed/timeout/canceled, use `--force-resume`.
- Ensure you pass the run directory (must include `checkpoint.json`; `config_used.json` is preferred).
- If `--resume` fails to load config, check `config_used.json` for a valid JSON payload.

### SCF non-convergence

- DFTFlow retries SCF with level shift/damping when convergence fails.
- Set `DFTFLOW_SCF_RETRY=0` to disable automatic retries.

## Configuration notes

- Set charge/multiplicity in XYZ comment line:
  - Example: `charge=0 multiplicity=1`
- If omitted, multiplicity is inferred from electron parity.
- `solvent_dielectric.json` provides PCM epsilon map.
- `frequency_dispersion_mode` defaults to `none`.
- SCF checkpoints default to `scf.chk` in the run directory for faster restarts.

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
packaging/
  pyscf-smd/
run_config.json
solvent_dielectric.json
~/DFTFlow/runs/
```

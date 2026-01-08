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
- Default: **numerical dispersion Hessian** (`frequency.dispersion: numerical`) so the
  vibrational analysis aligns with the dispersion-corrected PES. Use
  `frequency.dispersion: energy` to keep dispersion energy only, or
  `frequency.dispersion: none` to skip dispersion entirely.
- Thermochemistry (G/H) uses the dispersion-corrected energy when dispersion is enabled
  (`numerical`/`energy`); set `frequency.dispersion: none` to exclude dispersion.
- For PCM/SMD, Gibbs free energies include a default 1M standard-state correction
  (see `thermochemistry.standard_state_correction` in the frequency output).

### IRC
- IRC path from a TS; forward/reverse trajectories with energy profile.
- Writes `irc_profile.csv` with per-step energies and direction assessment.

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

## Test Report (2026-01-08)

Environment:
- OS: macOS 26.2 (arm64)
- Python: 3.12.12 (conda-forge)
- Conda env: dftflow-test
- Packages: dftflow 0.1.0, pyscf 2.11.0, ase 3.27.0, dftd3 1.2.1, dftd4 3.7.0

Tests:
- `pytest -q` (PYTHONPATH=src): 32 passed
- `dftflow smoke-test --smoke-mode quick`: 160/160 completed, 0 failed, 0 skipped
  - Run dir: `runs/smoke/2026-01-08_110458`

Notes:
- Smoke tests skip capability checks by design.
- Only osx-arm64 is validated so far; other platforms are untested.

## Output layout

Per run directory (example):

```
run.log
log/run_events.jsonl
metadata.json
config_used.json
qcschema_result.json
optimized.xyz
frequency_result.json
irc_result.json
irc_profile.csv
scan_result.json
scan_result.csv
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

Testing note: only `osx-arm64` has been validated so far. Packages for `linux-64`,
`osx-64`, and `win-64` are provided but not yet tested.

## Usage

### Desktop GUI

Install and launch the separate `dftflow_gui` app.

### CLI run

```bash
dftflow run path/to/input.xyz --config run_config.yaml
```

Optional profiling (writes timing + SCF cycle counts to metadata):

```bash
dftflow run path/to/input.xyz --config run_config.yaml --profile
DFTFLOW_PROFILE=1 dftflow run path/to/input.xyz --config run_config.yaml
```

### Config examples (YAML)

YAML is the recommended format in this guide; JSON is also supported (the bundled
`run_config.json` is a full template).

Save a config with a `.yaml` extension and pass it with `--config`:

```bash
dftflow run path/to/input.xyz --config config.yaml
```

#### Example 1: Single-point energy (single experiment, YAML)

```yaml
basis: def2-svp
xc: b3lyp
solvent: vacuum
calculation_mode: single_point
```

#### Example 2: Optimization -> frequency -> single-point (3-step workflow, YAML)

```yaml
basis: def2-svp
xc: b3lyp
solvent: water
solvent_model: pcm
dispersion: d3bj
calculation_mode: optimization
frequency_enabled: true

single_point:
  basis: def2-tzvp
  xc: b3lyp
  dispersion: d3bj
```

#### Example 3: TS optimization -> frequency -> IRC (3-step workflow, YAML)

```yaml
basis: def2-svp
xc: b3lyp
solvent: vacuum
calculation_mode: optimization
frequency_enabled: true
single_point_enabled: false
irc_enabled: true

optimizer:
  mode: transition_state

irc:
  steps: 20
  step_size: 0.05
  force_threshold: 0.02
```

#### Example 4: Frequency-only + thermochemistry (single experiment, YAML)

```yaml
basis: def2-svp
xc: b3lyp
solvent: vacuum
calculation_mode: frequency

thermo:
  T: 298.15
  P: 1.0
  unit: atm
```

#### Example 5: 1D relaxed bond scan (single experiment, YAML)

```yaml
basis: def2-svp
xc: b3lyp
solvent: vacuum
calculation_mode: scan

scan:
  mode: optimization
  dimensions:
    - type: bond
      i: 0
      j: 1
      start: 1.0
      end: 2.0
      step: 0.1
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
dftflow validate-config run_config.yaml
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

- Keep your config file (e.g., `run_config.yaml`) under version control; run `dftflow validate-config` before production runs.
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
- Enable profiling with `--profile` or `DFTFLOW_PROFILE=1` to record SCF/gradient/Hessian timings
  and cycles in `metadata.json` (per-stage payloads or `optimization.profiling` for optimization runs).

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

- Set charge/spin in XYZ comment line:
  - Example: `charge=0 spin=0`
- `spin_mode: auto` infers spin from electron parity when spin is omitted.
- `solvent_dielectric.json` provides PCM epsilon map.
- `frequency.dispersion` defaults to `numerical` (dispersion energy + finite-difference Hessian).
  Use `frequency.dispersion: energy` for energy-only, or `frequency.dispersion: none`
  to disable dispersion for frequencies.
- `frequency.dispersion_model` overrides the dispersion model for frequencies
  (defaults to the active dispersion setting for the stage).
- `frequency.dispersion_step` (Angstrom) controls the numerical step size (default 0.005);
  it applies only when `frequency.dispersion: numerical`.
- `frequency.use_chkfile` defaults to true; set it to false to disable SCF chkfile reuse
  during frequency calculations.
- `scf.retry_preset` controls SCF retry aggressiveness: `fast`, `default`, `stable`, `off`.
- `scf.diis_preset` sets a DIIS space preset when `scf.diis` is unset: `fast`, `default`,
  `stable`, `off`.
- `scf.reference` chooses the SCF reference: `auto`, `rks`, or `uks` (default: `auto`).
- For large systems, DFTFlow logs a recommendation to enable
  `scf.extra.density_fit: autoaux` for faster SCF/gradient/Hessian.
- SCF checkpoints default to `scf.chk` in the run directory for faster restarts.
- SCF checkpoints are reused across optimization → frequency/single-point and scan points (set `chkfile: null` to disable).
- `scan.executor` defaults to `local` (parallel). Use `serial` for single-process or `manifest`
  to generate a distributed scan manifest.
- `scan.max_workers` and `scan.threads_per_worker` are auto-adjusted for local scans to avoid
  CPU oversubscription (workers × threads_per_worker ≤ CPU count).
- `scan.batch_size` groups points per worker in local scans to reduce process overhead.
- `scan_result_csv_file` sets the scan CSV output path (default: `scan_result.csv`).
- `io.write_interval_steps` / `io.write_interval_seconds` control how often optimization
  metadata/checkpoints are written (defaults: 5 steps or 5 seconds).
- `io.scan_write_interval_points` controls how often scan results/metadata are written
  (default: every point).
- `irc_profile_csv_file` sets the IRC profile CSV output path (default: `irc_profile.csv`).
- `qcschema_output_file` sets the QCSchema output path (default: `qcschema_result.json`).
- `spin_mode` controls spin handling: `auto` uses parity when spin is missing,
  `strict` requires spin in the XYZ comment line (default).
- `ts_quality` tunes TS quality checks (imaginary count/range and optional mode projection);
  set `ts_quality.enforce: true` to skip IRC/single-point when checks fail (default: false).

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

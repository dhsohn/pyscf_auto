# DFTFlow

A lightweight workflow script that combines PySCF (DFT/SCF/gradient/Hessian) with ASE (optimization driver) to run **geometry optimization (minima/transition states), single-point (SP) energy, and frequency analysis** in one go.

- Entry point: `dftflow` CLI (implementation in `run_opt.py`, modules in the `src/` directory)
- Default config template: `run_config.json`
- Outputs are organized under `runs/YYYY-MM-DD_HHMMSS/` per execution.

## Program Overview

This project bundles the flow of **input structure (XYZ) → PySCF calculation → ASE/Sella optimization → result/metadata organization** into a single CLI, providing consistent handling for minimum/TS optimization plus optional SP/frequency calculations. Each stage is handled by a dedicated internal module; users only need to choose the calculation mode, provide inputs/settings, run, and inspect results.

Key design goals:
- **Reproducibility**: used settings/versions/environment info are automatically saved.
- **Automation**: optional chaining of frequency + single-point calculations after optimization.
- **Experimental extensibility**: scans/IRC/thermochemistry can continue in the same execution context.

### Execution flow (summary)
1. **Input/config loading**: `src/run_opt.py` parses CLI args and `src/run_opt_config.py` loads/validates JSON.
2. **Structure/charge/spin preparation**: reads XYZ + metadata; estimates `charge`/`multiplicity` if absent.
3. **Chemistry engine setup**: `run_opt_engine.py` applies PySCF settings (basis, functional, SCF options, solvent model, etc.).
4. **Optimizer selection/run**:
   - Minima: ASE (BFGS, etc.) + PySCF gradients
   - TS: Sella-based first-order saddle search (`order=1`)
5. **Post-processing**: optional frequency → validation (imaginary modes) → SP. For TS, imaginary mode validation and IRC can follow.
6. **Result logging**: `run_opt_logging.py` and `run_opt_metadata.py` organize logs, events, config snapshots, and results into a standard directory structure.

### Core design points
- **Single entry point**: `dftflow` provides `run/doctor/validate-config/status/queue` subcommands, supporting interactive, non-interactive, and queued runs.
- **Separated configuration**: JSON templates make settings reproducible. Used settings are stored as `config_used.json`.
- **Modular dispersion/solvent**: D3/D4 and PCM/SMD are optional; if the XC already includes dispersion, duplicate settings are ignored.
- **Diagnostics/utilities**: `dftflow doctor`, `dftflow status`, `dftflow queue` for fast environment and run checks.

### Output layout (per run)
- **Execution logs**: `run.log`, `log/run_events.jsonl`
- **Results**: `optimized.xyz`, `frequency_result.json`, etc.
- **Reproducibility info**: `metadata.json`, `config_used.json`

## QCSchema Unit/Provenance Mapping

- **Units**
  - QCSchema `return_result`, `properties.return_energy`, `properties.scf_total_energy` are **Hartree**.
  - IRC profile energies (`energy_ev`) are converted to **Hartree** in QCSchema.
  - Coordinates use Angstrom for input/output.
- **Property mapping**
  - `return_result` follows SP → frequency → metadata summary (`summary.final_energy`).
  - `properties.return_energy`, `properties.scf_total_energy` mirror `return_result`.
  - `properties.gradient` is **Hartree/Bohr** when present.
  - `properties.units` records energy/gradient units.
- **Model info**
  - `model.method`/`model.basis` use final SP/frequency settings if available.
  - `model.solvent`/`model.solvent_model`/`model.solvent_eps` mirror solvent settings.
- **Provenance**
  - `creator` is `DFTFlow`, `version` is installed `dftflow` package version.
  - `routine` records calculation mode, Python version, git commit.
  - `walltime` uses `summary.elapsed_seconds`, `hostname` uses execution host name.

## Features

### 1) Calculation capabilities
- **Geometry optimization (minimum)**: ASE optimizer (BFGS, etc.) + PySCF DFT gradients; convergence via `fmax`, `steps`.
- **Transition state optimization**: Sella `order=1` first-order saddle search; imaginary mode validation + IRC follow-ups.
- **Single-point energy**: energy at optimized structure; can run automatically after optimization.
- **Frequency analysis**: PySCF Hessian → harmonic analysis.  
  - Default: **no dispersion in Hessian**, only optional energy correction (`"frequency_dispersion_mode": "none"`).
- **IRC calculation**: tracks reaction coordinate from TS imaginary mode (ASE-based).  
  - Saves `irc_result.json` with forward/reverse paths and energy profile.
- **Thermochemistry**: ZPE/enthalpy/entropy/free energy from frequency results (optional).  
  - `thermo` settings (T/P/unit) are recorded in `frequency_result.json` and `metadata.json`.

### 2) Solvent models
- `vacuum` (default): no solvent.
- `pcm`: requires dielectric ε from `solvent_dielectric.json`.
- `smd`: PySCF must be built with SMD (`ENABLE_SMD=ON`).

### 3) Dispersion correction
- Supports `d3bj`, `d3zero`, `d4`. If XC already includes dispersion (`...-d`, `...d3`, etc.), extra settings are ignored.

### 4) Scan calculations (1D/2D)
Scan a bond/angle/dihedral grid with optimization or single-point at each point.

- **Dimensions**: 1D/2D
- **Types**: `bond`, `angle`, `dihedral` (indices)
- **Modes**: `optimization` or `single_point`
- **CLI precedence**: CLI scan options override config `scan` keys.
- **Grid**:
  - Range: `start,end,step`
  - Explicit list: `--scan-grid`
- **Results**: per-point outputs under run directory; `scan_result.json/csv` summarise energies/convergence/structures.

Example: 1D bond scan (0-1 bond length)

Example: 2D scan + custom grid

### 5) XYZ charge/multiplicity handling
- Put metadata on the **2nd line (comment)** of `.xyz`:
  - Example: `charge=0 multiplicity=1`
- If omitted, spin is estimated from electron parity. For radicals/TS/metals/diradicals, provide `multiplicity` explicitly to avoid wrong spin states.

### 6) Background queue & status/diagnostics
- `--background` runs via queue (priority/timeout supported).
- `dftflow queue status/cancel/retry/archive/prune` manages queue.
- `dftflow status <run_dir>` / `--recent` gives summary.
- `dftflow doctor`, `dftflow validate-config` check environment/config.

## Directory structure (summary)

```
.
├─ run_opt.py                 # main CLI/workflow wrapper
├─ src/
│  ├─ run_opt.py               # main CLI/workflow
│  ├─ run_opt_engine.py        # PySCF SP/frequency/solvent logic
│  ├─ run_opt_dispersion.py    # D3/D4 parsing & backend mapping
│  ├─ run_opt_config.py        # config load/validation
│  ├─ run_opt_logging.py       # logging/event logging
│  ├─ run_opt_metadata.py      # metadata/result aggregation
│  ├─ run_opt_resources.py     # thread/memory controls
├─ run_config.json             # default optimization template
├─ solvent_dielectric.json     # PCM dielectric map
├─ runs/                       # run outputs
```

## Installation

Supported Python version: **3.12**

### Recommended: Conda environment
Default path is "install scientific stack with conda + build PySCF from source".
```shell
git clone https://github.com/dhsohn/DFTFlow.git
```

#### 1) Create base environment with `environment.yml`
- Includes **Python 3.12**, toolchain/libs for PySCF build, ASE, D3/D4 dependencies.
- TS optimization (Sella) is **required**; install `sella`.
```shell
conda env create -f environment.yml
```
```shell
conda activate DFTFlow
```

#### 2) Use lock file (recommended for reproducibility)
Use `conda-lock` for platform-specific locks.
```shell
conda install -c conda-forge conda-lock -y
```
```shell
conda-lock lock -f environment.yml -p win-64
conda-lock lock -f environment.yml -p osx-arm64
conda-lock lock -f environment.yml -p linux-64
```
```shell
conda-lock install --name dftflow conda-lock.yml
```
```shell
pip install -e .
```
```shell
pip install sella basis-set-exchange
```

## How to run

### 1) Interactive mode (default, recommended)
Run from repo root:

```
dftflow run
```

Flow:
1. Choose calculation type
   - Geometry optimization / Single-point / Frequency
2. If optimization: choose intermediate vs transition state (TS)
3. Select XYZ path, basis, XC, solvent model
4. Optional: run frequency after optimization
5. Optional: if frequencies are acceptable (e.g., one imaginary for TS), run SP

Interactive mode stores `config_used.json` in `runs/...`.

### 2) Non-interactive mode (batch)
Use `--non-interactive` with JSON config + XYZ input.

Example: use minimum optimization template as-is

Example: TS optimization template

For TS optimization, adjust `run_config.json` for `transition_state`.

Example: IRC calculation mode (non-interactive)

### 2-1) Scan calculation (non-interactive only)
Scan is CLI-only and **not available in interactive mode**.

### 3) Resume
Use `--resume` to continue from an existing `runs/...` directory. It reads `checkpoint.json` and `config_used.json`.

- `--resume` cannot be used with `--run-dir`.
- Completed/failed/timeout/canceled runs require `--force-resume`.

### Useful options/commands
- `--run-dir <dir>`: set output directory
- `--run-id <uuid>`: fix run id
- `--solvent-map <json>`: path to solvent dielectric map
- `dftflow validate-config [config.json]`: validate config only
- `dftflow status <run_dir|metadata.json>`: run status summary
- `dftflow status --recent <N>`: recent run summaries
- `dftflow doctor`: environment diagnostics
- `--scan-dimension`, `--scan-grid`, `--scan-mode`: scan-only options

## Background queue execution/management

### 1) Enqueue a run

Options:
- `--queue-priority <int>`: higher runs sooner
- `--queue-max-runtime <sec>`: max runtime (seconds)

### 2) Queue status/manage
Queue file is `runs/queue.json`, queue runner log is `log/queue_runner.log`.

## Utility commands

### Environment diagnostics

### Config validation (short alias supported)
Legacy flags/aliases (`--doctor`, `--validate-only`, `--status*`, `--queue-*`) remain supported; subcommands are recommended.

## Built-in debugging/tests

### 1) Config validation smoke test
Goal: quick check that the default template loads and passes schema validation.

Run:

```
dftflow --validate-config
```

Required files:
- Config file: `run_config.json`

Output:
- Validation result printed to stdout (no `runs/` created).

Success criteria:
- `Config validation passed: <config>` and exit code 0.

### 2) Water single-cycle smoke test
Goal: run a tiny calculation with one SCF cycle to sanity-check the runtime stack.

Run smoke tests (all modes + all combinations):

```
dftflow smoke-test
```

Output:
- Smoke-test run directory created under `runs/` (each case writes a subdirectory).

Success criteria:
- `Smoke test completed: <run_dir> (<N> cases)` and exit code 0.

### 3) pytest unit tests
Goal: regression coverage for config parser and dispersion logic.

Run:

```
pytest
```

Required files:
- Template used by tests: `run_config.json`

Output:
- pytest results to stdout.

Success criteria:
- All `tests/...` are `passed`.

## Outputs (Results)

Each run creates a folder under `runs/YYYY-MM-DD_HHMMSS/` that typically includes:

- `run.log`: full log
- `log/run_events.jsonl`: event log (JSONL)
- `metadata.json`: execution metadata (env/version/time/status)
- `config_used.json`: config snapshot
- `optimized.xyz` / `<output_xyz>`: optimized structure
- `ase_opt.traj` or `ts_opt.traj`: ASE trajectory
- `frequency_result.json`: frequency output (if run)
- `irc_result.json`: IRC output (if run)
- `qcschema_result.json`: QCSchema AtomicResult summary
  - `schema_name`, `schema_version`
  - `molecule`: input structure (elements/coords/charge/multiplicity)
  - `return_result`: final energy (Hartree)
  - `extras.dftflow`: calculation metadata and stage snapshots
- `irc_profile.csv`: IRC energy profile
  - `direction`: forward/reverse
  - `step`: step index
  - `energy_ev`: energy (eV)
  - `energy_hartree`: energy (Hartree)
- `irc_forward.xyz`, `irc_reverse.xyz`: IRC path structures
- `scan_result.json`: scan results (scan mode)
- `scan_result.csv`: scan results CSV
  - `index`: scan point index
  - `values.*`: scan dimension values (e.g., `bond_0_1`)
  - `energy`: single-point energy
  - `converged`: convergence status
  - `cycles`: SCF cycles
  - `optimizer_steps`: optimization steps
  - `input_xyz`: input structure path
  - `output_xyz`: optimized structure path

## Config file (JSON) key fields

### Common
- `threads`, `memory_gb`: compute resources
- `basis`, `xc`: basis/functional
- `dispersion`: `"d3bj"`, `"d3zero"`, `"d4"` or `null`
- `solvent`, `solvent_model`, `solvent_map`: solvent settings
- `scf`: PySCF SCF settings
  - `extra`: detailed options
    - `grids.level`, `grids.prune`: DFT grid settings
    - `density_fit`: `true|false` (only for supported SCF types)
    - `init_guess`: PySCF `init_guess` string (e.g., `"atom"`, `"minao"`)

Example: `scf.extra`

### Geometry optimization
- `optimizer.output_xyz`: final structure output filename
- `optimizer.mode`: `"minimum"` or `"transition_state"`
- `optimizer.ase.optimizer`: `"bfgs"` or `"sella"`
- `optimizer.ase.fmax`, `optimizer.ase.steps`: convergence and max steps
- `optimizer.ase.trajectory`, `optimizer.ase.logfile`: log/trajectory filenames

### Dispersion (D3)
- (Optional) `optimizer.ase.d3_params.damping`: `s6, s8, a1, a2`, etc.

### IRC
- `irc_enabled`: `true|false` (force IRC after optimization)
- `irc_file`: IRC output path (default: `irc_result.json`)
- `qcschema_output_file`: QCSchema output path (default: `qcschema_result.json`)
- `irc_profile_csv_file`: IRC profile CSV path (default: `irc_profile.csv`)
- `irc.steps`, `irc.step_size`, `irc.force_threshold`: IRC settings

### SCF `extra` policies/constraints
- `density_fit` support:
  - Works only for mean-field objects that provide `density_fit()`.
  - Typically supported in DFT (RKS/UKS/ROKS). Unsupported SCF types error.
- `init_guess` vs `chkfile` priority:
  1. If `scf.chkfile` exists and `scf/dm` is stored, the chkfile density matrix is used and `init_guess` is ignored.
  2. If `chkfile` exists but `scf/dm` does not, and no `init_guess` is set, `init_guess` defaults to `"chkfile"`.
  3. If no `chkfile` or empty path, `init_guess` is used as provided.

### Thermochemistry
- `thermo.T`, `thermo.P`, `thermo.unit`: temperature/pressure settings (`"atm"`, `"bar"`, `"Pa"`, etc.)

## Troubleshooting

### 1) `KeyError: 'bj'` (D3 damping)
- Cause: passing `damping="bj"` to `dftd3-python` backend
- Fix: `dftd3-python` requires `damping="d3bj"`; the config parser should map accordingly.

### 2) JSON parse errors (e.g., "two JSON objects stuck together")
- Cause: invalid JSON (multiple objects, extra text, or comments).
- Fix: ensure a single valid JSON object with no comments.

### 3) PySCF import/namespace conflicts
- Cause: environment confusion, wrong `pyscf` installed/imported.
- Fix: reinstall PySCF, clean environment.

If output is strange or `None`, PySCF may need reinstall/cleanup.

### 4) `ModuleNotFoundError: No module named 'pytest'`
- Cause: dev/test dependencies not installed.
- Fix: install dev requirements.

### 6) `FileNotFoundError: run_config.json`
- Cause: running outside repo root.
- Fix: `cd DFTFlow` or pass absolute/relative path correctly.

## References/Credits

- PySCF: electronic structure engine (DFT/SCF/gradient/Hessian)
- ASE: optimization/Atoms model
- Sella: transition state optimization
- simple-dftd3 (dftd3-python): D3(BJ) dispersion correction

# pyscf_auto

**Status: Beta**

Local PySCF/ASE retry runner for `.inp`-based reaction directories.
The user-facing command surface is aligned with `orca_auto`.

- CLI entry point: `pyscf_auto`
- Main commands: `run-inp`, `status`, `organize`, `cleanup`
- App config: `~/.pyscf_auto/config.yaml` (or `PYSCF_AUTO_CONFIG`)
- Default roots: `~/pyscf_runs` (input/runs), `~/pyscf_outputs` (organized outputs)

## Quickstart

### 1) Install

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto
```

### Dependency Profiles

- `core` (default): CLI + state/organize/cleanup paths
- `engine`: adds PySCF/ASE runtime dependencies for `run-inp`
- `dispersion`: optional D3/D4 backends
- `full`: `engine + dispersion`

Example (pip editable/dev environment):

```bash
pip install -e ".[engine]"
pip install -e ".[full]"
```

### Optional Feature Flags

Heavy workflow stages can be disabled explicitly:

- `PYSCF_AUTO_DISABLE_SCAN=1`
- `PYSCF_AUTO_DISABLE_IRC=1`
- `PYSCF_AUTO_DISABLE_FREQUENCY=1`
- `PYSCF_AUTO_DISABLE_QCSCHEMA=1`

### 2) Run

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/my_reaction
```

### 3) Check status

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/my_reaction
```

### 4) Organize outputs (optional)

```bash
# dry-run
pyscf_auto organize --root ~/pyscf_runs

# apply
pyscf_auto organize --root ~/pyscf_runs --apply
```

### 5) Cleanup organized outputs (optional)

```bash
# dry-run
pyscf_auto cleanup --root ~/pyscf_outputs

# apply
pyscf_auto cleanup --root ~/pyscf_outputs --apply

# single organized run
pyscf_auto cleanup --reaction-dir ~/pyscf_outputs/single_point/H2O/run_001 --apply
```

Default keep/remove policy is configurable in `cleanup` section of config.

## Utility Scripts

```bash
./scripts/preflight_check.sh
./scripts/validate_inp.py path/to/input.inp
./scripts/validate_runtime_config.py --config ~/.pyscf_auto/config.yaml
```

## Development Checks

```bash
pip install -r requirements-dev.txt
pytest -q
pytest -q --cov=src --cov-report=term-missing
ruff check src tests
mypy src
```

## Scope

pyscf_auto is a local workstation tool. It does not provide distributed scheduling/orchestration.

## Architecture Guardrails

- Runner/execution boundary is fixed at `execution.entrypoint.execute_attempt`.
- Stage implementations are lazy-loaded via `execution.plugins`.
- Guardrail tests enforce this boundary and block direct stage imports from `execution/__init__.py`.

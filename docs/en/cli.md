# CLI Reference

## run-inp

Run a calculation from the newest `.inp` file in a reaction directory.

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/water_opt
```

Key options:

- `--max-retries N`: override retry count
- `--force`: rerun even if already completed
- `--json`: print JSON summary

Global options (placed before the command):

- `--verbose` / `-v`: debug logging
- `--config PATH`: app config path

Example:

```bash
pyscf_auto --config ~/.pyscf_auto/config.yaml -v run-inp --reaction-dir ~/pyscf_runs/water_opt
```

## status

Show run status for a reaction directory.

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt --json
```

## organize

Preview or apply organization of completed runs.

For `organize`, you must provide exactly one of:

- `--reaction-dir DIR`
- `--root ROOT` (`ROOT` must exactly match `allowed_root`)

```bash
# organize a single reaction directory
pyscf_auto organize --reaction-dir ~/pyscf_runs/water_opt --apply

# organize all under allowed_root
pyscf_auto organize --root ~/pyscf_runs --apply
```

Search organized outputs:

```bash
pyscf_auto organize --find --run-id run_20260223_120000_01234567
pyscf_auto organize --find --job-type single_point --limit 20 --json
```

Rebuild index:

```bash
pyscf_auto organize --rebuild-index --json
```

## Utility Scripts

Diagnostics and validation are provided as scripts:

```bash
./scripts/preflight_check.sh
./scripts/validate_inp.py input/water_opt.inp
./scripts/validate_runtime_config.py --config ~/.pyscf_auto/config.yaml
```

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
- `--profile`: enable profiling
- `--verbose`: debug logging
- `--config PATH`: app config path

## status

Show run status for a reaction directory.

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt --json
```

## organize

Preview or apply organization of completed runs.

```bash
# dry-run all runs under allowed_root
pyscf_auto organize

# organize a single reaction directory
pyscf_auto organize --reaction-dir ~/pyscf_runs/water_opt --apply

# organize all under a root
pyscf_auto organize --root ~/pyscf_runs --apply
```

Search organized outputs:

```bash
pyscf_auto organize --find RUN_ID
pyscf_auto organize --find RUN_ID --job-type opt --limit 20 --json
```

## doctor

Run dependency/runtime diagnostics.

```bash
pyscf_auto doctor
```

## validate

Validate a `.inp` file without running a job.

```bash
pyscf_auto validate input/water_opt.inp
```

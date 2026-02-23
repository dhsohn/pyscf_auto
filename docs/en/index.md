# pyscf_auto User Manual (English)

pyscf_auto is a local run/retry tool built on PySCF and ASE.
It executes optimization, single-point, frequency, IRC, and scan jobs from `.inp`
reaction directories with consistent state tracking and metadata.

## Quick Start

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

pyscf_auto run-inp --reaction-dir ~/pyscf_runs/example_reaction
```

## What This Manual Covers

- `run-inp` retry execution model
- Run status inspection and report files
- Organizing completed results
- Configuration and troubleshooting

## Default Paths

- App config: `~/.pyscf_auto/config.yaml`
- Default run root: `~/pyscf_runs`
- Default organized root: `~/pyscf_outputs`

## Next Reads

- [Install & Quickstart](getting-started.md)
- [CLI](cli.md)
- [Outputs](outputs.md)
- [Troubleshooting](troubleshooting.md)

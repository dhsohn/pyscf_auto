# pyscf_auto User Manual (English)

pyscf_auto is a local workflow tool built on PySCF and ASE. It runs optimization, single-point, frequency, IRC, and scans with consistent logging and metadata.

## Quick Start

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

pyscf_auto run path/to/input.xyz --config run_config.yaml
```

## What This Manual Covers

- Calculation modes and workflow flowcharts
- Queue/background execution and status transitions
- Scan execution including manifest-based distributed runs
- Configuration structure and key options
- Output files and troubleshooting

## Default Paths

- Default run directory: `~/pyscf_auto/runs/YYYY-MM-DD_HHMMSS/`
- Override with the `PYSCF_AUTO_BASE_DIR` environment variable.

## Next Reads

- [Install & Quickstart](getting-started.md)
- [Workflows](workflows.md)
- [Queue & Background](queue.md)
- [Scan](scan.md)

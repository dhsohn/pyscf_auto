# Troubleshooting

## SMD Errors

- You need an SMD-enabled PySCF build.
- Example message: "Install the SMD-enabled PySCF package..."

```bash
conda install -c daehyupsohn -c conda-forge pyscf
```

## SCF Convergence Failures

- pyscf_auto retries with level shift/damping by default.
- Disable retries with `PYSCF_AUTO_SCF_RETRY=0`.

## Queue Appears Stuck

- Check status: `pyscf_auto queue status`
- Check runner log: `~/pyscf_auto/log/queue_runner.log`
- Requeue failed: `pyscf_auto queue requeue-failed`

## Memory/Threads

- `memory_gb` is passed to PySCF `max_memory`.
- Threading depends on OpenMP availability.

## Diagnostics

```bash
pyscf_auto doctor
```

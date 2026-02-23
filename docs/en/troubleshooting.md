# Troubleshooting

## SMD-Related Errors

- You need a PySCF build with SMD support.
- Typical fix:

```bash
conda install -c daehyupsohn -c conda-forge pyscf
```

## SCF Convergence Failures

- pyscf_auto applies retry patches automatically.
- Increase retry budget with `--max-retries` if needed.
- Review `run_state.json` and attempt directories for details.

## "Run already in progress" Errors

- Another process is already running for the same reaction directory.
- Wait for completion or run a different reaction directory.

## Memory / Threads

- `memory_gb` and `threads` are set in `.inp` `%runtime` block.
- Effective threading depends on OpenMP availability.

## Diagnostics

```bash
pyscf_auto doctor
```

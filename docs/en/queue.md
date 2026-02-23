# Concurrency & Locking

Current `pyscf_auto` CLI does not expose a queue command.
Execution is local and directory-scoped through `run-inp`.

## What Happens on Concurrent Runs

- Each reaction directory uses a run lock.
- Starting `run-inp` twice for the same directory will reject the second run.
- Different reaction directories can run in parallel.

## Practical Pattern

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/reaction_A
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/reaction_B
```

## Notes

- Retry is built into each run (`--max-retries` / config default).
- Progress and terminal status are written to `run_state.json` in the reaction directory.

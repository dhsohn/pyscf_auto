# Queue & Background Execution

## Basics

- Use `--background` to enqueue a run; the queue runner executes it.
- Foreground runs are also recorded for status tracking and show up in `pyscf_auto queue status`.

## Queue/Background Flow

```mermaid
flowchart TD
  A[pyscf_auto run] --> B{background?}
  B -->|yes| C[enqueue run]
  C --> D[start queue runner]
  D --> E[queue runner picks entry]
  E --> F[subprocess runs pyscf_auto without background]
  F --> G[update status]
  B -->|no| H[foreground run]
  H --> I[register queue entry]
  I --> G
```

## Status Transitions

```mermaid
stateDiagram-v2
  [*] --> queued: enqueue / requeue
  queued --> running: queue runner picks entry
  queued --> canceled: queue cancel
  running --> completed: exit_code == 0
  running --> failed: exit_code != 0 or stale recovery
  running --> timeout: max_runtime_seconds exceeded
  failed --> queued: queue retry / requeue-failed
  timeout --> queued: queue retry / requeue-failed
  canceled --> queued: queue retry (manual)
  completed --> queued: queue retry (manual)
```

## Common Commands

```bash
pyscf_auto run input.xyz --config run_config.yaml --background

pyscf_auto queue status
pyscf_auto queue cancel <RUN_ID>
pyscf_auto queue retry <RUN_ID>
pyscf_auto queue requeue-failed
pyscf_auto queue prune --keep-days 30
pyscf_auto queue archive
```

## Related Files

- Queue file: `~/pyscf_auto/runs/queue.json`
- Queue runner log: `~/pyscf_auto/log/queue_runner.log`

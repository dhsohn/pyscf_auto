# 큐와 백그라운드 실행

## 기본 개념

- `--background` 옵션으로 실행하면 큐에 들어가고, 큐 러너가 실행을 담당합니다.
- 포그라운드 실행도 큐 상태에 기록되어 `pyscf_auto queue status`에서 함께 보입니다.

## 큐/백그라운드 흐름

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

## 상태 전이

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

## 주요 명령어

```bash
pyscf_auto run input.xyz --config run_config.yaml --background

pyscf_auto queue status
pyscf_auto queue cancel <RUN_ID>
pyscf_auto queue retry <RUN_ID>
pyscf_auto queue requeue-failed
pyscf_auto queue prune --keep-days 30
pyscf_auto queue archive
```

## 관련 파일

- 큐 파일: `~/pyscf_auto/runs/queue.json`
- 큐 러너 로그: `~/pyscf_auto/log/queue_runner.log`

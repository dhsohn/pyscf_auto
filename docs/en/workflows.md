# Run Execution Model

`pyscf_auto` executes one reaction directory at a time with `run-inp`.
The newest `.inp` file in that directory is selected, parsed, and then executed
with automatic retry.

## Top-Level Flow

```mermaid
flowchart TD
  A[pyscf_auto run-inp --reaction-dir DIR] --> B[select latest .inp]
  B --> C[parse .inp -> config]
  C --> D[load or create run_state.json]
  D --> E[attempt loop]
  E --> F[execute attempt_NNN]
  F --> G{completed?}
  G -->|yes| H[finalize completed]
  G -->|no| I{retry budget left?}
  I -->|yes| J[apply retry strategy]
  J --> E
  I -->|no| K[finalize failed]
```

## Retry Behavior

- Total attempts = `1 + max_retries`
- `max_retries` default comes from app config: `runtime.default_max_retries`
- Retry patches are recorded per attempt in `run_state.json`

## Status Files

Each reaction directory stores:

- `run_state.json`: machine-readable state
- `run_report.json`: summarized result
- `run_report.md`: human-readable report

## Run Status Values

- `running`
- `retrying`
- `completed`
- `failed`
- `interrupted`

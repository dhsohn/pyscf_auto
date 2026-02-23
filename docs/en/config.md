# Configuration

pyscf_auto uses two configuration layers.

## 1) Calculation Input (`.inp`)

Each reaction directory must contain at least one `.inp` file.
`run-inp` selects the most recently modified one.

A typical `.inp` includes:

- Route line (`! ...`) with job type/functional/basis
- Optional `%...` blocks (`%scf`, `%optimizer`, `%runtime`, etc.)
- Geometry block (`* xyz ... *` or `* xyzfile ...`)

Example:

```text
! Opt B3LYP def2-SVP D3BJ PCM(water)

%scf
  max_cycle 300
  conv_tol 1e-10
end

%runtime
  threads 4
  memory_gb 8
end

* xyz 0 1
O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0
*
```

Validate input without running:

```bash
pyscf_auto validate path/to/input.inp
```

## 2) App Runtime Config (`config.yaml`)

App-level runtime behavior is loaded from:

1. `--config PATH`
2. `PYSCF_AUTO_CONFIG`
3. `~/.pyscf_auto/config.yaml`

Key fields:

- `runtime.allowed_root`: allowed root for `--reaction-dir`
- `runtime.organized_root`: target root for `organize`
- `runtime.default_max_retries`: default retry count

Example:

```yaml
runtime:
  allowed_root: ~/pyscf_runs
  organized_root: ~/pyscf_outputs
  default_max_retries: 5
```

# Scan

## Overview

Scan jobs are triggered by setting the route line job type to `Scan` in `.inp`.
Run them with the same command used for other jobs:

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/scan_case
```

## Minimal `.inp` Example (1D Bond Scan)

```text
! Scan B3LYP def2-SVP

%scan
  type bond
  i 0
  j 1
  start 1.0
  end 1.8
  step 0.1
  mode optimization
  executor local
end

* xyz 0 1
O 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.970000
H 0.000000 0.920000 -0.240000
*
```

## Common Scan Keys

- `type`: `bond`, `angle`, `dihedral`
- Atom indices: `i`, `j`, `k`, `l`
- Range: `start`, `end`, `step`
- `mode`: `optimization` or `single_point`
- `executor`: `serial` or `local`

## Outputs

Scan outputs are written in the reaction directory and attempt directories,
including scan result JSON/CSV artifacts.

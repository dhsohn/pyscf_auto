# Workflows

## Calculation Modes

- `optimization`: geometry optimization (optionally followed by frequency/IRC/single-point)
- `single_point`: single-point energy
- `frequency`: harmonic frequencies (Hessian)
- `irc`: IRC (imaginary mode + IRC)
- `scan`: 1D/2D scan

## Top-Level Flow

```mermaid
flowchart TD
  A[CLI: pyscf_auto run] --> B[load config + build run context]
  B --> C{background?}
  C -->|yes| D[enqueue background run]
  C -->|no| E[setup logging + metadata]
  E --> F{calculation_mode}
  F -->|optimization| G[optimization stage]
  F -->|single_point| H[single-point stage]
  F -->|frequency| I[frequency stage]
  F -->|irc| J[IRC stage]
  F -->|scan| K[scan stage]
  G --> G1[optional frequency]
  G1 --> G2[optional IRC]
  G2 --> G3[optional single-point]
  I --> I1[optional IRC]
  I1 --> I2[optional single-point]
  J --> J1[optional single-point]
```

## Post-Optimization and Frequency Steps

- `frequency_enabled`: enable frequency after optimization.
- `irc_enabled`: enable IRC after optimization or frequency.
- `single_point_enabled`: enable single-point after optimization, frequency, or IRC.

With `calculation_mode: irc`, the optional single-point runs after IRC completes.

## TS Quality Gating

IRC and single-point execution are gated by imaginary count and TS quality checks.

- Expected imaginary count: 1 for TS optimization (`optimizer.mode: transition_state`), 0 otherwise.

### IRC gate

```mermaid
flowchart TD
  A[irc_enabled?] -->|no| Z[skip IRC]
  A -->|yes| B{frequency_enabled?}
  B -->|no| C[proceed IRC]
  B -->|yes| D{imaginary_count available?}
  D -->|no| E{ts_quality.enforce?}
  E -->|yes| Z1[skip IRC]
  E -->|no| C
  D -->|yes| F{allow_irc set?}
  F -->|yes| G{allow_irc?}
  G -->|no| H{ts_quality.enforce?}
  H -->|yes| Z2[skip IRC]
  H -->|no| C
  G -->|yes| C
  F -->|no| I{imag_count == expected?}
  I -->|yes| C
  I -->|no| J{ts_quality.enforce?}
  J -->|yes| Z3[skip IRC]
  J -->|no| C
```

### Single-point gate

```mermaid
flowchart TD
  A[single_point_enabled?] -->|no| Z[skip SP]
  A -->|yes| B{frequency_enabled?}
  B -->|no| C[run SP]
  B -->|yes| D{imaginary_count available?}
  D -->|no| E{ts_quality.enforce?}
  E -->|yes| Z1[skip SP]
  E -->|no| C
  D -->|yes| F{allow_single_point set?}
  F -->|yes| G{allow_single_point?}
  G -->|no| H{ts_quality.enforce?}
  H -->|yes| Z2[skip SP]
  H -->|no| C
  G -->|yes| C
  F -->|no| I{imag_count == expected?}
  I -->|yes| C
  I -->|no| J{ts_quality.enforce?}
  J -->|yes| Z3[skip SP]
  J -->|no| C
```

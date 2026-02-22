# 워크플로우

## 계산 모드

- `optimization`: 구조 최적화 (선택적으로 frequency/IRC/single-point를 이어서 실행)
- `single_point`: 단일점 에너지
- `frequency`: 진동수 (Hessian 포함)
- `irc`: IRC (가상 모드 계산 후 실행)
- `scan`: 1D/2D 스캔

## 전체 실행 흐름

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

## 최적화 및 주파수 후속 단계

- `frequency_enabled`: 최적화 후 진동수 계산 수행 여부.
- `irc_enabled`: 최적화 또는 주파수 후 IRC 실행 여부.
- `single_point_enabled`: 최적화/주파수/IRC 후 단일점 계산 수행 여부.

`calculation_mode: irc`에서는 IRC 완료 후 단일점 계산을 선택적으로 실행합니다.

## TS 품질(ts_quality) 게이트

주파수 결과의 imaginary count 및 TS 품질 검사 결과에 따라 IRC/단일점 실행이 자동으로 gating 됩니다.

- 기대 imaginary count: TS 최적화(`optimizer.mode: transition_state`)는 1, 일반 최소화는 0

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

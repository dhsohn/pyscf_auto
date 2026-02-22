# 설정 파일

pyscf_auto 설정 파일은 `.json`, `.yaml/.yml`, `.toml`을 지원합니다.

## 핵심 필드

| 필드 | 설명 |
| --- | --- |
| `calculation_mode` | `optimization`, `single_point`, `frequency`, `irc`, `scan` |
| `basis`, `xc` | 기준 basis/functional |
| `solvent`, `solvent_model` | 용매 설정 (`pcm`/`smd`, vacuum은 생략 가능) |
| `dispersion` | 분산 보정 (`d3bj`, `d3zero`, `d4` 등) |
| `scf` | SCF 설정 (`max_cycle`, `conv_tol`, `chkfile`, `extra`) |
| `optimizer` | 최적화 설정 (`mode`, `ase`) |
| `single_point` | 단일점 override 설정 |
| `frequency_enabled` | 최적화 후 주파수 실행 여부 |
| `single_point_enabled` | 최적화/주파수/IRC 후 단일점 실행 여부 |
| `irc_enabled` | 최적화/주파수 후 IRC 실행 여부 |
| `scan` / `scan2d` | 스캔 설정 (`dimensions`, `mode`, `executor`) |
| `ts_quality` | TS 품질 검사 옵션 (`enforce` 등) |
| `threads`, `memory_gb` | 자원 설정 |
| `io` | 쓰기 간격 (`scan_write_interval_points` 등) |

`calculation_mode: frequency`에서는 `irc_enabled`, `single_point_enabled`로
주파수 계산 뒤 후속 단계를 제어합니다. `calculation_mode: irc`에서는
`single_point_enabled`가 IRC 완료 후 단일점을 제어합니다.

## 기본 예시 (최적화 + 주파수 + 단일점)

```yaml
calculation_mode: optimization
basis: def2-svp
xc: b3lyp
solvent: vacuum
optimizer:
  mode: minimum
scf:
  max_cycle: 200
single_point_enabled: true
frequency_enabled: true
```

## 스캔 예시

```yaml
calculation_mode: scan
scan:
  mode: optimization
  executor: local
  dimensions:
    - type: bond
      i: 0
      j: 1
      start: 1.0
      end: 1.4
      step: 0.2
```

## 유효성 검사

```bash
pyscf_auto validate-config run_config.yaml
```

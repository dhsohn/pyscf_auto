# 스캔

## 개요

`.inp` 라우트 라인 작업 타입을 `Scan`으로 설정하면 스캔 계산을 수행합니다.
실행 명령은 다른 작업과 동일합니다.

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/scan_case
```

## 최소 `.inp` 예시 (1D 결합 스캔)

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

## 자주 쓰는 스캔 키

- `type`: `bond`, `angle`, `dihedral`
- 원자 인덱스: `i`, `j`, `k`, `l`
- 범위: `start`, `end`, `step`
- `mode`: `optimization` 또는 `single_point`
- `executor`: `serial` 또는 `local`

## 출력

스캔 결과는 반응 디렉터리와 `attempt_*` 디렉터리 내에 JSON/CSV 형태로 저장됩니다.

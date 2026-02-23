# 설정

pyscf_auto는 설정을 두 계층으로 사용합니다.

## 1) 계산 입력 (`.inp`)

각 반응 디렉터리에는 최소 1개의 `.inp` 파일이 필요합니다.
`run-inp`는 가장 최근 수정된 `.inp`를 선택해 실행합니다.

일반적인 `.inp` 구성:

- 라우트 라인(`! ...`): 작업 타입/함수/기저
- 선택 블록(`%scf`, `%optimizer`, `%runtime` 등)
- 지오메트리 블록(`* xyz ... *` 또는 `* xyzfile ...`)

예시:

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

실행 없이 입력 유효성만 검사:

```bash
pyscf_auto validate path/to/input.inp
```

## 2) 앱 런타임 설정 (`config.yaml`)

앱 레벨 설정은 다음 순서로 로드됩니다.

1. `--config PATH`
2. `PYSCF_AUTO_CONFIG`
3. `~/.pyscf_auto/config.yaml`

핵심 필드:

- `runtime.allowed_root`: `--reaction-dir` 허용 루트
- `runtime.organized_root`: `organize` 출력 루트
- `runtime.default_max_retries`: 기본 재시도 횟수

예시:

```yaml
runtime:
  allowed_root: ~/pyscf_runs
  organized_root: ~/pyscf_outputs
  default_max_retries: 5
```

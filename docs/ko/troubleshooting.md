# 문제 해결

## SMD 관련 오류

- SMD를 지원하는 PySCF 빌드가 필요합니다.
- 일반적인 해결 방법:

```bash
conda install -c daehyupsohn -c conda-forge pyscf
```

## SCF 수렴 실패

- pyscf_auto는 자동으로 재시도 패치를 적용합니다.
- 필요하면 `--max-retries`로 재시도 횟수를 늘리세요.
- 상세 원인은 `run_state.json`과 `attempt_*` 디렉터리를 확인하세요.

## "이미 실행 중" 오류

- 같은 반응 디렉터리에서 다른 프로세스가 이미 실행 중입니다.
- 기존 실행이 끝나길 기다리거나 다른 디렉터리에서 실행하세요.

## 메모리 / 스레드

- `.inp`의 `%runtime` 블록에서 `memory_gb`, `threads`를 설정합니다.
- 실제 스레드 적용은 OpenMP 환경에 따라 달라질 수 있습니다.

## 진단

```bash
./scripts/preflight_check.sh
```

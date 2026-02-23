# 동시 실행과 락

현재 `pyscf_auto` CLI는 별도 queue 명령을 노출하지 않습니다.
실행은 `run-inp` 기준의 로컬 디렉터리 단위로 동작합니다.

## 동시 실행 시 동작

- 반응 디렉터리마다 run lock을 사용합니다.
- 같은 디렉터리에서 `run-inp`를 동시에 실행하면 두 번째 실행은 거부됩니다.
- 서로 다른 반응 디렉터리는 병렬 실행할 수 있습니다.

## 권장 실행 패턴

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/reaction_A
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/reaction_B
```

## 참고

- 재시도는 각 실행 내부에서 처리됩니다(`--max-retries` 또는 설정 기본값).
- 진행 상태와 종료 상태는 반응 디렉터리의 `run_state.json`에 기록됩니다.

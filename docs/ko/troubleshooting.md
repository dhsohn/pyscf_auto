# 문제 해결

## SMD 관련 오류

- SMD가 포함된 PySCF가 필요합니다.
- 메시지 예: "Install the SMD-enabled PySCF package..."

```bash
conda install -c daehyupsohn -c conda-forge pyscf
```

## SCF 수렴 실패

- pyscf_auto는 기본적으로 level shift/damping 재시도를 수행합니다.
- 재시도를 끄려면 환경 변수 `PYSCF_AUTO_SCF_RETRY=0`를 사용하세요.

## 큐가 멈춘 것처럼 보일 때

- 큐 상태 확인: `pyscf_auto queue status`
- 러너 로그 확인: `~/pyscf_auto/log/queue_runner.log`
- 실패 건 재큐: `pyscf_auto queue requeue-failed`

## 메모리/스레드

- `memory_gb`는 PySCF의 `max_memory`에 전달됩니다.
- `threads`는 OpenMP 환경에 따라 반영되지 않을 수 있습니다.

## 진단

```bash
pyscf_auto doctor
```

# pyscf_auto 사용자 매뉴얼 (한국어)

pyscf_auto는 PySCF와 ASE를 기반으로 한 로컬 워크스테이션용 계산 워크플로우 도구입니다. 최적화, 단일점, 진동수, IRC, 스캔을 일관된 로그/메타데이터로 실행합니다.

## 빠른 시작

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

pyscf_auto run path/to/input.xyz --config run_config.yaml
```

## 이 매뉴얼에서 다루는 내용

- 계산 모드와 워크플로우 흐름
- 큐/백그라운드 실행 및 상태 전이
- 스캔(1D/2D) 실행과 매니페스트 기반 분산 실행
- 설정 파일 구조와 주요 옵션
- 결과 파일 구조 및 트러블슈팅

## 기본 경로

- 기본 실행 디렉터리: `~/pyscf_auto/runs/YYYY-MM-DD_HHMMSS/`
- `PYSCF_AUTO_BASE_DIR` 환경 변수로 기본 경로를 변경할 수 있습니다.

## 다음으로 읽기

- [설치 및 시작](getting-started.md)
- [워크플로우](workflows.md)
- [큐와 백그라운드 실행](queue.md)
- [스캔](scan.md)

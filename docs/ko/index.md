# pyscf_auto 사용자 매뉴얼 (한국어)

pyscf_auto는 PySCF/ASE 기반의 로컬 계산 실행·재시도 도구입니다.
`.inp` 반응 디렉터리를 기준으로 최적화, 단일점, 진동수, IRC, 스캔 계산을
상태 파일과 메타데이터와 함께 일관되게 수행합니다.

## 빠른 시작

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

pyscf_auto run-inp --reaction-dir ~/pyscf_runs/example_reaction
```

## 이 매뉴얼에서 다루는 내용

- `run-inp` 재시도 실행 모델
- 실행 상태 확인과 리포트 파일
- 완료 결과 정리(organize)
- 설정과 트러블슈팅

## 기본 경로

- 앱 설정: `~/.pyscf_auto/config.yaml`
- 기본 실행 루트: `~/pyscf_runs`
- 기본 정리 루트: `~/pyscf_outputs`

## 다음으로 읽기

- [설치 및 시작](getting-started.md)
- [CLI 레퍼런스](cli.md)
- [결과물](outputs.md)
- [트러블슈팅](troubleshooting.md)

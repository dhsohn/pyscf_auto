# CLI 레퍼런스

## run-inp

반응 디렉터리에서 가장 최신 `.inp` 파일을 골라 계산을 실행합니다.

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/water_opt
```

주요 옵션:

- `--max-retries N`: 재시도 횟수 덮어쓰기
- `--force`: 이미 완료된 경우에도 강제 재실행
- `--json`: JSON 요약 출력
- `--profile`: 프로파일링 활성화
- `--verbose`: 디버그 로그
- `--config PATH`: 앱 설정 파일 경로

## status

반응 디렉터리 실행 상태를 확인합니다.

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt --json
```

## organize

완료된 결과를 정리할 경로를 미리보기/적용합니다.

```bash
# allowed_root 전체 드라이런
pyscf_auto organize

# 단일 반응 디렉터리 정리
pyscf_auto organize --reaction-dir ~/pyscf_runs/water_opt --apply

# 특정 루트 전체 정리
pyscf_auto organize --root ~/pyscf_runs --apply
```

정리된 결과 검색:

```bash
pyscf_auto organize --find RUN_ID
pyscf_auto organize --find RUN_ID --job-type opt --limit 20 --json
```

## doctor

의존성/런타임 진단을 실행합니다.

```bash
pyscf_auto doctor
```

## validate

계산 없이 `.inp` 파일 유효성만 검사합니다.

```bash
pyscf_auto validate input/water_opt.inp
```

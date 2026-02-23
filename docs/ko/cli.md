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

전역 옵션(명령 앞에 위치):

- `--verbose` / `-v`: 디버그 로그
- `--config PATH`: 앱 설정 파일 경로

예시:

```bash
pyscf_auto --config ~/.pyscf_auto/config.yaml -v run-inp --reaction-dir ~/pyscf_runs/water_opt
```

## status

반응 디렉터리 실행 상태를 확인합니다.

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt --json
```

## organize

완료된 결과를 정리할 경로를 미리보기/적용합니다.

`organize`는 아래 중 하나를 반드시 지정해야 합니다.

- `--reaction-dir DIR`
- `--root ROOT` (`allowed_root`와 정확히 같은 경로만 허용)

```bash
# 단일 반응 디렉터리 정리
pyscf_auto organize --reaction-dir ~/pyscf_runs/water_opt --apply

# allowed_root 전체 정리
pyscf_auto organize --root ~/pyscf_runs --apply
```

정리된 결과 검색:

```bash
pyscf_auto organize --find --run-id run_20260223_120000_01234567
pyscf_auto organize --find --job-type single_point --limit 20 --json
```

인덱스 재생성:

```bash
pyscf_auto organize --rebuild-index --json
```

## 유틸리티 스크립트

진단/검증은 스크립트로 제공합니다.

```bash
./scripts/preflight_check.sh
./scripts/validate_inp.py input/water_opt.inp
./scripts/validate_runtime_config.py --config ~/.pyscf_auto/config.yaml
```

# 설치 및 시작

## 설치

pyscf_auto는 conda 채널로 배포됩니다.

```bash
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto
```

- `pip install pyscf_auto`는 지원되지 않습니다.

## 환경 점검

```bash
pyscf_auto doctor
```

## 앱 설정

pyscf_auto는 아래 우선순위로 앱 설정을 읽습니다.

1. `--config` 옵션
2. `PYSCF_AUTO_CONFIG` 환경 변수
3. `~/.pyscf_auto/config.yaml` (기본)

최소 예시(`~/.pyscf_auto/config.yaml`):

```yaml
runtime:
  allowed_root: ~/pyscf_runs
  organized_root: ~/pyscf_outputs
  default_max_retries: 5
```

## 반응 디렉터리 준비

`runtime.allowed_root` 아래 반응 디렉터리에 `.inp` 파일을 둡니다.

```bash
mkdir -p ~/pyscf_runs/water_opt
cp input/water_opt.inp ~/pyscf_runs/water_opt/
```

## 첫 실행

```bash
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/water_opt
```

## 상태 확인

```bash
pyscf_auto status --reaction-dir ~/pyscf_runs/water_opt
```

실행 산출물은 반응 디렉터리 안에 생성됩니다.

- `run_state.json`
- `run_report.json`
- `run_report.md`

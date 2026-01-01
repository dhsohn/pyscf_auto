# pDFT

PySCF(DFT/SCF/gradient/Hessian)와 ASE(최적화 드라이버)를 결합해 **구조 최적화(최소점/전이상태), 단일점(SP) 에너지, 프리퀀시(진동수) 분석**을 한 번에 실행하는 경량 워크플로우 스크립트입니다.

- 엔트리포인트: `pdft` CLI (내부 구현은 `run_opt.py`, 모듈은 `pdft_core/` 패키지)
- 기본 설정 템플릿: `run_config.json`
- 출력은 실행 시점별로 `runs/YYYY-MM-DD_HHMMSS/` 아래에 자동으로 정리됩니다.

---

## 프로그램 상세 설명

이 프로젝트는 **입력 구조(XYZ) → PySCF 계산 → ASE/Sella 최적화 → 결과/메타데이터 정리**의 흐름을 하나의 CLI로 묶어, 최소점/TS 최적화와 후속 SP/프리퀀시 계산까지 일관된 방식으로 처리합니다. 각 단계는 내부적으로 분리된 모듈이 담당하며, 사용자는 CLI 수준에서 **계산 모드 선택 → 설정/입력 지정 → 실행 → 결과 확인**만 수행하면 됩니다.

특히 다음의 목적을 염두에 두고 설계되었습니다.
- **재현성**: 실제로 사용된 설정/버전/환경 정보가 자동 저장되어 동일 조건 재실행이 용이합니다.
- **자동화**: 최적화 후의 프리퀀시 및 단일점 계산을 옵션으로 연결하여 불필요한 수동 작업을 줄입니다.
- **실험적 기능 확장**: 스캔/IRC/열화학 등 후속 분석을 같은 실행 컨텍스트에서 이어갈 수 있습니다.

### 실행 흐름(요약)
1. **입력/설정 로딩**: `pdft_core/run_opt.py`가 CLI 인자를 해석하고, `pdft_core/run_opt_config.py`에서 JSON 설정을 로드/검증합니다.
2. **구조/전하/스핀 준비**: XYZ 파일과 메타데이터를 읽고, `charge`/`multiplicity`가 없으면 전자수 기반으로 추정합니다.
3. **화학 계산 엔진 구성**: `run_opt_engine.py`가 PySCF 설정(기저, 함수, SCF 옵션, 용매 모델 등)을 적용합니다.
4. **옵티마이저 선택/구동**:
   - 최소점: ASE(BFGS 등) + PySCF 그라디언트
   - TS: Sella 기반 1차 안장점 탐색(`order=1`)
5. **후속 계산**: 선택 시 프리퀀시 계산 → 결과 검증(허수 진동수 확인) → SP 계산을 자동으로 연결합니다. TS일 경우 허수 모드 검증 및 IRC 옵션을 이어서 수행할 수 있습니다.
6. **결과 정리/기록**: `pdft_core/run_opt_logging.py`와 `pdft_core/run_opt_metadata.py`가 실행 로그, 이벤트 로그, 설정 스냅샷, 결과 파일을 표준 디렉토리 구조로 정리합니다. 결과는 실행 단위로 분리된 디렉터리에 저장되어 병렬 실행/관리에도 유리합니다.

### 핵심 설계 포인트
- **단일 엔트리포인트**: `pdft` 하나로 `run/doctor/validate-config/status/queue` 서브커맨드를 제공하며, 인터랙티브/비-인터랙티브 및 큐 실행까지 지원합니다. (내부 구현은 `pdft_core/` 패키지)
- **계산/설정 분리**: JSON 설정 템플릿을 통해 계산 조건을 재현 가능하게 유지합니다. 실행 시 사용된 설정은 `config_used.json`으로 저장되어 비교/재실행이 쉽습니다.
- **분산/용매 모듈화**: 분산 보정(D3/D4)과 용매 모델(PCM/SMD)을 필요 시 활성화하는 구조입니다. XC 함수 자체에 분산이 포함된 경우에는 중복 적용을 방지합니다.
- **진단/상태 유틸리티**: `pdft doctor`, `pdft status`, `pdft queue` 등으로 환경과 실행 상태를 빠르게 점검할 수 있습니다. 운영 중에도 문제 원인을 빠르게 식별하도록 로그/메타데이터가 표준화되어 있습니다.

### 결과물 구성(한 번의 실행 기준)
- **실행 로그**: `run.log`, `log/run_events.jsonl`
- **계산 결과**: `optimized.xyz`, `frequency_result.json` 등
- **재현 정보**: `metadata.json`, `config_used.json`

---

## 주요 특징

### 1) 계산 기능
- **구조 최적화(최소점)**: ASE 옵티마이저(BFGS 등) + PySCF DFT 그라디언트(힘). 수렴 조건(`fmax`, `steps`)을 설정으로 제어합니다.
- **전이상태(TS) 최적화**: Sella 기반 1차 안장점 탐색(`order=1`). 허수 모드 검증 및 IRC 후속 계산으로 반응 좌표를 확인할 수 있습니다.
- **단일점 에너지(SP)**: 최적화 구조에서 선택한 함수/기저/용매/분산으로 에너지 계산. 최적화 후 자동 실행하도록 연결할 수 있습니다.
- **프리퀀시(진동수) 계산**: PySCF Hessian → harmonic analysis로 진동수(허수 진동수 포함) 산출  
  - 현재 구현은 **Hessian에 분산보정(D3/D4)을 포함하지 않고**, 필요 시 **에너지 보정 항으로만** 더하는 모드가 기본입니다(설정에서 `"frequency_dispersion_mode": "none"`).
- **IRC 계산**: TS 모드에서 허수 모드 벡터를 기반으로 반응 좌표를 추적(ASE 기반).  
  - `irc_result.json`에 forward/reverse 경로와 에너지 프로파일이 저장됩니다.
- **열화학(thermochemistry)**: 프리퀀시 결과로부터 ZPE/엔탈피/엔트로피/깁스 자유 에너지를 계산(옵션).  
  - `thermo` 설정(T/P/단위)을 제공하면 `frequency_result.json`과 `metadata.json`에 함께 기록됩니다.

### 2) 용매 모델
- `vacuum`(기본): 용매 처리 없음. 기체상 계산을 위한 기본 모드입니다.
- `pcm`: 유전율(ε)이 필요하며 `solvent_dielectric.json`에서 solvent → ε를 조회합니다. 새로운 용매는 JSON에 추가 등록하여 사용합니다.
- `smd`: PySCF가 SMD를 포함하도록 빌드되어야 합니다. 빌드 옵션(`ENABLE_SMD=ON`)을 사용해 활성화합니다.

### 3) 분산 보정(Dispersion)
- `d3bj`, `d3zero`, `d4`를 지원합니다. 설정에서 선택한 보정은 에너지/기울기에 반영됩니다.
> 참고: XC 함수 자체에 dispersion이 포함된 것으로 판단되면(예: 이름이 `...-d`, `...d3` 등으로 끝나는 경우) 별도 D3/D4 설정을 무시하도록 되어 있습니다.

### 4) 스캔(1D/2D) 계산
구조의 특정 결합/각/이면각을 **그리드 스캔**하며 각 점에서 최적화 또는 단일점 계산을 수행합니다. 반응좌표나 잠재에너지 표면(PES)을 빠르게 훑어보는 용도에 적합합니다.

- **지원 차원**: 1D/2D 스캔(2D는 두 개의 좌표를 동시에 지정)
- **지원 타입**: `bond`, `angle`, `dihedral` (입력은 원자 인덱스 기준)
- **실행 모드**: `optimization` 또는 `single_point`
- **입력 방식**: `--scan-dimension`, `--scan-grid`, `--scan-mode`로 CLI에서 간단히 지정
- **CLI 우선 규칙**: 설정 파일/인터랙티브 입력에 `scan`이 있어도, CLI로 지정한 차원/그리드/모드가 우선 적용됩니다(나머지 `scan` 키는 유지).
- **그리드 방식**:
  - 범위 기반: `start,end,step`로 균일 간격 스캔
  - 명시적 리스트: `--scan-grid`로 임의의 값 목록 지정
- **결과물**: 스캔 포인트별 결과가 run 디렉터리 하위에 정리되며, `scan_result.json/csv`에서 에너지/수렴 여부/구조 경로를 한 번에 확인할 수 있습니다.

예: 1D bond 스캔(0-1 결합 길이)
```bash
python run_opt.py input.xyz \
  --config run_config.json \
  --non-interactive \
  --scan-dimension "bond,0,1,1.0,2.0,0.1" \
  --scan-mode optimization
```

예: 2D 스캔 + 커스텀 그리드
```bash
python run_opt.py input.xyz \
  --config run_config.json \
  --non-interactive \
  --scan-dimension "bond,0,1,1.0,2.0,0.1" \
  --scan-dimension "angle,0,1,2,90,130,5" \
  --scan-grid "1.0,1.2,1.4,1.6,1.8,2.0" \
  --scan-grid "90,100,110,120,130" \
  --scan-mode single_point
```

### 5) 입력 XYZ의 charge / multiplicity 처리
- `.xyz`의 **2번째 줄(comment line)** 에 아래처럼 메타데이터를 넣을 수 있습니다.
  - 예: `charge=0 multiplicity=1`
- 지정하지 않으면 전자수 parity로 spin을 자동 추정하며, 라디칼/TS/금속/diradical에서는 오상태 위험이 있어 경고가 출력됩니다. 이 경우 사용자가 직접 `multiplicity`를 명시하는 것이 안전합니다.

### 6) 백그라운드 큐 & 상태/진단 유틸리티
- `--background`로 큐에 넣어 **백그라운드 실행** 가능 (우선순위/타임아웃 지원). 긴 계산을 순차적으로 처리할 때 유용합니다.
- `pdft queue status/cancel/retry/archive/prune` 등으로 **큐 상태 관리**를 수행합니다.
- `pdft status <run_dir>` 및 `--recent`로 **실행 결과 요약 출력**(성공/실패/상태/핵심 파일).
- `pdft doctor`, `pdft validate-config`로 **환경 진단/설정 검증**(의존성 누락/설정 오류 점검).

---

## 디렉토리 구조(요약)

```
pDFT/
  run_opt.py                 # 메인 CLI/워크플로우(래퍼)
  src/
    pdft_core/
      __init__.py
      run_opt.py               # 메인 CLI/워크플로우
      run_opt_engine.py         # PySCF 기반 SP/frequency/solvent 등 화학 로직
      run_opt_dispersion.py     # D3/D4 파싱 및 백엔드 매핑(통합)
      run_opt_config.py         # config 로딩/검증
      run_opt_logging.py        # 로깅/이벤트 로깅
      run_opt_metadata.py       # 메타데이터/결과 정리
      run_opt_resources.py      # 스레드/메모리 리소스 제어
  run_config.json             # 기본 최적화 템플릿 (최소점/TS)
  solvent_dielectric.json     # PCM 유전율 맵
  runs/                       # 실행 결과가 자동 생성되는 폴더
```

---

## 설치

지원 Python 버전: **3.12**

### 권장: Conda 환경
기본 경로는 “conda로 과학계 스택 설치 + PySCF를 **소스 빌드**”입니다.

#### 1) `environment.yml`로 기본 환경 구성
```bash
conda env create -f environment.yml
conda activate pDFT
```

- `environment.yml`에는 **Python 3.12**, PySCF 빌드를 위한 툴체인/라이브러리, ASE, D3/D4 의존성이 포함되어 있습니다.
- TS 최적화(Sella)는 **필수**이므로 `sella`를 추가로 설치합니다.

```bash
python -m pip install sella
```

#### 2) 잠금 파일 사용(권장, 재현성 필요 시)
플랫폼별 잠금이 필요하면 `conda-lock`으로 잠금 파일을 생성합니다.

```bash
conda install -c conda-forge conda-lock -y
conda-lock lock -f environment.yml -p linux-64
conda-lock install --name pdft conda-lock.yml
```

```bash
conda install -c conda-forge conda-lock -y
conda-lock lock -f environment.yml -p osx-arm64
conda-lock install --name pdft conda-lock.yml
```

```bash
conda install -c conda-forge conda-lock -y
conda-lock lock -f environment.yml -p win-64
conda-lock install --name pdft conda-lock.yml
```

#### 3) PySCF 소스 빌드
```bash
git clone https://github.com/pyscf/pyscf.git
cd pyscf

mkdir -p build
cmake -S pyscf/lib -B build \
  -DENABLE_SMD=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build -j4

python -m pip install -e . --no-build-isolation
```

---

## 실행 방법

### 1) 인터랙티브 모드(기본, 추천)
프로젝트 루트에서 실행:

```bash
conda activate pDFT
cd pDFT
python run_opt.py
```

실행하면 아래 흐름으로 선택합니다.

1. 계산 종류 선택
   - 구조 최적화 / 단일점 / 프리퀀시
2. (구조 최적화일 때) intermediates vs transition states(TS) 선택
3. 입력 XYZ 경로, 기저(basis), 함수(xc), 용매모델 선택
4. (옵션) 최적화 후 프리퀀시 수행 여부 선택
5. (옵션) 프리퀀시 결과가 기대(예: TS이면 허수 1개)되면 SP 수행 여부 선택

> 인터랙티브 모드에서는 실제로 사용된 설정이 `runs/.../config_used.json`로 저장됩니다.

### 2) 비-인터랙티브 모드(배치 실행)
`--non-interactive`를 사용하면 JSON 설정 파일과 xyz 입력만으로 실행합니다.

예: 최소점 최적화 템플릿을 그대로 사용
```bash
python run_opt.py input.xyz --config run_config.json --non-interactive
```

예: TS 최적화 템플릿 사용
```bash
python run_opt.py input_ts.xyz --config run_config.json --non-interactive
```
TS 최적화를 위해서는 `run_config.json`에서 아래 값을 `transition_state`용으로 조정하세요.
```json
{
  "optimizer": {
    "mode": "transition_state",
    "output_xyz": "ts_optimized.xyz",
    "ase": {
      "optimizer": "sella",
      "trajectory": "ts_opt.traj",
      "logfile": "ts_opt.log",
      "sella": { "order": 1 }
    }
  }
}
```

예: IRC 계산 모드(비-인터랙티브)
```bash
python run_opt.py input_ts.xyz --config run_config.json --non-interactive
```

### 2-1) 스캔 계산(비-인터랙티브 전용)
스캔 계산은 CLI 옵션으로만 지정하며, **인터랙티브 모드에서는 사용할 수 없습니다.**

```bash
python run_opt.py input.xyz \
  --config run_config.json \
  --non-interactive \
  --scan-dimension "dihedral,0,1,2,3,0,180,15" \
  --scan-mode single_point
```

### 3) 재시작(resume)
기존 실행 디렉터리(`runs/...`)를 이어서 계산하려면 `--resume`을 사용합니다.  
이때 run 디렉터리의 `checkpoint.json`과 `config_used.json`을 읽어 입력/설정을 복구합니다.

```bash
python run_opt.py --resume runs/2024-01-01_120000
```

- `--resume`는 `--run-dir`과 동시에 사용할 수 없습니다.
- `completed/failed/timeout/canceled` 상태의 run 디렉터리를 재시작하려면 명시적으로 `--force-resume`를 지정해야 합니다.

```bash
python run_opt.py --resume runs/2024-01-01_120000 --force-resume
```
```json
{
  "calculation_mode": "irc",
  "optimizer": {
    "mode": "transition_state",
    "ase": { "optimizer": "sella", "sella": { "order": 1 } }
  },
  "irc_file": "irc_result.json",
  "irc": { "steps": 10, "step_size": 0.05, "force_threshold": 0.01 }
}
```

유용한 옵션/명령:
- `--run-dir <dir>`: 출력 폴더를 직접 지정
- `--run-id <uuid>`: run id를 고정
- `--solvent-map <json>`: solvent dielectric map 경로 지정
- `pdft validate-config [config.json]`: config 검증만 수행하고 종료
- `pdft status <run_dir|metadata.json>`: 특정 실행의 상태 요약 출력
- `pdft status --recent <N>`: 최근 N개 실행 요약 출력
- `pdft doctor`: 환경 진단(의존성/solvent map 등) 후 종료
- `--scan-dimension`, `--scan-grid`, `--scan-mode`: 스캔 계산 전용 옵션

---

## 백그라운드 큐 실행/관리

### 1) 큐에 실행 등록
```bash
python run_opt.py input.xyz --config run_config.json --non-interactive --background
```

옵션:
- `--queue-priority <int>`: 우선순위(높을수록 먼저 실행)
- `--queue-max-runtime <sec>`: 최대 실행 시간(초)

### 2) 큐 상태 확인/관리
```bash
pdft queue status
pdft queue cancel <RUN_ID>
pdft queue retry <RUN_ID>
pdft queue requeue-failed
pdft queue archive [--path <archive.json>]
pdft queue prune --keep-days 30
```

큐 파일은 `runs/queue.json`에 저장되며, 큐 러너 로그는 `log/queue_runner.log`에 기록됩니다.

---

## 유틸리티 명령

### 환경 진단
```bash
pdft doctor
```

### 설정 검증(단축 명령 지원)
```bash
pdft validate-config run_config.json
```

> 하위 호환 플래그/alias: 기존 `--doctor`, `--validate-only`, `--status*`, `--queue-*` 플래그는 **하위 호환(alias)** 으로 유지되며, 신규 서브커맨드 사용을 기본으로 권장합니다.

---

## 내장 기능 상세(명령 옵션 가이드)

### 실행 모드/입력
- `--interactive`: 대화형 입력(기본). 설정을 선택하면서 진행합니다.
- `--non-interactive` (`--advanced`): JSON/XYZ를 명시해 배치 실행합니다.
- `--config <path>`: 사용할 JSON 설정 파일 경로.
- `--solvent-map <path>`: 용매 유전율 맵(JSON) 경로.

### 실행 제어/복구
- `--run-dir <dir>`: 결과를 기록할 디렉터리를 지정합니다.
- `--run-id <uuid>`: run id를 고정합니다(대기열/재현성 관리에 유용).
- `--resume <dir>`: 기존 run 디렉터리에서 재개합니다.
- `--force-resume`: 완료/실패/타임아웃 run도 재개 허용합니다.

### 상태/진단
- `pdft status <run_dir|metadata.json>`: 특정 실행 요약 출력.
- `pdft status --recent <N>`: 최근 N개 실행 요약 출력.
- `pdft doctor`: 의존성/환경 점검(설치 누락, solvent map 등).
- `pdft validate-config [config.json]`: 설정 파일만 검증하고 종료합니다.

### 백그라운드 큐
- `--background`: 큐에 등록해 백그라운드 실행.
- `pdft queue status`: 큐/실행 상태를 요약 출력.
- `pdft queue cancel <RUN_ID>`: 대기열 항목 취소.
- `pdft queue retry <RUN_ID>`: 실패/타임아웃 항목 재시도.
- `pdft queue requeue-failed`: 실패/타임아웃 항목 일괄 재등록.
- `pdft queue archive [--path <archive.json>]`: 큐를 아카이브 파일로 저장 후 초기화.
- `pdft queue prune --keep-days <N>`: 완료/실패/타임아웃/취소 항목 중 오래된 항목 정리.
- `--queue-priority <int>`: 우선순위(높을수록 먼저 실행).
- `--queue-max-runtime <sec>`: 최대 실행 시간(초).

### 스캔 옵션(비-인터랙티브 전용)
- `--scan-dimension "type,i,j[,k[,l]],start,end,step"`  
  - `type`: `bond`(2개 인덱스), `angle`(3개), `dihedral`(4개)  
  - `start/end/step`: 스캔 범위/간격  
  - 2D 스캔 시 `--scan-dimension`을 **2번** 지정합니다.
- `--scan-grid "v1,v2,..."`: 차원별 **명시적 그리드** 지정(차원 수만큼 반복 입력).
- `--scan-mode`: 각 점에서 수행할 계산 모드(`optimization` 또는 `single_point`).

---

## 디버깅/테스트

### 1) 내장 설정 검증 테스트(가벼운 스모크 체크)
목적: 기본 템플릿 설정(JSON)이 로드/스키마 검증을 통과하는지 빠르게 확인합니다.

실행:
```bash
pdft validate-config run_config.json
```

필요한 파일:
- 설정 파일: `run_config.json`

로그/출력 위치:
- 콘솔(stdout)에 검증 결과가 출력됩니다. (`runs/` 폴더는 생성되지 않습니다.)

성공 기준:
- `Config validation passed: <config>` 메시지가 표시되고 종료 코드가 0이면 성공입니다.

### 2) pytest 기반 단위 테스트
목적: 설정 파서/분산 설정 로직 등 핵심 유틸리티의 회귀를 확인합니다.

실행:
```bash
python -m pytest tests
```

필요한 파일:
- 테스트가 참조하는 설정 템플릿: `run_config.json`

로그/출력 위치:
- pytest 결과가 콘솔(stdout)에 출력됩니다.

성공 기준:
- `tests/...`가 모두 `passed`로 표시되면 성공입니다.

---

## 출력(Results)

각 실행은 `runs/YYYY-MM-DD_HHMMSS/` 아래에 폴더가 생성되며, 보통 다음이 포함됩니다.

- `run.log`: 전체 로그
- `log/run_events.jsonl`: 이벤트 로그(JSONL)
- `metadata.json`: 실행 메타데이터(환경/버전/시간/성공여부 등)
- `config_used.json` 또는 `config_used.json`에 준하는 설정 스냅샷
- `optimized.xyz` / `<output_xyz>`: 최적화 결과 구조
- `ase_opt.traj` 또는 `ts_opt.traj`: ASE trajectory (옵티마이저 설정에 따라)
- `frequency_result.json`: 프리퀀시 결과(실행한 경우)
- `irc_result.json`: IRC 결과(IRC 모드 또는 IRC 후속 계산 실행 시)
- `qcschema_result.json`: QCSchema AtomicResult 요약(실행 완료 시)
  - `schema_name`, `schema_version`: QCSchema 버전 정보
  - `molecule`: 입력 구조(원소/좌표/전하/다중도)
  - `return_result`: 최종 에너지(Hartree)
  - `extras.pdft`: 계산 메타데이터와 stage 결과 스냅샷
- `irc_profile.csv`: IRC 경로 에너지 프로파일 CSV
  - `direction`: forward/reverse 방향
  - `step`: 스텝 인덱스
  - `energy_ev`: 에너지(eV)
  - `energy_hartree`: 에너지(Hartree)
- `irc_forward.xyz`, `irc_reverse.xyz`: IRC 경로 구조(ASE 출력)
- `scan_result.json`: 스캔 결과(JSON, 스캔 모드에서 생성)
- `scan_result.csv`: 스캔 결과 CSV(스캔 모드에서 생성)
  - `index`: 스캔 포인트 인덱스
  - `values.*`: 스캔 차원별 값(예: `bond_0_1`)
  - `energy`: 단일점 에너지
  - `converged`: 수렴 여부
  - `cycles`: SCF 사이클 수
  - `optimizer_steps`: 최적화 스텝 수(optimization 모드)
  - `input_xyz`: 입력 구조 파일 경로
  - `output_xyz`: 최적화 결과 구조 파일 경로(optimization 모드)

---

## 설정 파일(JSON) 핵심 필드

### 공통
- `calculation_mode`: `"optimization"`, `"single_point"`, `"frequency"`, `"irc"`
- `threads`, `memory_gb`: 계산 리소스
- `basis`, `xc`: 기저/함수
- `dispersion`: `"d3bj"`, `"d3zero"`, `"d4"` 또는 `null`
- `solvent`, `solvent_model`, `solvent_map`: 용매 설정
- `scf`: PySCF SCF 설정
  - `max_cycle`, `conv_tol`, `diis`, `level_shift`, `damping`
  - `extra`: 세부 옵션(그리드/DF/초기 추정 등)
    - `grids.level`, `grids.prune`: DFT 그리드 설정
    - `density_fit`: `true|false` (밀도피팅, 지원 SCF 타입에서만 동작)
    - `init_guess`: PySCF `init_guess` 문자열(예: `"atom"`, `"minao"`)

예: `scf.extra` 사용
```json
"scf": {
  "max_cycle": 200,
  "extra": {
    "grids": {
      "level": 4,
      "prune": "sg1"
    },
    "density_fit": true,
    "init_guess": "atom"
  }
}
```

### 구조 최적화
- `optimizer.output_xyz`: 최종 구조 저장 파일명
- `optimizer.mode`: `"minimum"` 또는 `"transition_state"`(TS)
- `optimizer.ase.optimizer`: `"bfgs"` 또는 `"sella"`(TS에서 주로 사용)
- `optimizer.ase.fmax`, `optimizer.ase.steps`: 수렴 조건/최대 스텝
- `optimizer.ase.trajectory`, `optimizer.ase.logfile`: 진행 로그/trajectory 파일명

### 분산(D3) 관련
- (선택) `optimizer.ase.d3_params.damping`: `s6, s8, a1, a2` 등

### IRC 관련
- `irc_enabled`: `true|false` (최적화 모드에서 IRC 후속 계산을 강제)
- `irc_file`: IRC 결과 파일 경로(기본값: `irc_result.json`)
- `qcschema_output_file`: QCSchema 결과 파일 경로(기본값: `qcschema_result.json`)
- `irc_profile_csv_file`: IRC 프로파일 CSV 경로(기본값: `irc_profile.csv`)
- `irc.steps`, `irc.step_size`, `irc.force_threshold`: IRC 경로 추적 설정

### SCF `extra` 정책/제약
- `density_fit` 지원 범위:
  - PySCF에서 `density_fit()` 메서드를 제공하는 mean-field 객체에서만 동작합니다.
  - 일반적으로 DFT(RKS/UKS/ROKS 등)에서 지원되며, 지원하지 않는 SCF 타입에서는 오류를 반환합니다.
- `init_guess` vs `chkfile` 우선순위:
  1. `scf.chkfile`이 존재하고 `scf/dm`이 저장되어 있으면 **chkfile의 density matrix가 최우선**으로 사용되며 `init_guess`는 무시됩니다.
  2. `chkfile`이 존재하지만 `scf/dm`이 없으면, `init_guess`가 없을 때만 `init_guess="chkfile"`로 자동 설정됩니다. (사용자 `init_guess`가 있으면 그대로 유지)
  3. `chkfile`이 없거나 경로가 비어 있으면 `init_guess` 설정만 적용됩니다.

### 열화학(thermochemistry) 관련
- `thermo.T`, `thermo.P`, `thermo.unit`: 온도/압력 설정(예: `"atm"`, `"bar"`, `"Pa"`)

---

## 트러블슈팅

### 1) `KeyError: 'bj'` (D3 damping 관련)
- 원인: `dftd3-python` 백엔드에 `damping="bj"`를 넘긴 경우
- 해결: `dftd3-python`에서는 `damping="d3bj"` 형태가 필요하며, 설정 파서가 이를 자동 매핑하도록 되어 있어야 합니다.

### 2) `JSONDecodeError: Extra data`
- 원인: 설정 파일이 “JSON 객체 2개가 붙어있음” 등으로 문법이 깨짐
- 해결:
  ```bash
  python -m json.tool run_config.json > /dev/null
  ```

### 3) `cannot import name 'dft' from 'pyscf'`
- 원인: 환경 꼬임/namespace 충돌(가짜 `pyscf`가 import됨) 가능성이 큼
- 해결:
  ```bash
  python -c "import pyscf; print(pyscf.__file__)"
  ```
  출력이 이상하거나 `None`이면 PySCF를 재설치/환경 정리가 필요합니다.

### 4) `ModuleNotFoundError: No module named 'pytest'` (테스트 실행 시)
- 원인: 개발용 테스트 의존성이 설치되지 않음
- 해결:
  ```bash
  python -m pip install -r requirements-dev.txt
  ```

### 6) `FileNotFoundError: run_config.json` (테스트/검증 실행 시)
- 원인: 저장소 루트가 아닌 다른 경로에서 실행
- 해결: `cd pDFT` 후 실행하거나, 설정 파일 경로를 절대/상대 경로로 정확히 지정하세요.

---

## 참고/크레딧
- PySCF: 전자구조 계산 엔진(DFT/SCF/gradient/Hessian)
- ASE: 최적화/Atoms 모델
- Sella: 전이상태 최적화
- simple-dftd3(dftd3-python): D3(BJ) 분산보정

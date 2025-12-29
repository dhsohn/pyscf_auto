# pDFT

PySCF(DFT/SCF/gradient/Hessian)와 ASE(최적화 드라이버)를 결합해 **구조 최적화(최소점/전이상태), 단일점(SP) 에너지, 프리퀀시(진동수) 분석**을 한 번에 실행하는 경량 워크플로우 스크립트입니다.

- 엔트리포인트: `run_opt.py`
- 기본 설정 템플릿: `run_config_ase.json`(최소점), `run_config_ts.json`(TS)
- 출력은 실행 시점별로 `runs/YYYY-MM-DD_HHMMSS/` 아래에 자동으로 정리됩니다.

---

## 주요 특징

### 1) 계산 기능
- **구조 최적화(최소점)**: ASE 옵티마이저(BFGS 등) + PySCF DFT 그라디언트(힘)
- **전이상태(TS) 최적화**: Sella 기반 1차 안장점 탐색(`order=1`)
- **단일점 에너지(SP)**: 최적화 구조에서 선택한 함수/기저/용매/분산으로 에너지 계산
- **프리퀀시(진동수) 계산**: PySCF Hessian → harmonic analysis로 진동수(허수 진동수 포함) 산출  
  - 현재 구현은 **Hessian에 분산보정(D3/D4)을 포함하지 않고**, 필요 시 **에너지 보정 항으로만** 더하는 모드가 기본입니다(설정에서 `"frequency_dispersion_mode": "none"`).

### 2) 용매 모델
- `vacuum`(기본): 용매 처리 없음
- `pcm`: 유전율(ε)이 필요하며 `solvent_dielectric.json`에서 solvent → ε를 조회
- `smd`: PySCF가 SMD를 포함하도록 빌드/설치된 경우에만 사용 가능

### 3) 분산 보정(Dispersion)
- `d3bj`, `d3zero`, `d4`를 지원합니다.
- **권장 백엔드: `dftd3-python`(simple-dftd3)**  
  외부 `dftd3` 실행파일 없이 동작하도록 설계되어 있습니다.
- ASE 외부 바이너리 래퍼(legacy)도 지원하지만, 그 경우 `d3_command`로 실제 `dftd3` 경로가 필요합니다.

> 참고: XC 함수 자체에 dispersion이 포함된 것으로 판단되면(예: 이름이 `...-d`, `...d3` 등으로 끝나는 경우) 별도 D3/D4 설정을 무시하도록 되어 있습니다.

### 4) 입력 XYZ의 charge / multiplicity 처리
- `.xyz`의 **2번째 줄(comment line)** 에 아래처럼 메타데이터를 넣을 수 있습니다.
  - 예: `charge=0 multiplicity=1`
- 지정하지 않으면 전자수 parity로 spin을 자동 추정하며, 라디칼/TS/금속/diradical에서는 오상태 위험이 있어 경고가 출력됩니다.

### 5) 백그라운드 큐 & 상태/진단 유틸리티
- `--background`로 큐에 넣어 **백그라운드 실행** 가능 (우선순위/타임아웃 지원)
- `--queue-status`, `--queue-cancel`, `--queue-retry` 등으로 **큐 상태 관리**
- `--status`, `--status-recent`로 **실행 결과 요약 출력**
- `--doctor`, `--validate-only`로 **환경 진단/설정 검증**

---

## 디렉토리 구조(요약)

```
pDFT/
  run_opt.py                 # 메인 CLI/워크플로우
  run_opt_chemistry.py        # PySCF 기반 SP/frequency/solvent 등 화학 로직
  run_opt_dispersion.py       # D3/D4 파싱 및 백엔드 매핑(통합)
  run_opt_config.py           # config 로딩/검증
  run_opt_logging.py          # 로깅/이벤트 로깅
  run_opt_metadata.py         # 메타데이터/결과 정리
  run_opt_resources.py        # 스레드/메모리 리소스 제어
  run_config_ase.json         # 최소점 최적화 템플릿
  run_config_ts.json          # TS 최적화 템플릿
  solvent_dielectric.json     # PCM 유전율 맵
  runs/                       # 실행 결과가 자동 생성되는 폴더
```

---

## 설치

### 권장: Conda 환경
가장 일반적인 흐름은 “conda로 과학계 스택 설치 + (필요 시) PySCF를 SMD 옵션으로 빌드”입니다.

#### 1) 필수 패키지 설치(최소 기능)
```bash
conda create -n DFT python=3.12 -y
conda activate DFT

conda install -c conda-forge \
  numpy scipy h5py \
  ase \
  libxc xcfun libcint \
  openblas \
  dftd3-python \
  -y

python -m pip install sella
```

- `dftd3-python`는 D3(BJ) 등을 외부 바이너리 없이 쓰기 위한 권장 의존성입니다.
- TS 최적화(Sella)를 쓰려면 `sella`가 필요합니다.

#### 2) PySCF 설치
##### (A) 빠른 설치(권장, SMD가 필요 없으면)
```bash
conda install -c conda-forge pyscf -y
```

##### (B) SMD 사용이 필요하면: PySCF 소스 빌드(ENABLE_SMD=ON)
기본 배포판에서 SMD가 동작하지 않는 환경이라면, 아래처럼 PySCF를 직접 빌드해 SMD를 활성화할 수 있습니다.

```bash
git clone https://github.com/pyscf/pyscf.git
cd pyscf
git checkout v2.11.0

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
conda activate DFT
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
python run_opt.py input.xyz --config run_config_ase.json --non-interactive
```

예: TS 최적화 템플릿 사용
```bash
python run_opt.py input_ts.xyz --config run_config_ts.json --non-interactive
```

유용한 옵션:
- `--run-dir <dir>`: 출력 폴더를 직접 지정
- `--run-id <uuid>`: run id를 고정
- `--solvent-map <json>`: solvent dielectric map 경로 지정
- `--validate-only`: config 검증만 수행하고 종료
- `--status <run_dir|metadata.json>`: 특정 실행의 상태 요약 출력
- `--status-recent <N>`: 최근 N개 실행 요약 출력
- `--doctor`: 환경 진단(의존성/solvent map 등) 후 종료

---

## 백그라운드 큐 실행/관리

### 1) 큐에 실행 등록
```bash
python run_opt.py input.xyz --config run_config_ase.json --non-interactive --background
```

옵션:
- `--queue-priority <int>`: 우선순위(높을수록 먼저 실행)
- `--queue-max-runtime <sec>`: 최대 실행 시간(초)

### 2) 큐 상태 확인/관리
```bash
python run_opt.py --queue-status
python run_opt.py --queue-cancel <RUN_ID>
python run_opt.py --queue-retry <RUN_ID>
python run_opt.py --queue-requeue-failed
```

큐 파일은 `runs/queue.json`에 저장되며, 큐 러너 로그는 `log/queue_runner.log`에 기록됩니다.

---

## 유틸리티 명령

### 환경 진단
```bash
python run_opt.py --doctor
```

### 설정 검증(단축 명령 지원)
```bash
python run_opt.py --validate-only --config run_config_ase.json
python run_opt.py validate-config run_config_ase.json
```

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

---

## 설정 파일(JSON) 핵심 필드

### 공통
- `threads`, `memory_gb`: 계산 리소스
- `basis`, `xc`: 기저/함수
- `dispersion`: `"d3bj"`, `"d3zero"`, `"d4"` 또는 `null`
- `solvent`, `solvent_model`, `solvent_map`: 용매 설정
- `scf`: PySCF SCF 설정
  - `max_cycle`, `conv_tol`, `diis`, `level_shift`, `damping`

### 구조 최적화
- `optimizer.output_xyz`: 최종 구조 저장 파일명
- `optimizer.mode`: `"minimum"` 또는 `"transition_state"`(TS)
- `optimizer.ase.optimizer`: `"bfgs"` 또는 `"sella"`(TS에서 주로 사용)
- `optimizer.ase.fmax`, `optimizer.ase.steps`: 수렴 조건/최대 스텝
- `optimizer.ase.trajectory`, `optimizer.ase.logfile`: 진행 로그/trajectory 파일명

### 분산(D3) 관련
- 권장:
  - `optimizer.ase.d3_backend: "dftd3"`
  - `optimizer.ase.d3_command: null`
  - (선택) `optimizer.ase.d3_params.damping`: `s6, s8, a1, a2` 등

---

## 트러블슈팅

### 1) `DFTD3 command '/path/to/dftd3' was not found`
- 원인: `optimizer.ase.d3_backend`가 `"ase"`로 잡혔고, `d3_command`가 실제 경로가 아님
- 해결(권장): `d3_backend="dftd3"`, `d3_command=null`로 통일

### 2) `KeyError: 'bj'` (D3 damping 관련)
- 원인: `dftd3-python` 백엔드에 `damping="bj"`를 넘긴 경우
- 해결: `dftd3-python`에서는 `damping="d3bj"` 형태가 필요하며, 설정 파서가 이를 자동 매핑하도록 되어 있어야 합니다.

### 3) `JSONDecodeError: Extra data`
- 원인: 설정 파일이 “JSON 객체 2개가 붙어있음” 등으로 문법이 깨짐
- 해결:
  ```bash
  python -m json.tool run_config_ts.json > /dev/null
  ```

### 4) `cannot import name 'dft' from 'pyscf'`
- 원인: 환경 꼬임/namespace 충돌(가짜 `pyscf`가 import됨) 가능성이 큼
- 해결:
  ```bash
  python -c "import pyscf; print(pyscf.__file__)"
  ```
  출력이 이상하거나 `None`이면 PySCF를 재설치/환경 정리가 필요합니다.

---

## 참고/크레딧
- PySCF: 전자구조 계산 엔진(DFT/SCF/gradient/Hessian)
- ASE: 최적화/Atoms 모델
- Sella: 전이상태 최적화
- simple-dftd3(dftd3-python): D3(BJ) 분산보정

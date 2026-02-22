# pyscf_auto 리팩토링 가능성 진단 보고서

작성일: 2026-02-06  
대상 저장소: `/Users/daehyupsohn/pyscf_auto`

## 1) 결론 요약

결론: **리팩토링은 충분히 가능하며, 효과가 큽니다.**

- 현재 구조는 실행 가능하고 테스트도 통과하지만, 핵심 로직이 일부 거대 함수에 집중되어 변경 비용이 빠르게 증가하는 상태입니다.
- 코드베이스를 "재작성"할 필요는 없고, **기능 보존형 점진 리팩토링**이 적합합니다.
- 우선순위는 `stage_opt/run_opt_engine/run_opt_config` 분해, 중복 제거, 테스트 범위 확장입니다.

## 2) 진단 방법

- 정적 구조 점검: 파일/함수 길이, 함수 복잡도, 중복 패턴 확인
- 실행 점검: `pytest -q`
- 제한 사항:
  - `ruff` 미설치로 린트 미실행
  - `mypy` 미설치로 타입 점검 미실행
  - `pytest-cov` 미설치로 커버리지 수치 미산출

## 3) 정량 지표 (핵심 근거)

- `src/` 총 코드 라인: **14,253 LOC**
- 가장 긴 함수:
  - `src/workflow/stage_opt.py:69` `run_optimization_stage` (1303 lines)
  - `src/run_opt_config.py:515` `validate_run_config` (787 lines)
  - `src/workflow/__init__.py:151` `run` (647 lines)
  - `src/run_opt_engine.py:1089` `compute_frequencies` (594 lines)
- 근사 복잡도 상위:
  - `run_optimization_stage`: cc~229
  - `validate_run_config`: cc~205
  - `compute_frequencies`: cc~142
  - `workflow.run`: cc~115
- 테스트:
  - `pytest -q` 결과: **32 passed**
  - 테스트 대상은 주로 `config/dispersion/workflow.utils/queue/CLI`이며, 대형 오케스트레이션 함수 직접 검증은 제한적

## 4) 구조적 리팩토링 포인트

### A. 단일 함수 과밀화

- `src/workflow/stage_opt.py:69`  
  최적화, 체크포인트, 스냅샷, 주파수, IRC, SP, 메타데이터, 큐 업데이트를 단일 함수가 처리
- `src/workflow/__init__.py:151`  
  컨텍스트 구성, 솔벤트/분산 정규화, capability check, stage 분기, 메타데이터 초기화까지 집중
- `src/run_opt_engine.py:1089`  
  주파수 계산 함수가 SCF/Hessian/분산/열역학/TS 품질 판정을 모두 포함

### B. 중복 로직

- 용매 키 정규화 중복:
  - `src/run_opt_engine.py:741`
  - `src/run_opt.py:75`
  - `src/run_opt_config.py:36`
- TS quality enforcement 판별 중복:
  - `src/workflow/stage_opt.py:46`
  - `src/workflow/stage_freq.py:23`
- 제약조건 검증/변환 로직 유사 중복:
  - `src/run_opt_engine.py:48`
  - `src/ase_backend.py:301`
  - `src/run_opt_config.py:1214`
- IRC 체크포인트/스냅샷 로직 유사 중복:
  - `src/workflow/stage_opt.py:1085`
  - `src/workflow/stage_irc.py:138`

### C. 설정 검증 파이프라인 이원화

- `run_opt_config`에 Pydantic 모델이 있으나, 실검증은 거대한 수동 validator에 집중
- 결과적으로 규칙 추가/수정 시 동기화 누락 위험 존재

### D. 엔진 추상화 미완성

- `src/engines/base.py`, `src/engines/registry.py`는 존재
- 현재 워크플로우는 엔진 레지스트리를 실사용하지 않음
- 문서화된 설계와 런타임 구조 사이에 갭 존재

### E. 타입 경계 약함

- `src/workflow/types.py`의 `RunContext`가 매우 크고 `Any` 비중이 높아 리팩토링 안전성 저하

## 5) 리팩토링 가능성 평가

평가: **높음 (High)**

- 장점:
  - 단계별 모듈(`stage_sp/freq/irc/scan`)이 이미 존재
  - 메타데이터/리소스/로깅 유틸이 분리되어 있음
  - 테스트가 최소한의 회귀 안전망 역할 수행
- 위험:
  - 거대 함수 분해 시 부수효과(파일 출력, 큐 상태, 체크포인트) 순서 의존성
  - 계산 도메인 특성상 예외 처리 경로가 많아 단계별 회귀 테스트 필요

## 6) 권장 리팩토링 로드맵 (점진)

### Phase 0. 안전망 강화 (1~2일)

- `ruff`, `mypy`, `pytest-cov` 도입
- 핵심 smoke 테스트 추가:
  - `stage_opt`의 gating(IRC/SP 실행 여부)
  - checkpoint update 시나리오
  - frequency 결과 저장 포맷 회귀

### Phase 1. 중복 제거 (3~5일)

- 공통 유틸 모듈 신설:
  - solvent key/solvent normalization
  - ts_quality enforce 판별
  - constraints validator/normalizer
- 중복 함수 교체 후 회귀 테스트 보강

### Phase 2. 거대 함수 분해 (1~2주)

- `run_optimization_stage`를 서비스 단위로 분해:
  - `OptimizationLifecycleService`
  - `FrequencyPostProcessService`
  - `IrcExecutionService`
  - `CheckpointSnapshotService`
- `compute_frequencies`를 단계 함수로 분해:
  - scf/hessian/dispersion/thermochemistry/ts_quality

### Phase 3. 컨텍스트 타입 정리 (3~5일)

- `RunContext`를 도메인 dataclass로 분해
- stage 입력 모델 축소로 인자 전달 오류 감소

### Phase 4. 엔진 추상화 실사용 (1주+)

- `run_opt_engine` 직접 호출 경로를 `engines` 인터페이스로 이관
- 우선 `PySCFEngine` 구현 후 stage에서 registry 사용

## 7) 예상 효과

- 변경 영향 범위 축소 및 회귀 리스크 감소
- 기능 추가(예: 신규 엔진, 신규 stage 옵션) 시 구현 비용 감소
- 장애 원인 추적 시간 단축

## 8) 최종 판단

이 프로젝트는 현재도 동작 안정성은 준수하지만, 구조적 복잡도가 높아 장기 유지보수 비용이 빠르게 상승할 수 있는 단계입니다.  
**지금 시점에서 점진 리팩토링을 시작하는 것이 가장 비용 효율적**입니다.

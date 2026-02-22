# pyscf_auto ORCA 지원 도입 계획 보고서

## 1. 목적
pyscf_auto에 ORCA 실행 백엔드를 추가해 기존 워크플로우(`single_point`, `optimization`, `frequency`, `irc`, `scan`)를 동일한 인터페이스로 ORCA에서도 실행 가능하게 한다.

## 2. 1차 목표 범위
- 지원 스테이지: `single_point`, `optimization`, `frequency`
- 실행 모드: 로컬 실행 우선
- 결과 정규화: 기존 메타데이터/QCSchema 출력 구조 유지
- 기존 PySCF 경로와 완전 분리(백엔드 선택형)

### 2.1 2차 이관 항목(1차 범위 제외)
- ORCA-NEB, 고급 excited-state, 복잡한 relativistic 옵션
- 분산 실행/HPC 스케줄러 통합 고도화

## 3. 현재 구조 활용 전략
현재 `WorkflowEngineAdapter` 기반 호출 구조가 정리되어 있으므로 다음 방식으로 확장한다.
- `WorkflowEngineAdapter` 인터페이스 유지
- ORCA 전용 구현체(예: `OrcaEngineAdapter`) 추가
- 워크플로우 stage 코드는 adapter 주입만 사용

핵심 원칙은 stage 로직 변경 최소화와 엔진 구현체 교체 가능성 확보다.

## 4. 단계별 추진 계획

### Phase 0. 요구사항 고정 (2~3일)
- ORCA 지원 메서드/기능 목록 확정
- config 스키마 초안 정의 (`engine: orca`, binary path, nprocs, memory, keywords)
- 성공 기준 정의:
  - smoke test 통과
  - 샘플 분자 3종에서 정상 수렴/출력 생성

### Phase 1. 실행 기반 구현 (4~5일)
- ORCA 실행 래퍼 모듈 추가
  - 입력 파일 생성
  - 서브프로세스 실행
  - stdout/stderr/log 캡처
  - 실패 코드/타임아웃 처리
- 산출물 경로 규약 통일 (`run_dir` 기준)

### Phase 2. Adapter 구현 (5~7일)
- `OrcaEngineAdapter` 구현:
  - `compute_single_point_energy`
  - `run_ase_optimizer` (가능하면 ASE ORCA calculator 경유)
  - `compute_frequencies`
  - `run_capability_check`
  - 1차 미지원 메서드는 명시적 `NotImplementedError`와 안내 메시지 제공
- stage 호출 경로에서 adapter 선택만으로 동작 검증

### Phase 3. 결과 파서/정규화 (4~5일)
- ORCA output parser 구현:
  - 에너지, 수렴 여부, 사이클 수
  - 진동수/imaginary count
  - 실패 원인 분류(SCF 미수렴, geometry fail 등)
- 기존 metadata schema로 매핑
- QCSchema export 최소 필수 필드 유지

### Phase 4. 테스트/검증/문서화 (4~6일)
- 단위 테스트:
  - 입력 생성/파싱/에러 분기
- 통합 테스트:
  - `single_point`, `optimization`, `frequency` e2e
- smoke profile 추가:
  - `--engine orca` 조합
- 사용자 문서:
  - 설치/환경변수/예제 config/트러블슈팅

## 5. 아키텍처 제안

### 5.1 파일 구조(안)
- `src/workflow/engine_adapter.py`: 공통 인터페이스(기존)
- `src/workflow/engine_orca.py` (신규): ORCA adapter 구현
- `src/orca_runner.py` (신규): 실행/입력파일/프로세스 관리
- `src/orca_parser.py` (신규): 출력 파싱/정규화

### 5.2 config 확장(안)
- `engine: "pyscf" | "orca"`
- `orca.binary`
- `orca.nprocs`
- `orca.maxcore`
- `orca.extra_keywords`

## 6. 리스크 및 대응
- ORCA 출력 포맷 버전 차이:
  - 파서 버전 감지/호환 레이어 도입
- 엔진별 수렴 동작 차이:
  - ORCA 전용 기본 SCF/optimizer preset 제공
- ASE 경유 최적화의 한계:
  - 1차는 안정 우선, 필요 시 ORCA native optimization 경로를 2차에 추가

## 7. 완료 기준 (Definition of Done)
- `engine=orca`로 `single_point/optimization/frequency` 실행 성공
- 메타데이터/결과 파일 형식이 기존 파이프라인과 호환
- 실패 시 원인 분류 및 재현 가능한 로그 제공
- CI에서 ORCA mock 테스트 + 가능 시 실제 실행 smoke 1세트 통과

## 8. 일정 추정
- 총 3~4주 (1인 기준)
  - 주 1: Phase 0~1
  - 주 2: Phase 2
  - 주 3: Phase 3~4
  - 버퍼: 3~5일

## 9. 즉시 실행 가능한 착수 작업
1. `engine` 설정 파서 및 기본값 정책 확정
2. `OrcaEngineAdapter` 스켈레톤 + 메서드 시그니처 추가
3. ORCA 입력 파일 템플릿/러너 최소 구현
4. `single_point` 경로 우선 연결 후 smoke 테스트 1건 통과

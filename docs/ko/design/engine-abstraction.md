# 엔진 추상화 설계

## 목표

- PySCF/ORCA를 동일한 워크플로우 인터페이스로 실행한다.
- 엔진별 기능 차이는 capability check로 분기한다.
- 기존 사용자 설정과 실행 흐름을 최대한 유지한다.

## 범위

- 계산 엔진 API를 추상화하고 PySCF 코드를 그 뒤로 옮길 수 있는 구조를 만든다.
- ORCA 통합은 별도 단계에서 구현한다.

## 제안 구조

```
src/engines/
  base.py        # 공통 타입과 엔진 인터페이스
  registry.py    # 엔진 레지스트리
  pyscf.py       # PySCF 구현 (추후)
  orca.py        # ORCA 구현 (추후)
```

## 설정 키(초안)

```yaml
engine: pyscf   # 기본값
engine_settings:
  pyscf: {}
  orca:
    binary: /path/to/orca
    nprocs: 8
    memory_mb: 8000
```

## 엔진 API 타입 정의 (스켈레톤)

아래 코드는 `src/engines/base.py`에 들어가는 초안입니다.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class EngineCapabilities:
    supports_energy: bool = True
    supports_gradient: bool = False
    supports_hessian: bool = False
    supports_frequency: bool = False
    supports_irc: bool = False
    solvent_models: Sequence[str] = field(default_factory=tuple)
    dispersion_models: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class EngineContext:
    atom_spec: str
    charge: int
    spin: int
    multiplicity: int
    basis: str
    xc: str
    scf_config: Mapping[str, Any] | None
    solvent_model: str | None
    solvent_name: str | None
    solvent_eps: float | None
    dispersion_model: str | None
    constraints: Mapping[str, Any] | None
    run_dir: str | None
    memory_mb: int | None
    thread_count: int | None
    optimizer_mode: str | None
    profiling_enabled: bool = False
    engine_settings: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SinglePointResult:
    energy: float | None
    converged: bool | None
    cycles: int | None
    dispersion: Mapping[str, Any] | None = None
    profiling: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FrequencyResult:
    energy: float | None
    converged: bool | None
    cycles: int | None
    frequencies_wavenumber: Sequence[float] | None
    frequencies_au: Sequence[float] | None
    imaginary_count: int | None
    imaginary_check: Mapping[str, Any] | None
    thermochemistry: Mapping[str, Any] | None
    ts_quality: Mapping[str, Any] | None
    dispersion: Mapping[str, Any] | None = None
    profiling: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ImaginaryModeResult:
    mode: Sequence[float]
    eigenvalue: float | None
    hessian: Sequence[float] | None = None
    profiling: Mapping[str, Any] | None = None


class Engine(Protocol):
    name: str

    def capabilities(self) -> EngineCapabilities: ...
    def normalize_config(self, config: Mapping[str, Any]) -> None: ...

    def single_point(self, ctx: EngineContext) -> SinglePointResult: ...
    def frequency(self, ctx: EngineContext) -> FrequencyResult: ...
    def imaginary_mode(self, ctx: EngineContext) -> ImaginaryModeResult: ...

    def make_ase_calculator(self, ctx: EngineContext) -> Any: ...
```

## 단계적 마이그레이션

1. `src/engines/base.py` + `registry.py` 추가
2. 기존 PySCF 코드를 `PySCFEngine`으로 이동
3. `stage_*`에서 엔진 인터페이스 호출로 교체
4. ORCA 엔진 추가

## 리스크

- ORCA 출력 포맷 변화에 따른 파서 유지보수 부담
- 기능 격차(예: solvent/dispersion/헤시안)
- Windows 프로세스/경로 처리 이슈

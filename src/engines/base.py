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


__all__ = [
    "Engine",
    "EngineCapabilities",
    "EngineContext",
    "FrequencyResult",
    "ImaginaryModeResult",
    "SinglePointResult",
]

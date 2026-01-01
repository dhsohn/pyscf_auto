from typing import Any, NotRequired, TypedDict


class RunContext(TypedDict):
    config_dict: dict[str, Any]
    config_raw: str
    calculation_mode: str
    basis: str
    xc: str
    solvent_name: str | None
    solvent_model: str | None
    dispersion_model: str | None
    optimizer_config: Any
    optimizer_ase_dict: dict[str, Any]
    optimizer_mode: str | None
    constraints: Any
    scan_config: dict[str, Any] | None
    scan_mode: str | None
    solvent_map_path: str
    single_point_config: Any
    thermo: Any
    frequency_enabled: bool
    single_point_enabled: bool
    thread_count: int | None
    memory_gb: float | None
    verbose: bool
    run_dir: str
    log_path: str
    scf_config: dict[str, Any]
    optimized_xyz_path: str
    run_metadata_path: str
    frequency_output_path: str
    irc_output_path: str
    scan_result_path: str
    scan_result_csv_path: str
    event_log_path: str | None
    run_id: str
    attempt: int
    run_id_history: list[str]
    resume_dir: str | None
    pyscf_chkfile: str | None
    irc_enabled: bool
    irc_config: Any
    previous_status: str | None
    checkpoint_path: str
    eps: NotRequired[float | None]
    dispersion_info: NotRequired[dict[str, Any] | None]
    applied_scf: NotRequired[dict[str, Any] | None]
    sp_basis: NotRequired[str]
    sp_xc: NotRequired[str]
    sp_scf_config: NotRequired[dict[str, Any]]
    sp_solvent_name: NotRequired[str | None]
    sp_solvent_model: NotRequired[str | None]
    sp_dispersion_model: NotRequired[str | None]
    sp_solvent_map_path: NotRequired[str]
    sp_eps: NotRequired[float | None]
    freq_dispersion_mode: NotRequired[str]
    freq_dispersion_model: NotRequired[str | None]


class MoleculeContext(TypedDict):
    atom_spec: str
    charge: int
    spin: int
    multiplicity: int
    mol: Any
    mf: Any
    ks_type: str
    total_electrons: int

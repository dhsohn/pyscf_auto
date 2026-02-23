from ase_backend import _run_ase_irc, _run_ase_optimizer
from run_opt_engine import (
    compute_frequencies,
    compute_imaginary_mode,
    compute_single_point_energy,
    load_xyz,
    run_capability_check,
)


class WorkflowEngineAdapter:
    def run_capability_check(self, *args, **kwargs):
        return run_capability_check(*args, **kwargs)

    def compute_frequencies(self, *args, **kwargs):
        return compute_frequencies(*args, **kwargs)

    def compute_imaginary_mode(self, *args, **kwargs):
        return compute_imaginary_mode(*args, **kwargs)

    def compute_single_point_energy(self, *args, **kwargs):
        return compute_single_point_energy(*args, **kwargs)

    def run_ase_optimizer(self, *args, **kwargs):
        return _run_ase_optimizer(*args, **kwargs)

    def run_ase_irc(self, *args, **kwargs):
        return _run_ase_irc(*args, **kwargs)

    def load_xyz(self, *args, **kwargs):
        return load_xyz(*args, **kwargs)


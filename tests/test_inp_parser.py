"""Tests for the .inp file parser."""

import os
import sys
import tempfile
import textwrap

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inp.route_line import RouteLineResult, parse_route_line
from inp.geometry import GeometryResult, parse_geometry_block
from inp.parser import InpConfig, parse_inp_file, inp_config_to_dict, inp_config_to_xyz_content


# --- Route Line Tests ---

class TestParseRouteLine:
    def test_basic_opt(self):
        result = parse_route_line("! Opt B3LYP def2-SVP")
        assert result.job_type == "optimization"
        assert result.optimizer_mode == "minimum"
        assert result.functional == "B3LYP"
        assert result.basis == "def2-svp"

    def test_optts(self):
        result = parse_route_line("! OptTS B3LYP def2-SVP")
        assert result.job_type == "optimization"
        assert result.optimizer_mode == "transition_state"

    def test_single_point(self):
        result = parse_route_line("! SP B3LYP def2-TZVP")
        assert result.job_type == "single_point"
        assert result.basis == "def2-tzvp"

    def test_frequency(self):
        result = parse_route_line("! Freq PBE0 6-31g*")
        assert result.job_type == "frequency"
        assert result.functional == "PBE0"
        assert result.basis == "6-31g*"

    def test_irc(self):
        result = parse_route_line("! IRC B3LYP def2-SVP")
        assert result.job_type == "irc"

    def test_dispersion_d3bj(self):
        result = parse_route_line("! Opt B3LYP def2-SVP D3BJ")
        assert result.dispersion == "d3bj"

    def test_dispersion_d4(self):
        result = parse_route_line("! SP B3LYP def2-SVP D4")
        assert result.dispersion == "d4"

    def test_solvent_pcm(self):
        result = parse_route_line("! Opt B3LYP def2-SVP PCM(water)")
        assert result.solvent_model == "pcm"
        assert result.solvent_name == "water"

    def test_solvent_smd(self):
        result = parse_route_line("! Opt B3LYP def2-SVP SMD(DMSO)")
        assert result.solvent_model == "smd"
        assert result.solvent_name == "DMSO"

    def test_extra_stages(self):
        result = parse_route_line("! OptTS B3LYP def2-SVP +Freq +IRC")
        assert "freq" in result.extra_stages
        assert "irc" in result.extra_stages

    def test_full_route_line(self):
        result = parse_route_line("! OptTS B3LYP def2-SVP D3BJ PCM(water) +Freq +SP")
        assert result.job_type == "optimization"
        assert result.optimizer_mode == "transition_state"
        assert result.functional == "B3LYP"
        assert result.basis == "def2-svp"
        assert result.dispersion == "d3bj"
        assert result.solvent_model == "pcm"
        assert result.solvent_name == "water"
        assert "freq" in result.extra_stages
        assert "sp" in result.extra_stages

    def test_missing_job_type(self):
        with pytest.raises(ValueError, match="job type"):
            parse_route_line("! B3LYP def2-SVP")

    def test_missing_functional(self):
        with pytest.raises(ValueError, match="functional"):
            parse_route_line("! Opt def2-SVP")

    def test_missing_basis(self):
        with pytest.raises(ValueError, match="basis"):
            parse_route_line("! Opt B3LYP")

    def test_empty_route(self):
        with pytest.raises(ValueError, match="empty"):
            parse_route_line("!")

    def test_case_insensitive(self):
        result = parse_route_line("! opt b3lyp DEF2-SVP d3bj")
        assert result.job_type == "optimization"
        assert result.dispersion == "d3bj"


# --- Geometry Block Tests ---

class TestParseGeometryBlock:
    def test_inline_geometry(self):
        lines = [
            "! Opt B3LYP def2-SVP",
            "* xyz 0 1",
            "O  0.0  0.0  0.0",
            "H  1.0  0.0  0.0",
            "H  0.0  1.0  0.0",
            "*",
        ]
        result = parse_geometry_block(lines, "/tmp")
        assert result.charge == 0
        assert result.multiplicity == 1
        assert result.source_type == "inline"
        assert "O" in result.atom_spec
        assert result.atom_spec.count("\n") == 2  # 3 atoms, 2 newlines

    def test_charged_system(self):
        lines = [
            "* xyz -1 2",
            "O  0.0  0.0  0.0",
            "H  1.0  0.0  0.0",
            "*",
        ]
        result = parse_geometry_block(lines, "/tmp")
        assert result.charge == -1
        assert result.multiplicity == 2

    def test_xyzfile_reference(self):
        # Create a temp xyz file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("3\ntest\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
            xyz_path = f.name

        try:
            lines = [
                f"* xyzfile 0 1 {xyz_path}",
            ]
            result = parse_geometry_block(lines, os.path.dirname(xyz_path))
            assert result.charge == 0
            assert result.multiplicity == 1
            assert result.source_type == "xyzfile"
            assert "O" in result.atom_spec
        finally:
            os.unlink(xyz_path)

    def test_missing_geometry(self):
        lines = ["! Opt B3LYP def2-SVP"]
        with pytest.raises(ValueError, match="No geometry block"):
            parse_geometry_block(lines, "/tmp")

    def test_unterminated_geometry(self):
        lines = [
            "* xyz 0 1",
            "O  0.0  0.0  0.0",
        ]
        with pytest.raises(ValueError, match="not terminated"):
            parse_geometry_block(lines, "/tmp")

    def test_invalid_charge(self):
        lines = [
            "* xyz abc 1",
            "O  0.0  0.0  0.0",
            "*",
        ]
        with pytest.raises(ValueError, match="Invalid charge"):
            parse_geometry_block(lines, "/tmp")


# --- Full Parser Tests ---

class TestParseInpFile:
    def _write_inp(self, content: str) -> str:
        """Write an .inp file to a temp location and return its path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".inp", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(textwrap.dedent(content))
            return f.name

    def test_basic_inp_file(self):
        path = self._write_inp("""\
            ! Opt B3LYP def2-SVP
            * xyz 0 1
            O  0.0  0.0  0.0
            H  1.0  0.0  0.0
            H  0.0  1.0  0.0
            *
        """)
        try:
            result = parse_inp_file(path)
            assert result.job_type == "optimization"
            assert result.functional == "B3LYP"
            assert result.basis == "def2-svp"
            assert result.charge == 0
            assert result.multiplicity == 1
        finally:
            os.unlink(path)

    def test_with_blocks(self):
        path = self._write_inp("""\
            ! Opt B3LYP def2-SVP D3BJ PCM(water)

            %scf
              max_cycle 300
              conv_tol 1e-10
            end

            %runtime
              threads 4
              memory_gb 8
            end

            * xyz 0 1
            O  0.0  0.0  0.0
            H  1.0  0.0  0.0
            H  0.0  1.0  0.0
            *
        """)
        try:
            result = parse_inp_file(path)
            assert result.scf["max_cycle"] == 300
            assert result.scf["conv_tol"] == 1e-10
            assert result.runtime["threads"] == 4
            assert result.runtime["memory_gb"] == 8
            assert result.dispersion == "d3bj"
            assert result.solvent_model == "pcm"
            assert result.solvent_name == "water"
        finally:
            os.unlink(path)

    def test_with_tag(self):
        path = self._write_inp("""\
            # TAG: Water_Optimization
            ! Opt B3LYP def2-SVP
            * xyz 0 1
            O  0.0  0.0  0.0
            H  1.0  0.0  0.0
            H  0.0  1.0  0.0
            *
        """)
        try:
            result = parse_inp_file(path)
            assert result.tag == "Water_Optimization"
        finally:
            os.unlink(path)

    def test_extra_stages(self):
        path = self._write_inp("""\
            ! OptTS B3LYP def2-SVP +Freq +IRC
            * xyz 0 1
            O  0.0  0.0  0.0
            H  1.0  0.0  0.0
            H  0.0  1.0  0.0
            *
        """)
        try:
            result = parse_inp_file(path)
            assert result.frequency_enabled is True
            assert result.irc_enabled is True
            assert result.optimizer_mode == "transition_state"
        finally:
            os.unlink(path)


# --- Config Conversion Tests ---

class TestInpConfigToDict:
    def test_basic_conversion(self):
        config = InpConfig(
            job_type="optimization",
            functional="B3LYP",
            basis="def2-svp",
            charge=0,
            multiplicity=1,
            atom_spec="O 0 0 0\nH 1 0 0\nH 0 1 0",
            xyz_source="inline",
        )
        d = inp_config_to_dict(config)
        assert d["basis"] == "def2-svp"
        assert d["xc"] == "B3LYP"
        assert d["calculation_mode"] == "optimization"

    def test_with_solvent(self):
        config = InpConfig(
            job_type="single_point",
            functional="PBE0",
            basis="def2-tzvp",
            charge=0,
            multiplicity=1,
            atom_spec="O 0 0 0",
            xyz_source="inline",
            solvent_name="water",
            solvent_model="pcm",
        )
        d = inp_config_to_dict(config)
        assert d["solvent"] == "water"
        assert d["solvent_model"] == "pcm"

    def test_with_runtime(self):
        config = InpConfig(
            job_type="optimization",
            functional="B3LYP",
            basis="def2-svp",
            charge=0,
            multiplicity=1,
            atom_spec="O 0 0 0",
            xyz_source="inline",
            runtime={"threads": 8, "memory_gb": 16},
        )
        d = inp_config_to_dict(config)
        assert d["threads"] == 8
        assert d["memory_gb"] == 16


class TestInpConfigToXyz:
    def test_xyz_generation(self):
        config = InpConfig(
            job_type="optimization",
            functional="B3LYP",
            basis="def2-svp",
            charge=0,
            multiplicity=1,
            atom_spec="O  0.0  0.0  0.0\nH  1.0  0.0  0.0\nH  0.0  1.0  0.0",
            xyz_source="inline",
        )
        content = inp_config_to_xyz_content(config)
        lines = content.strip().split("\n")
        assert lines[0] == "3"
        assert "charge=0" in lines[1]
        assert "spin=0" in lines[1]
        assert "O" in lines[2]


# --- Molecule Key Tests ---

class TestMoleculeKey:
    def test_hill_formula_water(self):
        from organizer.molecule_key import hill_formula
        assert hill_formula("O 0 0 0\nH 1 0 0\nH 0 1 0") == "H2O"

    def test_hill_formula_ethanol(self):
        from organizer.molecule_key import hill_formula
        spec = "C 0 0 0\nC 1 0 0\nH 2 0 0\nH 3 0 0\nH 4 0 0\nH 5 0 0\nH 6 0 0\nO 7 0 0"
        assert hill_formula(spec) == "C2H5O"

    def test_derive_with_tag(self):
        from organizer.molecule_key import derive_molecule_key
        key = derive_molecule_key("O 0 0 0", tag="My_Molecule")
        assert key == "My_Molecule"

    def test_derive_without_tag(self):
        from organizer.molecule_key import derive_molecule_key
        key = derive_molecule_key("O 0 0 0\nH 1 0 0\nH 0 1 0")
        assert key == "H2O"


# --- Retry Strategy Tests ---

class TestRetryStrategies:
    def test_first_retry_increases_max_cycle(self):
        from runner.retry_strategies import apply_retry_strategy
        config = {"scf": {"max_cycle": 200}}
        modified, patches = apply_retry_strategy(config, 2, "error_scf")
        assert modified["scf"]["max_cycle"] >= 500
        assert len(patches) == 1

    def test_second_retry_adds_level_shift(self):
        from runner.retry_strategies import apply_retry_strategy
        config = {"scf": {"max_cycle": 200}}
        modified, patches = apply_retry_strategy(config, 3, "error_scf")
        assert modified["scf"]["level_shift"] == 0.2
        assert len(patches) == 2

    def test_cumulative_strategies(self):
        from runner.retry_strategies import apply_retry_strategy
        config = {}
        modified, patches = apply_retry_strategy(config, 5, "error_scf")
        assert modified["scf"]["max_cycle"] >= 500
        assert modified["scf"]["level_shift"] == 0.2
        assert modified["scf"]["diis_preset"] == "stable"
        assert modified["scf"]["damping"] == 0.3
        assert len(patches) == 4

    def test_no_retry_on_first_attempt(self):
        from runner.retry_strategies import apply_retry_strategy
        config = {"scf": {"max_cycle": 200}}
        modified, patches = apply_retry_strategy(config, 1, "error_scf")
        assert modified == config
        assert patches == []

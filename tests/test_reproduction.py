"""Check the submission workflow without rerunning the full study."""

from unittest.mock import patch

import numpy as np
import pytest

import methods
from reproduce import REFERENCE, render_cached, study_commands, verify_reference
from simulation import make_task


def test_reference_has_safe_arrays_and_detects_changed_input(tmp_path):
    """A changed cached input must fail validation before rendering."""
    import shutil

    verify_reference()
    with np.load(REFERENCE / "application_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["dates"].dtype.kind == "U"
        assert arrays["exposure"].shape == arrays["deaths"].shape
    for source in REFERENCE.iterdir():
        shutil.copyfile(source, tmp_path / source.name)
    (tmp_path / "full_testing.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_reference(tmp_path)


def test_single_task_preserves_design_and_seed_identity():
    """Selecting one task retains the full-study cell/replication identity."""
    task = make_task("testing", "full", "python", 4, 0)
    assert task["key"] == "testing:4:0"
    assert task["scenario"] == "N0" and task["death_scale"] == 1.0
    with pytest.raises(ValueError, match="cell"):
        make_task("testing", "full", "python", -1, 0)
    with pytest.raises(ValueError, match="replication"):
        make_task("testing", "demo", "python", 4, 100)


def test_existing_source_matched_library_needs_no_compiler():
    """Recipients can load a supplied matching library without rebuilding it."""
    methods.build_c_backend()
    with (
        patch.object(methods, "_compiler", side_effect=RuntimeError("No compiler")),
        patch.object(methods, "_C_HANDLE", None),
    ):
        assert methods.build_c_backend() == methods.C_LIBRARY
        assert methods._load_c_backend() is not None


@pytest.mark.parametrize(
    "system,flags",
    [
        ("linux", ("-shared", "-fPIC")),
        ("darwin", ("-dynamiclib", "-fPIC")),
    ],
)
def test_unix_compilation_uses_shared_position_independent_code(system, flags):
    """Unix builds use PIC and avoid the Windows-specific runtime flag."""
    with patch.object(methods.os, "name", "posix"), patch.object(methods.sys, "platform", system):
        command = methods._compile_command("cc")
    assert all(flag in command for flag in flags)
    assert "-static-libgcc" not in command
    assert command[-1] == "-lm"


def test_study_commands_use_original_manuscript_counts():
    """The master invokes the existing scripts with their frozen counts."""
    commands = study_commands("full", "c", 2, ["simulations", "application", "sensitivity"])
    assert commands[0][-1] == "--resume"
    assert "2000" in commands[1] and "5000" in commands[2]
    assert "20260809" in commands[1] and "20260809" in commands[2]


def test_cached_rendering_creates_all_displays_in_selected_directory(tmp_path):
    """All manuscript displays can be redrawn without patient-level files."""
    report = render_cached(tmp_path)
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 4
    assert len(list((tmp_path / "tables").glob("*.tex"))) == 3
    assert report["source_sha256"]
    assert (tmp_path / "testing_monte_carlo_se.csv").is_file()
    table = (tmp_path / "tables/table_2_testing.tex").read_text()
    assert "0.042" in table and "0.230" in table

import hashlib

from scripts.realdata_sensitivity import _input_hashes


def test_input_hashes_need_no_manifest_and_track_actual_bytes(tmp_path):
    first = tmp_path / "2021.parquet"
    second = tmp_path / "2022.parquet"
    first.write_bytes(b"first snapshot")
    second.write_bytes(b"second snapshot")
    assert not (tmp_path / "DATA_MANIFEST.json").exists()
    expected = {
        first.name: hashlib.sha256(first.read_bytes()).hexdigest(),
        second.name: hashlib.sha256(second.read_bytes()).hexdigest(),
    }
    assert _input_hashes([first, second]) == expected
    first.write_bytes(b"changed snapshot")
    assert _input_hashes([first])[first.name] != expected[first.name]

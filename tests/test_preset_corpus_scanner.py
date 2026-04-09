# Feature: preset-corpus-pipeline, Property 1: Scanner returns exactly all valid .vital files
"""
Property-based test for PresetCorpusScanner.

**Validates: Requirements 1.1, 1.3, 1.4**

Property 1: For any directory tree containing a mix of .vital files
(valid JSON and invalid JSON) and non-.vital files, the
PresetCorpusScanner.scan() result's preset_paths should contain exactly
the set of .vital files with valid JSON, failed_paths should contain
exactly the .vital files with invalid JSON, and total_found should equal
len(preset_paths) + len(failed_paths).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.preset_corpus_scanner import PresetCorpusScanner, ScanResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe filenames: short alphanumeric strings (avoid filesystem edge cases)
_safe_name = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
    min_size=1,
    max_size=12,
)

# Valid JSON content for a .vital file (minimal valid JSON object)
_valid_json = st.dictionaries(
    keys=st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
        min_size=1,
        max_size=8,
    ),
    values=st.one_of(st.integers(-100, 100), st.floats(allow_nan=False, allow_infinity=False)),
    min_size=0,
    max_size=5,
).map(json.dumps)

# Invalid JSON content (guaranteed to fail json.loads)
_invalid_json = st.sampled_from([
    "{invalid json",
    "not json at all",
    "{key: no quotes}",
    "{'single': 'quotes'}",
    "",
    "{",
    "[",
    "}{",
])

# Non-.vital file extensions
_non_vital_ext = st.sampled_from([".txt", ".wav", ".json", ".py", ".xml", ".zip"])


@st.composite
def mixed_directory_layout(draw: st.DrawFn):
    """Generate a specification for a mixed directory with valid .vital,
    invalid .vital, and non-.vital files.

    Returns a tuple of:
      - valid_vital_files: list[(relative_path, json_content)]
      - invalid_vital_files: list[(relative_path, invalid_content)]
      - non_vital_files: list[(relative_path, content)]
    """
    valid_vital: list[tuple[str, str]] = []
    invalid_vital: list[tuple[str, str]] = []
    non_vital: list[tuple[str, str]] = []

    # Use a set to avoid duplicate filenames
    used_names: set[str] = set()

    # Generate valid .vital files (0-5)
    n_valid = draw(st.integers(min_value=0, max_value=5))
    for _ in range(n_valid):
        name = draw(_safe_name)
        while name in used_names:
            name = draw(_safe_name)
        used_names.add(name)
        content = draw(_valid_json)
        valid_vital.append((f"{name}.vital", content))

    # Generate invalid .vital files (0-5)
    n_invalid = draw(st.integers(min_value=0, max_value=5))
    for _ in range(n_invalid):
        name = draw(_safe_name)
        while name in used_names:
            name = draw(_safe_name)
        used_names.add(name)
        content = draw(_invalid_json)
        invalid_vital.append((f"{name}.vital", content))

    # Generate non-.vital files (0-5)
    n_other = draw(st.integers(min_value=0, max_value=5))
    for _ in range(n_other):
        name = draw(_safe_name)
        while name in used_names:
            name = draw(_safe_name)
        used_names.add(name)
        ext = draw(_non_vital_ext)
        non_vital.append((f"{name}{ext}", "some content"))

    return valid_vital, invalid_vital, non_vital


def _create_files(
    tmp_path: Path,
    valid_vital: list[tuple[str, str]],
    invalid_vital: list[tuple[str, str]],
    non_vital: list[tuple[str, str]],
) -> None:
    """Materialize the file layout on disk."""
    for filename, content in valid_vital:
        (tmp_path / filename).write_text(content, encoding="utf-8")
    for filename, content in invalid_vital:
        (tmp_path / filename).write_text(content, encoding="utf-8")
    for filename, content in non_vital:
        (tmp_path / filename).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Property 1 test
# ---------------------------------------------------------------------------


@given(layout=mixed_directory_layout())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.data_too_large])
def test_scanner_returns_exactly_all_valid_vital_files(tmp_path: Path, layout):
    """Property 1: Scanner returns exactly all valid .vital files.

    **Validates: Requirements 1.1, 1.3, 1.4**

    For any directory tree containing a mix of .vital files (valid JSON and
    invalid JSON) and non-.vital files:
    - preset_paths contains exactly the valid-JSON .vital files
    - failed_paths contains exactly the invalid-JSON .vital files
    - total_found == len(preset_paths) + len(failed_paths)
    """
    valid_vital, invalid_vital, non_vital = layout

    # Clean tmp_path to avoid leftover files from previous hypothesis examples
    for child in tmp_path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    _create_files(tmp_path, valid_vital, invalid_vital, non_vital)

    scanner = PresetCorpusScanner()
    result: ScanResult = scanner.scan(tmp_path)

    # Build expected sets of absolute paths
    expected_valid = {tmp_path / fname for fname, _ in valid_vital}
    expected_failed = {tmp_path / fname for fname, _ in invalid_vital}

    actual_valid = set(result.preset_paths)
    actual_failed = set(result.failed_paths)

    # preset_paths contains exactly the valid .vital files
    assert actual_valid == expected_valid, (
        f"preset_paths mismatch.\n"
        f"  Expected: {sorted(str(p) for p in expected_valid)}\n"
        f"  Actual:   {sorted(str(p) for p in actual_valid)}"
    )

    # failed_paths contains exactly the invalid .vital files
    assert actual_failed == expected_failed, (
        f"failed_paths mismatch.\n"
        f"  Expected: {sorted(str(p) for p in expected_failed)}\n"
        f"  Actual:   {sorted(str(p) for p in actual_failed)}"
    )

    # total_found == len(preset_paths) + len(failed_paths)
    assert result.total_found == len(result.preset_paths) + len(result.failed_paths), (
        f"total_found={result.total_found} != "
        f"len(preset_paths)={len(result.preset_paths)} + "
        f"len(failed_paths)={len(result.failed_paths)}"
    )

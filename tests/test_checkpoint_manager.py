"""
CheckpointManager 属性测试与单元测试。

Feature: robust-checkpoint-and-resampling
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib import Path

from src.checkpoint_manager import CheckpointManager, PHASE_ORDER, _NPZ_PHASES


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

phase_names_st = st.sampled_from(PHASE_ORDER)

# JSON-serializable leaf values (no numpy)
json_leaf = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.text(min_size=0, max_size=30),
    st.booleans(),
    st.none(),
)

json_dict_st = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=json_leaf,
    min_size=1,
    max_size=10,
)

# Small numpy arrays for NPZ phases
numpy_array_st = st.builds(
    lambda shape: np.random.default_rng(42).standard_normal(shape).astype(np.float64),
    shape=st.tuples(
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=1, max_value=10),
    ),
)

npz_dict_st = st.dictionaries(
    keys=st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True),
    values=numpy_array_st,
    min_size=1,
    max_size=3,
)


@st.composite
def phase_with_data(draw: st.DrawFn):
    """Draw a phase name together with compatible random data."""
    phase = draw(phase_names_st)
    if phase in _NPZ_PHASES:
        data = draw(npz_dict_st)
    else:
        data = draw(json_dict_st)
    return phase, data


# ---------------------------------------------------------------------------
# Property 1: 阶段检查点持久化完整性
# ---------------------------------------------------------------------------


class TestProperty1PhasePersistenceIntegrity:
    """Feature: robust-checkpoint-and-resampling, Property 1: 阶段检查点持久化完整性

    Validates: Requirements 1.1, 1.3, 1.5
    """

    @given(data=phase_with_data())
    @settings(max_examples=100, deadline=None)
    def test_round_trip_equivalence(self, tmp_path_factory, data):
        """complete_phase() 后 load_phase_data() 应返回等价数据。

        **Validates: Requirements 1.1**
        """
        phase_name, phase_data = data
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        mgr.complete_phase(phase_name, phase_data)
        loaded = mgr.load_phase_data(phase_name)

        if phase_name in _NPZ_PHASES:
            assert set(loaded.keys()) == set(phase_data.keys())
            for key in phase_data:
                np.testing.assert_array_equal(loaded[key], phase_data[key])
        else:
            assert loaded == phase_data

    @given(data=phase_with_data())
    @settings(max_examples=100, deadline=None)
    def test_master_records_correct_status(self, tmp_path_factory, data):
        """checkpoint_master.json 应记录正确的完成状态和检查点文件路径。

        **Validates: Requirements 1.3**
        """
        phase_name, phase_data = data
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        mgr.complete_phase(phase_name, phase_data)

        # Read master file directly from disk
        master = json.loads(mgr.master_path.read_text(encoding="utf-8"))

        phase_info = master["phases"][phase_name]
        assert phase_info["status"] == "completed"
        assert phase_info["checkpoint_file"] is not None
        assert Path(phase_info["checkpoint_file"]).exists()

    @given(data=phase_with_data())
    @settings(max_examples=100, deadline=None)
    def test_master_metadata_fields(self, tmp_path_factory, data):
        """checkpoint_master.json 应包含完成时间戳和输出样本数量。

        **Validates: Requirements 1.5**
        """
        phase_name, phase_data = data
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        mgr.complete_phase(phase_name, phase_data)

        master = json.loads(mgr.master_path.read_text(encoding="utf-8"))
        phase_info = master["phases"][phase_name]

        # Must have a completed_at timestamp
        assert "completed_at" in phase_info
        assert isinstance(phase_info["completed_at"], str)
        assert len(phase_info["completed_at"]) > 0

        # Must have output_count
        assert "output_count" in phase_info
        assert isinstance(phase_info["output_count"], int)
        assert phase_info["output_count"] >= 0

        # Must have file_size_bytes
        assert "file_size_bytes" in phase_info
        assert phase_info["file_size_bytes"] > 0


# ---------------------------------------------------------------------------
# Property 4: 自动恢复阶段检测
# ---------------------------------------------------------------------------


class TestProperty4AutoResumePhaseDetection:
    """Feature: robust-checkpoint-and-resampling, Property 4: 自动恢复阶段检测

    **Validates: Requirements 2.4**
    """

    @given(n=st.integers(min_value=0, max_value=len(PHASE_ORDER)))
    @settings(max_examples=100, deadline=None)
    def test_resume_phase_after_n_consecutive_completions(self, tmp_path_factory, n):
        """完成前 n 个连续阶段后，get_resume_phase() 应返回第 n+1 个阶段或 None。

        **Validates: Requirements 2.4**
        """
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        # Complete the first n phases in order with dummy data
        for i in range(n):
            phase = PHASE_ORDER[i]
            if phase in _NPZ_PHASES:
                data = {"arr": np.array([[1.0, 2.0]])}
            else:
                data = {"dummy": True}
            mgr.complete_phase(phase, data)

        result = mgr.get_resume_phase()

        if n < len(PHASE_ORDER):
            assert result == PHASE_ORDER[n]
        else:
            assert result is None


# ---------------------------------------------------------------------------
# Property 12: 检查点清理策略
# ---------------------------------------------------------------------------


class TestProperty12CheckpointCleanupStrategy:
    """Feature: robust-checkpoint-and-resampling, Property 12: 检查点清理策略

    Validates: Requirements 7.1, 7.2
    """

    @given(
        n_phases=st.integers(min_value=1, max_value=len(PHASE_ORDER)),
    )
    @settings(max_examples=100, deadline=None)
    def test_keep_false_deletes_previous_non_npz(self, tmp_path_factory, n_phases):
        """keep_checkpoints=False 时，完成新阶段后前一阶段的非 .npz 文件应被删除。

        **Validates: Requirements 7.1**
        """
        tmp_path = tmp_path_factory.mktemp("cleanup_false")
        mgr = CheckpointManager(output_dir=tmp_path, keep_checkpoints=False)

        phases_to_complete = PHASE_ORDER[:n_phases]

        for phase in phases_to_complete:
            if phase in _NPZ_PHASES:
                data = {"arr": np.array([[1.0, 2.0]])}
            else:
                data = {"dummy": True}
            mgr.complete_phase(phase, data)

        # After completing all n phases, check the state of each file
        for i, phase in enumerate(phases_to_complete):
            phase_file = mgr._phase_file(phase)
            is_last = i == n_phases - 1

            if is_last:
                # The last completed phase's file should always exist
                assert phase_file.exists(), (
                    f"Last completed phase {phase!r} file should exist"
                )
            else:
                # A non-last phase was the "previous" phase when phase i+1
                # was completed. It should be deleted unless it's .npz.
                if phase_file.suffix == ".npz":
                    assert phase_file.exists(), (
                        f"NPZ phase {phase!r} file should be preserved"
                    )
                else:
                    assert not phase_file.exists(), (
                        f"Non-NPZ phase {phase!r} file should be deleted"
                    )

    @given(
        n_phases=st.integers(min_value=1, max_value=len(PHASE_ORDER)),
    )
    @settings(max_examples=100, deadline=None)
    def test_keep_true_preserves_all(self, tmp_path_factory, n_phases):
        """keep_checkpoints=True 时，所有历史检查点文件应保留。

        **Validates: Requirements 7.2**
        """
        tmp_path = tmp_path_factory.mktemp("cleanup_true")
        mgr = CheckpointManager(output_dir=tmp_path, keep_checkpoints=True)

        phases_to_complete = PHASE_ORDER[:n_phases]

        for phase in phases_to_complete:
            if phase in _NPZ_PHASES:
                data = {"arr": np.array([[1.0, 2.0]])}
            else:
                data = {"dummy": True}
            mgr.complete_phase(phase, data)

        # All checkpoint files should still exist
        for phase in phases_to_complete:
            phase_file = mgr._phase_file(phase)
            assert phase_file.exists(), (
                f"Phase {phase!r} file should be preserved with keep_checkpoints=True"
            )


# ---------------------------------------------------------------------------
# Unit Tests: CheckpointManager 基础功能 (Task 1.5)
# ---------------------------------------------------------------------------


class TestCheckpointManagerUnit:
    """CheckpointManager 基础功能单元测试。

    Requirements: 1.1, 1.2, 1.4, 2.5, 7.3
    """

    # -- 1. 初始化和目录创建 (Req 1.1) --

    def test_init_creates_checkpoint_dir(self, tmp_path):
        """初始化时应创建 checkpoints/ 子目录。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        assert mgr.checkpoint_dir.exists()
        assert mgr.checkpoint_dir == tmp_path / "checkpoints"

    def test_init_creates_master_file(self, tmp_path):
        """初始化时应创建 checkpoint_master.json。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        assert mgr.master_path.exists()
        master = json.loads(mgr.master_path.read_text(encoding="utf-8"))
        assert master["version"] == 3
        assert "created_at" in master
        assert "phases" in master

    def test_init_idempotent(self, tmp_path):
        """多次初始化同一目录不应报错，且保留已有数据。"""
        mgr1 = CheckpointManager(output_dir=tmp_path)
        mgr1.complete_phase("sampling", {"arr": np.array([[1.0]])})

        mgr2 = CheckpointManager(output_dir=tmp_path)
        assert "sampling" in mgr2.get_completed_phases()

    # -- 2. 阶段名称合法性验证 (Req 2.5) --

    @pytest.mark.parametrize("phase", PHASE_ORDER)
    def test_valid_phase_names_accepted(self, tmp_path, phase):
        """7 个合法阶段名称应被 complete_phase 接受。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        if phase in _NPZ_PHASES:
            data = {"arr": np.array([[1.0]])}
        else:
            data = {"key": "value"}
        path = mgr.complete_phase(phase, data)
        assert path.exists()

    @pytest.mark.parametrize(
        "bad_name",
        ["unknown", "Sampling", "RENDERING", "embed", "", "phase1"],
    )
    def test_invalid_phase_names_rejected(self, tmp_path, bad_name):
        """非法阶段名称应触发 ValueError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(ValueError, match="未知阶段名称"):
            mgr.complete_phase(bad_name, {"key": "value"})

    def test_load_invalid_phase_raises(self, tmp_path):
        """load_phase_data 对非法阶段名称应触发 ValueError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(ValueError, match="未知阶段名称"):
            mgr.load_phase_data("nonexistent_phase")

    # -- 3. 空数据的检查点写入和加载 (Req 1.1) --

    def test_empty_json_data_round_trip(self, tmp_path):
        """空字典应能正确写入和加载（JSON 阶段）。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("rendering", {})
        loaded = mgr.load_phase_data("rendering")
        assert loaded == {}

    def test_empty_npz_data_round_trip(self, tmp_path):
        """空字典应能正确写入和加载（NPZ 阶段）。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("sampling", {})
        loaded = mgr.load_phase_data("sampling")
        assert loaded == {} or len(loaded) == 0

    # -- 4. 旧版 checkpoint.json 向后兼容迁移 (Req 1.2) --

    def test_legacy_checkpoint_v2_migration(self, tmp_path):
        """检测到旧版 checkpoint.json (version 2) 时应自动标记 rendering 为已完成。"""
        legacy = {
            "version": 2,
            "samples": [
                {"id": "s1", "status": "completed"},
                {"id": "s2", "status": "completed"},
            ],
        }
        legacy_path = tmp_path / "checkpoint.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

        mgr = CheckpointManager(output_dir=tmp_path)
        completed = mgr.get_completed_phases()
        assert "rendering" in completed

        master = json.loads(mgr.master_path.read_text(encoding="utf-8"))
        rendering_info = master["phases"]["rendering"]
        assert rendering_info["status"] == "completed"
        assert rendering_info["output_count"] == 2

    def test_legacy_checkpoint_non_v2_ignored(self, tmp_path):
        """非 version 2 的旧版 checkpoint.json 不应触发迁移。"""
        legacy = {"version": 1, "data": "old"}
        legacy_path = tmp_path / "checkpoint.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

        mgr = CheckpointManager(output_dir=tmp_path)
        assert "rendering" not in mgr.get_completed_phases()

    def test_legacy_checkpoint_missing_no_error(self, tmp_path):
        """不存在旧版 checkpoint.json 时不应报错。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        assert mgr.get_completed_phases() == []

    # -- 5. 原子写入 (Req 1.4) --

    def test_atomic_write_no_tmp_residue(self, tmp_path):
        """complete_phase 完成后不应残留 .tmp 文件。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("validation", {"valid_ids": ["s1", "s2"]})

        tmp_files = list(mgr.checkpoint_dir.glob("*.tmp"))
        assert tmp_files == [], f"残留临时文件: {tmp_files}"

    def test_atomic_write_interrupted_preserves_old(self, tmp_path, monkeypatch):
        """模拟写入中断：os.replace 失败时，旧检查点应保持不变。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        # 先写入一个有效检查点
        mgr.complete_phase("rendering", {"original": True})

        # 模拟 os.replace 在阶段文件写入时失败
        import src.checkpoint_manager as ckpt_mod

        original_replace = os.replace

        def failing_replace(src, dst):
            if "rendering.json" in str(dst) and not str(src).endswith(".tmp"):
                # 不应到达这里，但以防万一
                return original_replace(src, dst)
            if "rendering.json" in str(dst):
                raise OSError("模拟磁盘故障")
            return original_replace(src, dst)

        monkeypatch.setattr(ckpt_mod.os, "replace", failing_replace)

        with pytest.raises(OSError, match="模拟磁盘故障"):
            mgr.complete_phase("rendering", {"corrupted": True})

        # 旧检查点应保持不变
        monkeypatch.setattr(ckpt_mod.os, "replace", original_replace)
        loaded = mgr.load_phase_data("rendering")
        assert loaded == {"original": True}

    # -- 6. 日志输出检查点文件大小 (Req 7.3) --

    def test_log_checkpoint_file_size(self, tmp_path, caplog):
        """complete_phase 应在日志中输出检查点文件大小。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with caplog.at_level(logging.INFO, logger="src.checkpoint_manager"):
            mgr.complete_phase("rendering", {"data": list(range(100))})

        # 检查日志中包含 bytes 信息
        assert any("bytes" in record.message for record in caplog.records), (
            f"日志中未找到文件大小信息。日志内容: {[r.message for r in caplog.records]}"
        )

    def test_get_checkpoint_size_returns_positive(self, tmp_path):
        """get_checkpoint_size 应返回正整数。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("rendering", {"data": [1, 2, 3]})
        size = mgr.get_checkpoint_size("rendering")
        assert isinstance(size, int)
        assert size > 0

    def test_get_checkpoint_size_missing_raises(self, tmp_path):
        """get_checkpoint_size 对不存在的检查点应抛出 FileNotFoundError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            mgr.get_checkpoint_size("rendering")


# ---------------------------------------------------------------------------
# Unit Tests: validate_resume_from & get_phases_to_execute (Task 2.1)
# ---------------------------------------------------------------------------


class TestValidateResumeFrom:
    """validate_resume_from 单元测试。

    Requirements: 2.1, 2.2, 2.3, 2.5
    """

    def test_first_phase_no_prerequisites(self, tmp_path):
        """从第一个阶段恢复时无需前置检查点，不应抛出异常。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        # sampling is the first phase — no prerequisites
        mgr.validate_resume_from("sampling")

    def test_raises_on_missing_prerequisite(self, tmp_path):
        """前置检查点缺失时应抛出 FileNotFoundError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        # rendering requires sampling checkpoint
        with pytest.raises(FileNotFoundError, match="sampling"):
            mgr.validate_resume_from("rendering")

    def test_passes_when_all_prerequisites_exist(self, tmp_path):
        """所有前置检查点存在时不应抛出异常。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("sampling", {"arr": np.array([[1.0]])})
        mgr.complete_phase("rendering", {"data": True})
        # preprocessing requires sampling + rendering
        mgr.validate_resume_from("preprocessing")

    def test_error_message_contains_file_path(self, tmp_path):
        """错误信息应包含缺失文件路径。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="sampling.npz"):
            mgr.validate_resume_from("rendering")

    def test_error_message_contains_phase_name(self, tmp_path):
        """错误信息应包含需要重新执行的阶段名称。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("sampling", {"arr": np.array([[1.0]])})
        # Skip rendering, try to resume from preprocessing
        with pytest.raises(FileNotFoundError, match="rendering"):
            mgr.validate_resume_from("preprocessing")

    def test_invalid_phase_name_raises_value_error(self, tmp_path):
        """非法阶段名称应抛出 ValueError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(ValueError, match="未知阶段名称"):
            mgr.validate_resume_from("nonexistent")

    def test_detects_corrupted_prerequisite(self, tmp_path):
        """前置检查点文件损坏时应抛出 FileNotFoundError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        # Write a corrupted sampling checkpoint
        corrupted_file = mgr.checkpoint_dir / "sampling.npz"
        corrupted_file.write_text("not a valid npz file")
        with pytest.raises(FileNotFoundError, match="sampling"):
            mgr.validate_resume_from("rendering")

    def test_last_phase_requires_all_predecessors(self, tmp_path):
        """恢复到最后一个阶段时需要所有前置检查点。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        # Complete all phases except the last
        for phase in PHASE_ORDER[:-1]:
            if phase in _NPZ_PHASES:
                data = {"arr": np.array([[1.0]])}
            else:
                data = {"ok": True}
            mgr.complete_phase(phase, data)
        # Should not raise
        mgr.validate_resume_from(PHASE_ORDER[-1])


class TestGetPhasesToExecute:
    """get_phases_to_execute 单元测试。

    Requirements: 2.1, 2.2, 2.5
    """

    def test_none_returns_all_phases(self, tmp_path):
        """resume_from=None 时应返回所有阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        result = mgr.get_phases_to_execute(None)
        assert result == PHASE_ORDER

    def test_first_phase_returns_all(self, tmp_path):
        """从第一个阶段恢复时应返回所有阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        result = mgr.get_phases_to_execute("sampling")
        assert result == PHASE_ORDER

    def test_middle_phase_returns_suffix(self, tmp_path):
        """从中间阶段恢复时应返回该阶段及之后的阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        result = mgr.get_phases_to_execute("embedding")
        expected = PHASE_ORDER[PHASE_ORDER.index("embedding"):]
        assert result == expected

    def test_last_phase_returns_single(self, tmp_path):
        """从最后一个阶段恢复时应只返回该阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        result = mgr.get_phases_to_execute(PHASE_ORDER[-1])
        assert result == [PHASE_ORDER[-1]]

    def test_invalid_phase_raises_value_error(self, tmp_path):
        """非法阶段名称应抛出 ValueError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(ValueError, match="未知阶段名称"):
            mgr.get_phases_to_execute("nonexistent")

    @pytest.mark.parametrize("phase", PHASE_ORDER)
    def test_all_valid_phases_accepted(self, tmp_path, phase):
        """所有合法阶段名称都应被接受。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        result = mgr.get_phases_to_execute(phase)
        assert result[0] == phase
        assert result[-1] == PHASE_ORDER[-1]

    def test_returns_new_list(self, tmp_path):
        """返回的列表应是新对象，修改不影响 PHASE_ORDER。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        result = mgr.get_phases_to_execute(None)
        result.clear()
        assert len(PHASE_ORDER) == 7


# ---------------------------------------------------------------------------
# Property 2: 阶段跳过与恢复正确性
# ---------------------------------------------------------------------------


class TestProperty2PhaseSkipAndResumeCorrectness:
    """Feature: robust-checkpoint-and-resampling, Property 2: 阶段跳过与恢复正确性

    **Validates: Requirements 2.1, 2.2**
    """

    @given(phase_name=phase_names_st)
    @settings(max_examples=100, deadline=None)
    def test_phases_to_execute_equals_suffix_from_phase(self, tmp_path_factory, phase_name):
        """get_phases_to_execute(phase_name) 应返回 PHASE_ORDER[idx:] 。

        **Validates: Requirements 2.1, 2.2**
        """
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        result = mgr.get_phases_to_execute(phase_name)
        idx = PHASE_ORDER.index(phase_name)
        expected = PHASE_ORDER[idx:]

        assert result == expected

    @settings(max_examples=1, deadline=None)
    @given(data=st.just(None))
    def test_resume_from_none_returns_all_phases(self, tmp_path_factory, data):
        """resume_from=None 时应返回全部阶段列表。

        **Validates: Requirements 2.1**
        """
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        result = mgr.get_phases_to_execute(None)

        assert result == list(PHASE_ORDER)


# ---------------------------------------------------------------------------
# Property 3: 缺失检查点错误报告
# ---------------------------------------------------------------------------

# Phases that have at least one prerequisite (exclude "sampling" which is first)
_phases_with_prereqs_st = st.sampled_from(PHASE_ORDER[1:])


class TestProperty3MissingCheckpointErrorReport:
    """Feature: robust-checkpoint-and-resampling, Property 3: 缺失检查点错误报告

    **Validates: Requirements 2.3**
    """

    @given(
        phase_name=_phases_with_prereqs_st,
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_prerequisite_raises_with_path_and_phase(
        self, tmp_path_factory, phase_name, data
    ):
        """删除前置检查点后，validate_resume_from() 应抛出 FileNotFoundError，
        错误信息包含缺失文件路径和需要重新执行的阶段名称。

        **Validates: Requirements 2.3**
        """
        tmp_path = tmp_path_factory.mktemp("ckpt")
        mgr = CheckpointManager(output_dir=tmp_path)

        # Complete all prerequisite phases for the chosen phase
        idx = PHASE_ORDER.index(phase_name)
        prereqs = PHASE_ORDER[:idx]

        for prereq in prereqs:
            if prereq in _NPZ_PHASES:
                mgr.complete_phase(prereq, {"arr": np.array([[1.0, 2.0]])})
            else:
                mgr.complete_phase(prereq, {"ok": True})

        # Pick a random prerequisite to delete
        victim = data.draw(st.sampled_from(prereqs), label="deleted_prereq")
        victim_file = mgr._phase_file(victim)
        assert victim_file.exists(), f"Prerequisite file should exist: {victim_file}"
        victim_file.unlink()

        # validate_resume_from should raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            mgr.validate_resume_from(phase_name)

        error_msg = str(exc_info.value)

        # The error message must contain the missing file path
        assert str(victim_file) in error_msg, (
            f"Error message should contain missing file path {victim_file!r}, "
            f"got: {error_msg!r}"
        )

        # The error message must contain the phase name that needs re-execution
        assert victim in error_msg, (
            f"Error message should contain phase name {victim!r}, "
            f"got: {error_msg!r}"
        )


# ---------------------------------------------------------------------------
# Unit Tests: Embedding 持久化方法 (Task 4.1)
# ---------------------------------------------------------------------------


class TestEmbeddingPersistence:
    """Embedding 持久化方法单元测试。

    Requirements: 3.1, 3.2, 3.3, 3.4
    """

    # -- save_embedding_batch --

    def test_save_single_batch(self, tmp_path):
        """单个批次保存后 partial 文件应存在且包含正确数据。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        ids = ["s1", "s2", "s3"]
        emb = np.random.randn(3, 1024).astype(np.float32)

        mgr.save_embedding_batch(ids, emb)

        partial = tmp_path / "checkpoints" / "embedding_partial.npz"
        assert partial.exists()

        data = np.load(partial, allow_pickle=False)
        np.testing.assert_array_equal(data["sample_ids"], np.array(ids, dtype=str))
        np.testing.assert_array_almost_equal(data["embeddings"], emb)

    def test_save_multiple_batches_accumulates(self, tmp_path):
        """多个批次保存后 partial 文件应包含所有累积数据。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        ids1 = ["a", "b"]
        emb1 = np.ones((2, 4), dtype=np.float32)
        ids2 = ["c", "d", "e"]
        emb2 = np.ones((3, 4), dtype=np.float32) * 2

        mgr.save_embedding_batch(ids1, emb1)
        mgr.save_embedding_batch(ids2, emb2)

        saved_ids, saved_emb = mgr.load_partial_embeddings()
        assert saved_ids == ["a", "b", "c", "d", "e"]
        assert saved_emb.shape == (5, 4)
        np.testing.assert_array_almost_equal(saved_emb[:2], emb1)
        np.testing.assert_array_almost_equal(saved_emb[2:], emb2)

    # -- finalize_embeddings --

    def test_finalize_creates_final_file(self, tmp_path):
        """finalize 后应生成 embedding.npz 且 partial 文件消失。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_embedding_batch(["x"], np.array([[1.0] * 8], dtype=np.float32))

        mgr.finalize_embeddings()

        final = tmp_path / "checkpoints" / "embedding.npz"
        partial = tmp_path / "checkpoints" / "embedding_partial.npz"
        assert final.exists()
        assert not partial.exists()

    def test_finalize_without_partial_raises(self, tmp_path):
        """没有 partial 文件时 finalize 应抛出 FileNotFoundError。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="embedding_partial.npz"):
            mgr.finalize_embeddings()

    def test_finalize_data_integrity(self, tmp_path):
        """finalize 后的 embedding.npz 应包含与 partial 相同的数据。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        ids = ["p1", "p2"]
        emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mgr.save_embedding_batch(ids, emb)

        mgr.finalize_embeddings()

        final = np.load(tmp_path / "checkpoints" / "embedding.npz", allow_pickle=False)
        assert list(final["sample_ids"].astype(str)) == ids
        np.testing.assert_array_almost_equal(final["embeddings"], emb)

    # -- load_partial_embeddings --

    def test_load_partial_no_file_returns_empty(self, tmp_path):
        """没有任何 embedding 文件时应返回空数据。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        ids, emb = mgr.load_partial_embeddings()
        assert ids == []
        assert emb.shape[0] == 0

    def test_load_partial_prefers_partial_over_final(self, tmp_path):
        """同时存在 partial 和 final 时应优先加载 partial。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        ckpt_dir = tmp_path / "checkpoints"

        # Save final with different data
        np.savez(
            ckpt_dir / "embedding.npz",
            sample_ids=np.array(["final_id"], dtype=str),
            embeddings=np.zeros((1, 4), dtype=np.float32),
        )
        # Save partial
        np.savez(
            ckpt_dir / "embedding_partial.npz",
            sample_ids=np.array(["partial_id"], dtype=str),
            embeddings=np.ones((1, 4), dtype=np.float32),
        )

        ids, emb = mgr.load_partial_embeddings()
        assert ids == ["partial_id"]

    def test_load_partial_falls_back_to_final(self, tmp_path):
        """只有 final 文件时应加载 final。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        ckpt_dir = tmp_path / "checkpoints"

        np.savez(
            ckpt_dir / "embedding.npz",
            sample_ids=np.array(["final_only"], dtype=str),
            embeddings=np.zeros((1, 4), dtype=np.float32),
        )

        ids, emb = mgr.load_partial_embeddings()
        assert ids == ["final_only"]

    # -- get_missing_sample_ids --

    def test_missing_ids_all_missing(self, tmp_path):
        """没有已保存数据时所有 ID 都应缺失。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        expected = ["a", "b", "c"]
        missing = mgr.get_missing_sample_ids(expected)
        assert missing == expected

    def test_missing_ids_none_missing(self, tmp_path):
        """所有 ID 都已保存时应返回空列表。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_embedding_batch(["a", "b"], np.zeros((2, 4), dtype=np.float32))
        missing = mgr.get_missing_sample_ids(["a", "b"])
        assert missing == []

    def test_missing_ids_partial_overlap(self, tmp_path):
        """部分 ID 已保存时应返回差集，保持顺序。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_embedding_batch(["a", "c"], np.zeros((2, 4), dtype=np.float32))
        missing = mgr.get_missing_sample_ids(["a", "b", "c", "d"])
        assert missing == ["b", "d"]

    def test_missing_ids_preserves_order(self, tmp_path):
        """返回的缺失 ID 应保持 expected_ids 中的顺序。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_embedding_batch(["x"], np.zeros((1, 4), dtype=np.float32))
        missing = mgr.get_missing_sample_ids(["d", "c", "b", "a"])
        assert missing == ["d", "c", "b", "a"]


# ---------------------------------------------------------------------------
# Property 5: Embedding 持久化往返一致性
# ---------------------------------------------------------------------------


class TestProperty5EmbeddingRoundTripConsistency:
    """Feature: robust-checkpoint-and-resampling, Property 5: Embedding 持久化往返一致性

    **Validates: Requirements 3.1**
    """

    @given(
        sample_ids=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P")),
                min_size=1,
                max_size=20,
            ),
            min_size=1,
            max_size=20,
            unique=True,
        ),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_save_load_round_trip(self, tmp_path_factory, sample_ids, data):
        """保存随机 sample_id 和 1024 维向量后加载应得到等价数据。

        **Validates: Requirements 3.1**
        """
        n = len(sample_ids)
        embeddings = data.draw(
            st.builds(
                lambda: np.random.default_rng().standard_normal((n, 1024)).astype(np.float32),
            ),
            label="embeddings",
        )

        tmp_path = tmp_path_factory.mktemp("emb_rt")
        mgr = CheckpointManager(output_dir=tmp_path)

        mgr.save_embedding_batch(sample_ids, embeddings)
        loaded_ids, loaded_emb = mgr.load_partial_embeddings()

        assert loaded_ids == sample_ids
        assert loaded_emb.shape == embeddings.shape
        np.testing.assert_array_equal(loaded_emb, embeddings)


# ---------------------------------------------------------------------------
# Property 6: Embedding 批次级增量保存
# ---------------------------------------------------------------------------


@st.composite
def batch_sequence(draw):
    """Generate 1-5 batches, each with 1-10 unique sample_ids and 1024-dim vectors."""
    n_batches = draw(st.integers(min_value=1, max_value=5))
    all_ids: list[str] = []
    batches: list[tuple[list[str], np.ndarray]] = []
    counter = 0
    for _ in range(n_batches):
        batch_size = draw(st.integers(min_value=1, max_value=10))
        ids = [f"sample_{counter + j}" for j in range(batch_size)]
        counter += batch_size
        emb = np.random.default_rng().standard_normal((batch_size, 1024)).astype(np.float32)
        batches.append((ids, emb))
        all_ids.extend(ids)
    return batches, all_ids


class TestProperty6EmbeddingIncrementalBatchSave:
    """Feature: robust-checkpoint-and-resampling, Property 6: Embedding 批次级增量保存

    **Validates: Requirements 3.3**
    """

    @given(data=batch_sequence())
    @settings(max_examples=100, deadline=None)
    def test_incremental_batches_accumulate_correctly(self, tmp_path_factory, data):
        """每个批次保存后磁盘文件应包含截至当前批次的所有数据，
        最终合并后包含完整数据。

        **Validates: Requirements 3.3**
        """
        batches, all_ids = data
        tmp_path = tmp_path_factory.mktemp("emb_inc")
        mgr = CheckpointManager(output_dir=tmp_path)

        cumulative_ids: list[str] = []
        cumulative_embs: list[np.ndarray] = []

        for batch_ids, batch_emb in batches:
            mgr.save_embedding_batch(batch_ids, batch_emb)
            cumulative_ids.extend(batch_ids)
            cumulative_embs.append(batch_emb)

            # Verify cumulative state after each batch
            loaded_ids, loaded_emb = mgr.load_partial_embeddings()
            assert loaded_ids == cumulative_ids
            expected_emb = np.concatenate(cumulative_embs, axis=0)
            np.testing.assert_array_equal(loaded_emb, expected_emb)

        # Finalize and verify complete data
        mgr.finalize_embeddings()
        loaded_ids, loaded_emb = mgr.load_partial_embeddings()
        assert loaded_ids == all_ids
        expected_all = np.concatenate([e for _, e in batches], axis=0)
        np.testing.assert_array_equal(loaded_emb, expected_all)


# ---------------------------------------------------------------------------
# Property 7: 缺失 Embedding 增量提取
# ---------------------------------------------------------------------------


@st.composite
def expected_and_saved_ids(draw):
    """Generate a set of expected IDs and a subset of saved IDs."""
    all_ids = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=15,
            ),
            min_size=1,
            max_size=30,
            unique=True,
        )
    )
    # Draw a random subset to be the "saved" IDs
    if len(all_ids) == 0:
        return all_ids, []
    n_saved = draw(st.integers(min_value=0, max_value=len(all_ids)))
    saved_indices = draw(
        st.lists(
            st.sampled_from(range(len(all_ids))),
            min_size=n_saved,
            max_size=n_saved,
            unique=True,
        )
    )
    saved_ids = [all_ids[i] for i in sorted(saved_indices)]
    return all_ids, saved_ids


class TestProperty7MissingEmbeddingExtraction:
    """Feature: robust-checkpoint-and-resampling, Property 7: 缺失 Embedding 增量提取

    **Validates: Requirements 3.4**
    """

    @given(data=expected_and_saved_ids())
    @settings(max_examples=100, deadline=None)
    def test_missing_ids_equals_set_difference(self, tmp_path_factory, data):
        """get_missing_sample_ids() 应返回期望集合与已有集合的差集，
        且保持 expected_ids 中的顺序。

        **Validates: Requirements 3.4**
        """
        expected_ids, saved_ids = data
        tmp_path = tmp_path_factory.mktemp("emb_miss")
        mgr = CheckpointManager(output_dir=tmp_path)

        # Save the subset if non-empty
        if saved_ids:
            dim = 4  # small dim for speed
            emb = np.zeros((len(saved_ids), dim), dtype=np.float32)
            mgr.save_embedding_batch(saved_ids, emb)

        missing = mgr.get_missing_sample_ids(expected_ids)

        saved_set = set(saved_ids)
        expected_missing = [sid for sid in expected_ids if sid not in saved_set]

        assert missing == expected_missing


# ---------------------------------------------------------------------------
# Property 8: 预处理统计信息往返一致性
# ---------------------------------------------------------------------------

# Strategy: generate a dict of sample_id -> preprocessing stats record
_preprocessing_record_st = st.fixed_dictionaries({
    "original_rms_db": st.floats(
        min_value=-120.0, max_value=0.0,
        allow_nan=False, allow_infinity=False,
    ),
    "clipping_ratio": st.floats(
        min_value=0.0, max_value=1.0,
        allow_nan=False, allow_infinity=False,
    ),
    "is_filtered": st.booleans(),
    "filter_reason": st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    "output_sample_rate": st.sampled_from([16000, 44100]),
    "resampled": st.booleans(),
})

_preprocessing_stats_st = st.dictionaries(
    keys=st.from_regex(r"preset_[0-9]{5}_C[345]_v(80|120)", fullmatch=True),
    values=_preprocessing_record_st,
    min_size=1,
    max_size=20,
)


class TestProperty8PreprocessingStatsRoundTrip:
    """Feature: robust-checkpoint-and-resampling, Property 8: 预处理统计信息往返一致性

    **Validates: Requirements 4.1, 4.3**
    """

    @given(stats=_preprocessing_stats_st)
    @settings(max_examples=100, deadline=None)
    def test_save_load_round_trip(self, tmp_path_factory, stats):
        """使用 Hypothesis 生成随机预处理结果记录，验证 JSON 保存后加载得到等价数据。

        **Validates: Requirements 4.1, 4.3**
        """
        tmp_path = tmp_path_factory.mktemp("preproc")
        mgr = CheckpointManager(output_dir=tmp_path)

        mgr.save_preprocessing_stats(stats)
        loaded = mgr.load_preprocessing_stats()

        assert loaded == stats

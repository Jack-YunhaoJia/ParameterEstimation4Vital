"""
集成测试：完整流水线断点恢复。

验证 CheckpointManager 在模拟流水线场景中的断点恢复能力，
包括阶段级恢复、Embedding 增量提取、预处理统计持久化等。

Feature: robust-checkpoint-and-resampling
Requirements: 1.1, 2.1, 2.2, 3.2, 3.4, 4.2
"""

from __future__ import annotations

import numpy as np
import pytest

from src.checkpoint_manager import CheckpointManager, PHASE_ORDER


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_phase_data(phase: str, seed: int = 0) -> dict:
    """为指定阶段生成确定性的测试数据。"""
    rng = np.random.default_rng(seed)
    if phase == "sampling":
        return {"params": rng.standard_normal((10, 45)).astype(np.float64)}
    elif phase == "embedding":
        return {
            "sample_ids": np.array([f"s_{i}" for i in range(10)], dtype=str),
            "embeddings": rng.standard_normal((10, 1024)).astype(np.float32),
        }
    elif phase == "rendering":
        return {"samples": [{"id": f"s_{i}", "status": "rendered"} for i in range(10)]}
    elif phase == "preprocessing":
        return {
            f"s_{i}": {
                "original_rms_db": float(rng.uniform(-60, 0)),
                "clipping_ratio": float(rng.uniform(0, 0.05)),
                "is_filtered": False,
                "filter_reason": None,
                "output_sample_rate": 16000,
                "resampled": True,
            }
            for i in range(10)
        }
    elif phase == "validation":
        return {"valid_ids": [f"s_{i}" for i in range(8)], "report": "ok"}
    elif phase == "saving":
        return {"hdf5_path": "/tmp/data.hdf5", "train_count": 6, "val_count": 2}
    elif phase == "analysis":
        return {"coverage": 0.95, "uniformity": 0.88}
    return {"data": True}


# ---------------------------------------------------------------------------
# 测试 1：完整流水线断点恢复模拟
# ---------------------------------------------------------------------------


class Test完整流水线断点恢复:
    """模拟流水线在不同阶段中断，验证恢复逻辑正确性。

    Requirements: 1.1, 2.1, 2.2
    """

    def test_中断后恢复检测正确的下一阶段(self, tmp_path):
        """完成前 3 个阶段后模拟中断，新实例应检测到下一阶段为 embedding。"""
        mgr1 = CheckpointManager(output_dir=tmp_path)

        # 完成阶段 1-3
        for phase in ["sampling", "rendering", "preprocessing"]:
            mgr1.complete_phase(phase, _make_phase_data(phase))

        # 模拟中断：创建新的 CheckpointManager 实例
        mgr2 = CheckpointManager(output_dir=tmp_path)

        assert mgr2.get_resume_phase() == "embedding"
        assert mgr2.get_completed_phases() == ["sampling", "rendering", "preprocessing"]

    def test_恢复后继续执行剩余阶段(self, tmp_path):
        """中断后恢复，完成剩余阶段，验证所有阶段数据完整。"""
        mgr1 = CheckpointManager(output_dir=tmp_path)

        # 第一轮：完成前 3 个阶段
        for phase in ["sampling", "rendering", "preprocessing"]:
            mgr1.complete_phase(phase, _make_phase_data(phase))

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        resume_phase = mgr2.get_resume_phase()
        assert resume_phase == "embedding"

        # 第二轮：从 embedding 开始完成剩余阶段
        remaining = mgr2.get_phases_to_execute(resume_phase)
        assert remaining == ["embedding", "validation", "saving", "analysis"]

        for phase in remaining:
            mgr2.complete_phase(phase, _make_phase_data(phase))

        # 验证所有阶段已完成
        assert mgr2.get_completed_phases() == list(PHASE_ORDER)
        assert mgr2.get_resume_phase() is None

    def test_多次中断恢复(self, tmp_path):
        """模拟多次中断和恢复，每次恢复后继续执行。"""
        # 第一轮：完成 sampling
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("sampling", _make_phase_data("sampling"))

        # 中断 1
        mgr = CheckpointManager(output_dir=tmp_path)
        assert mgr.get_resume_phase() == "rendering"
        mgr.complete_phase("rendering", _make_phase_data("rendering"))
        mgr.complete_phase("preprocessing", _make_phase_data("preprocessing"))

        # 中断 2
        mgr = CheckpointManager(output_dir=tmp_path)
        assert mgr.get_resume_phase() == "embedding"
        mgr.complete_phase("embedding", _make_phase_data("embedding"))

        # 中断 3
        mgr = CheckpointManager(output_dir=tmp_path)
        assert mgr.get_resume_phase() == "validation"

        for phase in ["validation", "saving", "analysis"]:
            mgr.complete_phase(phase, _make_phase_data(phase))

        assert mgr.get_resume_phase() is None

    def test_恢复后阶段数据与原始一致(self, tmp_path):
        """恢复后加载的前置阶段数据应与原始写入数据一致。"""
        mgr1 = CheckpointManager(output_dir=tmp_path)
        original_rendering_data = _make_phase_data("rendering", seed=42)
        mgr1.complete_phase("sampling", _make_phase_data("sampling", seed=42))
        mgr1.complete_phase("rendering", original_rendering_data)

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        loaded = mgr2.load_phase_data("rendering")
        assert loaded == original_rendering_data


# ---------------------------------------------------------------------------
# 测试 2：Embedding 增量恢复
# ---------------------------------------------------------------------------


class TestEmbedding增量恢复:
    """验证 Embedding 增量提取在中断后正确恢复。

    Requirements: 3.2, 3.4
    """

    def test_部分批次保存后恢复(self, tmp_path):
        """保存 3/5 批次后模拟中断，恢复后应能加载已有数据并识别缺失样本。"""
        rng = np.random.default_rng(123)
        all_ids = [f"sample_{i:04d}" for i in range(50)]
        all_emb = rng.standard_normal((50, 1024)).astype(np.float32)

        # 第一轮：保存前 3 个批次（每批 10 个样本）
        mgr1 = CheckpointManager(output_dir=tmp_path)
        for batch_idx in range(3):
            start = batch_idx * 10
            end = start + 10
            mgr1.save_embedding_batch(all_ids[start:end], all_emb[start:end])

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        saved_ids, saved_emb = mgr2.load_partial_embeddings()

        assert len(saved_ids) == 30
        assert saved_emb.shape == (30, 1024)
        np.testing.assert_array_almost_equal(saved_emb, all_emb[:30])

    def test_恢复后识别缺失样本(self, tmp_path):
        """恢复后 get_missing_sample_ids 应返回未保存的样本 ID。"""
        rng = np.random.default_rng(456)
        all_ids = [f"sample_{i:04d}" for i in range(50)]

        mgr1 = CheckpointManager(output_dir=tmp_path)
        # 保存前 30 个
        mgr1.save_embedding_batch(
            all_ids[:30],
            rng.standard_normal((30, 1024)).astype(np.float32),
        )

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        missing = mgr2.get_missing_sample_ids(all_ids)

        assert missing == all_ids[30:]
        assert len(missing) == 20

    def test_补全缺失批次后合并(self, tmp_path):
        """补全缺失批次并 finalize 后，最终文件应包含所有数据。"""
        rng = np.random.default_rng(789)
        all_ids = [f"sample_{i:04d}" for i in range(50)]
        all_emb = rng.standard_normal((50, 1024)).astype(np.float32)

        # 第一轮：保存前 3 个批次
        mgr1 = CheckpointManager(output_dir=tmp_path)
        for batch_idx in range(3):
            start = batch_idx * 10
            end = start + 10
            mgr1.save_embedding_batch(all_ids[start:end], all_emb[start:end])

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        missing = mgr2.get_missing_sample_ids(all_ids)
        assert len(missing) == 20

        # 补全剩余批次
        for batch_idx in range(3, 5):
            start = batch_idx * 10
            end = start + 10
            mgr2.save_embedding_batch(all_ids[start:end], all_emb[start:end])

        # 验证所有样本已保存
        assert mgr2.get_missing_sample_ids(all_ids) == []

        # Finalize
        mgr2.finalize_embeddings()

        # 验证最终文件
        final_ids, final_emb = mgr2.load_partial_embeddings()
        assert len(final_ids) == 50
        assert set(final_ids) == set(all_ids)
        np.testing.assert_array_almost_equal(final_emb, all_emb)


# ---------------------------------------------------------------------------
# 测试 3：预处理统计持久化与恢复
# ---------------------------------------------------------------------------


class Test预处理统计持久化恢复:
    """验证预处理统计在恢复后正确加载。

    Requirements: 4.2
    """

    def test_保存后恢复加载统计(self, tmp_path):
        """保存预处理统计后模拟中断，恢复后应能正确加载。"""
        stats = {
            f"sample_{i:04d}": {
                "original_rms_db": -25.0 + i,
                "clipping_ratio": 0.001 * i,
                "is_filtered": i % 5 == 0,
                "filter_reason": "silence" if i % 5 == 0 else None,
                "output_sample_rate": 16000,
                "resampled": True,
            }
            for i in range(20)
        }

        mgr1 = CheckpointManager(output_dir=tmp_path)
        mgr1.save_preprocessing_stats(stats)

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        loaded = mgr2.load_preprocessing_stats()

        assert loaded == stats
        assert len(loaded) == 20

    def test_统计可用于跳过已处理样本(self, tmp_path):
        """加载的统计信息可用于判断哪些样本已处理，跳过重复处理。"""
        stats = {
            "sample_0000": {
                "original_rms_db": -20.0,
                "clipping_ratio": 0.0,
                "is_filtered": False,
                "filter_reason": None,
                "output_sample_rate": 16000,
                "resampled": True,
            },
            "sample_0001": {
                "original_rms_db": -80.0,
                "clipping_ratio": 0.0,
                "is_filtered": True,
                "filter_reason": "silence",
                "output_sample_rate": 16000,
                "resampled": True,
            },
        }

        mgr1 = CheckpointManager(output_dir=tmp_path)
        mgr1.save_preprocessing_stats(stats)

        # 模拟中断 + 恢复
        mgr2 = CheckpointManager(output_dir=tmp_path)
        loaded = mgr2.load_preprocessing_stats()

        # 使用统计信息跳过已处理样本
        all_sample_ids = ["sample_0000", "sample_0001", "sample_0002"]
        already_processed = set(loaded.keys())
        to_process = [sid for sid in all_sample_ids if sid not in already_processed]

        assert to_process == ["sample_0002"]

    def test_统计字段完整性(self, tmp_path):
        """恢复后的统计记录应包含所有必需字段。"""
        stats = {
            "sample_0000": {
                "original_rms_db": -25.3,
                "clipping_ratio": 0.001,
                "is_filtered": False,
                "filter_reason": None,
                "output_sample_rate": 16000,
                "resampled": True,
            }
        }

        mgr1 = CheckpointManager(output_dir=tmp_path)
        mgr1.save_preprocessing_stats(stats)

        mgr2 = CheckpointManager(output_dir=tmp_path)
        loaded = mgr2.load_preprocessing_stats()

        record = loaded["sample_0000"]
        required_fields = [
            "original_rms_db", "clipping_ratio", "is_filtered",
            "filter_reason", "output_sample_rate", "resampled",
        ]
        for field in required_fields:
            assert field in record, f"缺少字段: {field}"


# ---------------------------------------------------------------------------
# 测试 4：validate_resume_from 集成验证
# ---------------------------------------------------------------------------


class Test恢复验证:
    """验证 validate_resume_from 在集成场景中的行为。

    Requirements: 2.1, 2.2
    """

    def test_已完成前置阶段时验证通过(self, tmp_path):
        """完成 sampling/rendering/preprocessing 后，从 embedding 恢复应通过验证。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        for phase in ["sampling", "rendering", "preprocessing"]:
            mgr.complete_phase(phase, _make_phase_data(phase))

        # 不应抛出异常
        mgr.validate_resume_from("embedding")

    def test_跳过中间阶段时验证失败(self, tmp_path):
        """完成 sampling/rendering/preprocessing 后，从 validation 恢复应失败（缺少 embedding）。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        for phase in ["sampling", "rendering", "preprocessing"]:
            mgr.complete_phase(phase, _make_phase_data(phase))

        with pytest.raises(FileNotFoundError) as exc_info:
            mgr.validate_resume_from("validation")

        error_msg = str(exc_info.value)
        assert "embedding" in error_msg

    def test_错误信息包含有用信息(self, tmp_path):
        """验证失败时错误信息应包含缺失阶段名称和文件路径。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.complete_phase("sampling", _make_phase_data("sampling"))
        # 跳过 rendering，尝试从 preprocessing 恢复

        with pytest.raises(FileNotFoundError) as exc_info:
            mgr.validate_resume_from("preprocessing")

        error_msg = str(exc_info.value)
        # 应包含缺失的阶段名称
        assert "rendering" in error_msg
        # 应包含文件路径信息
        assert "rendering.json" in error_msg


# ---------------------------------------------------------------------------
# 测试 5：get_phases_to_execute 集成验证
# ---------------------------------------------------------------------------


class Test阶段执行列表:
    """验证 get_phases_to_execute 在不同恢复点的行为。

    Requirements: 2.1, 2.2
    """

    def test_全部完成后无需执行(self, tmp_path):
        """所有阶段完成后，get_resume_phase 返回 None。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        for phase in PHASE_ORDER:
            mgr.complete_phase(phase, _make_phase_data(phase))

        assert mgr.get_resume_phase() is None

    def test_从embedding恢复返回正确列表(self, tmp_path):
        """完成前 3 个阶段后，从 embedding 恢复应返回后 4 个阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        for phase in ["sampling", "rendering", "preprocessing"]:
            mgr.complete_phase(phase, _make_phase_data(phase))

        phases = mgr.get_phases_to_execute("embedding")
        assert phases == ["embedding", "validation", "saving", "analysis"]

    def test_resume_from_none返回全部阶段(self, tmp_path):
        """resume_from=None 应返回全部 7 个阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        phases = mgr.get_phases_to_execute(None)
        assert phases == list(PHASE_ORDER)
        assert len(phases) == 7

    def test_从最后阶段恢复只返回一个(self, tmp_path):
        """从 analysis 恢复应只返回 ['analysis']。"""
        mgr = CheckpointManager(output_dir=tmp_path)
        phases = mgr.get_phases_to_execute("analysis")
        assert phases == ["analysis"]


# ---------------------------------------------------------------------------
# 测试 6：向后兼容性 — 旧版 checkpoint.json 自动迁移
# ---------------------------------------------------------------------------


class Test向后兼容性:
    """验证旧版 checkpoint.json（version 2）的自动迁移行为。

    Requirements: 1.2
    """

    @staticmethod
    def _write_legacy_checkpoint(output_dir, completed=100, total=200):
        """在 output_dir 下写入旧版 checkpoint.json（version 2）。"""
        import json

        legacy = {
            "version": 2,
            "completed_presets": completed,
            "total_presets": total,
            "timestamp": "2024-01-01T00:00:00",
        }
        path = output_dir / "checkpoint.json"
        path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        return path

    def test_旧版checkpoint自动迁移(self, tmp_path):
        """旧版 checkpoint.json（version 2）存在时，rendering 阶段应被标记为已完成。"""
        self._write_legacy_checkpoint(tmp_path)

        mgr = CheckpointManager(output_dir=tmp_path)

        completed = mgr.get_completed_phases()
        assert "rendering" in completed, (
            f"旧版 checkpoint.json 迁移后 rendering 应在已完成列表中，"
            f"实际: {completed}"
        )

    def test_迁移后从rendering之后恢复(self, tmp_path):
        """旧版 checkpoint 迁移后，get_phases_to_execute 从 preprocessing 开始应返回正确列表。"""
        self._write_legacy_checkpoint(tmp_path)

        mgr = CheckpointManager(output_dir=tmp_path)

        # rendering 已完成，从 preprocessing 开始执行
        phases = mgr.get_phases_to_execute("preprocessing")
        assert phases == [
            "preprocessing", "embedding", "validation", "saving", "analysis"
        ]

        # 验证 rendering 确实在已完成列表中
        assert "rendering" in mgr.get_completed_phases()

    def test_无旧版checkpoint时正常初始化(self, tmp_path):
        """没有旧版 checkpoint.json 时，CheckpointManager 应从零开始，无已完成阶段。"""
        mgr = CheckpointManager(output_dir=tmp_path)

        assert mgr.get_completed_phases() == []
        assert mgr.get_resume_phase() == "sampling"
        assert mgr.get_phases_to_execute(None) == list(PHASE_ORDER)

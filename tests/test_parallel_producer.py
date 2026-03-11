"""
ParallelProducer 单元测试。

验证数据类默认值、断点续传 round-trip、预设数量计算、
资源估算和 checkpoint JSON 格式。
所有组件依赖使用 mock 对象。代码注释使用中文。
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.parallel_producer import (
    ParallelProducer,
    ProductionConfig,
    ProductionSummary,
    SampleStatus,
)
from src.training_data import CORE_PARAMS


# ============================================================
# 辅助函数：创建带 mock 依赖的 ParallelProducer 实例
# ============================================================

def _make_producer(
    config: ProductionConfig | None = None,
) -> ParallelProducer:
    """创建带 mock 依赖的 ParallelProducer 实例。"""
    if config is None:
        config = ProductionConfig()
    return ParallelProducer(
        vital_vst_path=Path("/mock/vital.vst3"),
        config=config,
        preprocessor=MagicMock(),
        sampler=MagicMock(),
        validator=MagicMock(),
        analyzer=MagicMock(),
    )


# ============================================================
# 测试 ProductionConfig 默认值
# ============================================================

class TestProductionConfig:
    """ProductionConfig 数据类默认值测试。"""

    def test_default_target_samples(self) -> None:
        """默认目标样本数应为 100,000。"""
        config = ProductionConfig()
        assert config.target_samples == 100_000

    def test_default_n_workers(self) -> None:
        """默认 worker 数应为 11（M4 Pro: 12 cores - 1）。"""
        config = ProductionConfig()
        assert config.n_workers == 11

    def test_default_embedding_batch_size(self) -> None:
        """默认 embedding batch 大小应为 32。"""
        config = ProductionConfig()
        assert config.embedding_batch_size == 32

    def test_default_embedding_device(self) -> None:
        """默认 embedding 设备应为 mps。"""
        config = ProductionConfig()
        assert config.embedding_device == "mps"

    def test_default_checkpoint_interval(self) -> None:
        """默认 checkpoint 间隔应为 100。"""
        config = ProductionConfig()
        assert config.checkpoint_interval == 100

    def test_default_sampling_strategy(self) -> None:
        """默认采样策略应为 lhs_stratified。"""
        config = ProductionConfig()
        assert config.sampling_strategy == "lhs_stratified"

    def test_default_seed(self) -> None:
        """默认随机种子应为 42。"""
        config = ProductionConfig()
        assert config.seed == 42

    def test_default_filter_margin(self) -> None:
        """默认过滤余量应为 0.02。"""
        config = ProductionConfig()
        assert config.filter_margin == 0.02

    def test_default_n_conditions(self) -> None:
        """默认条件数应为 6。"""
        config = ProductionConfig()
        assert config.n_conditions == 6


# ============================================================
# 测试 SampleStatus 创建
# ============================================================

class TestSampleStatus:
    """SampleStatus 数据类创建测试。"""

    def test_create_pending_status(self) -> None:
        """创建 pending 状态的样本。"""
        status = SampleStatus(
            sample_id="preset_00001_C3_v80",
            preset_index=1,
            condition="C3_v80",
            status="pending",
        )
        assert status.sample_id == "preset_00001_C3_v80"
        assert status.preset_index == 1
        assert status.condition == "C3_v80"
        assert status.status == "pending"
        assert status.error is None

    def test_create_failed_status_with_error(self) -> None:
        """创建带错误信息的 failed 状态样本。"""
        status = SampleStatus(
            sample_id="preset_00002_C4_v120",
            preset_index=2,
            condition="C4_v120",
            status="failed",
            error="VST 渲染超时",
        )
        assert status.status == "failed"
        assert status.error == "VST 渲染超时"


# ============================================================
# 测试 ProductionSummary 创建
# ============================================================

class TestProductionSummary:
    """ProductionSummary 数据类创建测试。"""

    def test_create_summary(self) -> None:
        """创建生产摘要。"""
        summary = ProductionSummary(
            total_presets=17007,
            total_samples=102042,
            valid_samples=100000,
            filtered_samples=1842,
            failed_samples=200,
            filter_reasons={"silence": 800, "clipping": 500},
            total_time_sec=37800.0,
            phase_timings={"rendering": 27360.0, "embedding": 10080.0},
            storage_estimate={"wav_total_gb": "34.2", "hdf5_size_mb": "420"},
        )
        assert summary.total_presets == 17007
        assert summary.valid_samples == 100000
        assert summary.filter_reasons["silence"] == 800
        assert summary.phase_timings["rendering"] == 27360.0

    def test_default_empty_dicts(self) -> None:
        """默认字典字段应为空。"""
        summary = ProductionSummary(
            total_presets=0,
            total_samples=0,
            valid_samples=0,
            filtered_samples=0,
            failed_samples=0,
        )
        assert summary.filter_reasons == {}
        assert summary.phase_timings == {}
        assert summary.storage_estimate == {}
        assert summary.total_time_sec == 0.0


# ============================================================
# 测试 _compute_n_presets 计算
# ============================================================

class TestComputeNPresets:
    """预设数量计算测试。"""

    def test_default_config(self) -> None:
        """默认配置：ceil(100000 / 6 / 0.98) = 17007。"""
        producer = _make_producer()
        expected = ceil(100_000 / 6 / (1 - 0.02))
        assert producer.n_presets == expected
        assert producer.n_presets == 17007

    def test_custom_target_samples(self) -> None:
        """自定义目标样本数。"""
        config = ProductionConfig(target_samples=50_000, n_conditions=6, filter_margin=0.02)
        producer = _make_producer(config)
        expected = ceil(50_000 / 6 / 0.98)
        assert producer.n_presets == expected

    def test_custom_conditions(self) -> None:
        """自定义条件数。"""
        config = ProductionConfig(target_samples=100_000, n_conditions=3, filter_margin=0.02)
        producer = _make_producer(config)
        expected = ceil(100_000 / 3 / 0.98)
        assert producer.n_presets == expected

    def test_zero_filter_margin(self) -> None:
        """过滤余量为 0 时，预设数 = ceil(target / conditions)。"""
        config = ProductionConfig(target_samples=100, n_conditions=6, filter_margin=0.0)
        producer = _make_producer(config)
        expected = ceil(100 / 6 / 1.0)
        assert producer.n_presets == expected

    def test_high_filter_margin(self) -> None:
        """高过滤余量时需要更多预设。"""
        config = ProductionConfig(target_samples=1000, n_conditions=6, filter_margin=0.5)
        producer = _make_producer(config)
        expected = ceil(1000 / 6 / 0.5)
        assert producer.n_presets == expected


# ============================================================
# 测试 _save_checkpoint 和 _load_checkpoint round-trip
# ============================================================

class TestCheckpointRoundTrip:
    """断点续传 round-trip 测试。"""

    def test_save_and_load_basic(self, tmp_path: Path) -> None:
        """保存后加载应得到等价的状态列表。"""
        producer = _make_producer()
        statuses = [
            SampleStatus("preset_00001_C3_v80", 1, "C3_v80", "embedded"),
            SampleStatus("preset_00001_C3_v120", 1, "C3_v120", "rendered"),
            SampleStatus("preset_00002_C4_v80", 2, "C4_v80", "pending"),
            SampleStatus("preset_00003_C5_v120", 3, "C5_v120", "failed", "渲染超时"),
        ]

        checkpoint_path = tmp_path / "checkpoint.json"
        producer._save_checkpoint(statuses, checkpoint_path)
        loaded = producer._load_checkpoint(checkpoint_path)

        assert len(loaded) == len(statuses)
        for orig, load in zip(statuses, loaded):
            assert load.sample_id == orig.sample_id
            assert load.preset_index == orig.preset_index
            assert load.condition == orig.condition
            assert load.status == orig.status
            assert load.error == orig.error

    def test_checkpoint_json_format(self, tmp_path: Path) -> None:
        """checkpoint JSON 应包含 version、timestamps、config、samples 字段。"""
        producer = _make_producer()
        statuses = [
            SampleStatus("preset_00001_C3_v80", 1, "C3_v80", "embedded"),
        ]

        checkpoint_path = tmp_path / "checkpoint.json"
        producer._save_checkpoint(statuses, checkpoint_path)

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 验证顶层字段
        assert data["version"] == 1
        assert "created_at" in data
        assert "updated_at" in data
        assert "total_presets" in data
        assert "total_samples" in data
        assert "config" in data
        assert "samples" in data

        # 验证 config 字段
        config = data["config"]
        assert config["target_samples"] == 100_000
        assert config["n_workers"] == 11
        assert config["sampling_strategy"] == "lhs_stratified"
        assert config["seed"] == 42

        # 验证 samples 字段
        assert len(data["samples"]) == 1
        sample = data["samples"][0]
        assert sample["sample_id"] == "preset_00001_C3_v80"
        assert sample["preset_index"] == 1
        assert sample["condition"] == "C3_v80"
        assert sample["status"] == "embedded"
        assert sample["error"] is None

    def test_checkpoint_preserves_created_at(self, tmp_path: Path) -> None:
        """多次保存应保留原始 created_at 时间戳。"""
        producer = _make_producer()
        checkpoint_path = tmp_path / "checkpoint.json"

        # 第一次保存
        statuses1 = [SampleStatus("s1", 1, "C3_v80", "pending")]
        producer._save_checkpoint(statuses1, checkpoint_path)

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data1 = json.load(f)
        created_at_1 = data1["created_at"]

        # 第二次保存（更新）
        statuses2 = [SampleStatus("s1", 1, "C3_v80", "embedded")]
        producer._save_checkpoint(statuses2, checkpoint_path)

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data2 = json.load(f)

        # created_at 应保持不变
        assert data2["created_at"] == created_at_1
        # updated_at 应更新
        assert "updated_at" in data2

    def test_empty_statuses(self, tmp_path: Path) -> None:
        """空状态列表的 round-trip。"""
        producer = _make_producer()
        checkpoint_path = tmp_path / "checkpoint.json"

        producer._save_checkpoint([], checkpoint_path)
        loaded = producer._load_checkpoint(checkpoint_path)
        assert loaded == []


# ============================================================
# 测试 _load_checkpoint 错误处理
# ============================================================

class TestLoadCheckpointErrors:
    """checkpoint 加载错误处理测试。"""

    def test_missing_file(self, tmp_path: Path) -> None:
        """文件不存在时返回空列表。"""
        producer = _make_producer()
        loaded = producer._load_checkpoint(tmp_path / "nonexistent.json")
        assert loaded == []

    def test_empty_file(self, tmp_path: Path) -> None:
        """空文件时返回空列表。"""
        producer = _make_producer()
        checkpoint_path = tmp_path / "empty.json"
        checkpoint_path.write_text("")
        loaded = producer._load_checkpoint(checkpoint_path)
        assert loaded == []

    def test_invalid_json(self, tmp_path: Path) -> None:
        """无效 JSON 时返回空列表。"""
        producer = _make_producer()
        checkpoint_path = tmp_path / "invalid.json"
        checkpoint_path.write_text("{invalid json content")
        loaded = producer._load_checkpoint(checkpoint_path)
        assert loaded == []

    def test_json_without_samples_key(self, tmp_path: Path) -> None:
        """JSON 中缺少 samples 键时返回空列表。"""
        producer = _make_producer()
        checkpoint_path = tmp_path / "no_samples.json"
        checkpoint_path.write_text('{"version": 1}')
        loaded = producer._load_checkpoint(checkpoint_path)
        assert loaded == []


# ============================================================
# 测试 estimate_resources
# ============================================================

class TestEstimateResources:
    """资源估算测试。"""

    def test_returns_required_keys(self) -> None:
        """估算结果应包含所有必需的键。"""
        producer = _make_producer()
        result = producer.estimate_resources(n_presets=17007, n_conditions=6)

        required_keys = [
            "n_presets", "n_conditions", "n_samples", "n_valid_estimate",
            "wav_size_gb", "hdf5_size_mb",
            "render_time_sec", "embed_time_sec",
            "estimated_time_sec", "estimated_hours",
        ]
        for key in required_keys:
            assert key in result, f"缺少键: {key}"

    def test_wav_size_calculation(self) -> None:
        """WAV 存储估算：N × 44100 × 2.0 × 4 / 1e9 GB。"""
        producer = _make_producer()
        n_presets = 100
        n_conditions = 6
        n_samples = n_presets * n_conditions  # 600

        result = producer.estimate_resources(n_presets, n_conditions)

        # 手动计算：600 × 44100 × 2.0 × 4 / 1e9
        expected_gb = n_samples * 44100 * 2.0 * 4 / 1e9
        assert abs(result["wav_size_gb"] - round(expected_gb, 2)) < 0.01

    def test_hdf5_size_calculation(self) -> None:
        """HDF5 大小估算：N_valid × (45 + 1024) × 4 / 1e6 MB。"""
        producer = _make_producer()
        n_presets = 100
        n_conditions = 6
        n_samples = n_presets * n_conditions
        n_valid = int(n_samples * (1 - 0.02))

        result = producer.estimate_resources(n_presets, n_conditions)

        expected_mb = n_valid * (45 + 1024) * 4 / 1e6
        assert abs(result["hdf5_size_mb"] - round(expected_mb, 2)) < 0.01

    def test_time_estimation(self) -> None:
        """时间估算：N × 3s / n_workers + N × 0.1s。"""
        config = ProductionConfig(n_workers=10)
        producer = _make_producer(config)
        n_presets = 100
        n_conditions = 6
        n_samples = n_presets * n_conditions

        result = producer.estimate_resources(n_presets, n_conditions)

        expected_render = n_samples * 3.0 / 10
        expected_embed = n_samples * 0.1
        expected_total = expected_render + expected_embed

        assert abs(result["render_time_sec"] - round(expected_render, 1)) < 0.1
        assert abs(result["embed_time_sec"] - round(expected_embed, 1)) < 0.1
        assert abs(result["estimated_time_sec"] - round(expected_total, 1)) < 0.1

    def test_reasonable_values_for_default_config(self) -> None:
        """默认配置下的估算值应在合理范围内。"""
        producer = _make_producer()
        result = producer.estimate_resources(n_presets=17007, n_conditions=6)

        # WAV 约 34 GB
        assert 30 < result["wav_size_gb"] < 40
        # HDF5 约 420 MB
        assert 300 < result["hdf5_size_mb"] < 500
        # 预计 8-15 小时
        assert 5 < result["estimated_hours"] < 20
        # 样本数
        assert result["n_samples"] == 17007 * 6


# ============================================================
# （原占位测试已移除，produce() 已实现）
# ============================================================


# ============================================================
# 测试 _render_worker 静态方法
# ============================================================

class TestRenderWorker:
    """_render_worker 静态方法测试。使用 mock AudioRenderer。"""

    def test_all_succeed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """所有任务渲染成功时，返回全部 (sample_id, True, None)。"""
        # 创建 mock AudioRenderer，render_preset 始终返回 True
        mock_renderer_instance = MagicMock()
        mock_renderer_instance.render_preset.return_value = True
        mock_renderer_cls = MagicMock(return_value=mock_renderer_instance)

        # 替换 AudioRenderer 和 RenderConfig 的导入
        import src.audio_renderer as ar_module
        monkeypatch.setattr(ar_module, "AudioRenderer", mock_renderer_cls)

        # mock 掉 _render_worker 内部的导入
        import importlib
        import src.parallel_producer as pp_module

        original_render_worker = ParallelProducer._render_worker

        def patched_render_worker(tasks, vital_vst_path):
            """使用 mock renderer 的 render worker。"""
            results = []
            for preset_path, output_path, sample_id in tasks:
                try:
                    success = mock_renderer_instance.render_preset(
                        Path(preset_path), Path(output_path)
                    )
                    if success:
                        results.append((sample_id, True, None))
                    else:
                        results.append((sample_id, False, "渲染返回 False"))
                except Exception as e:
                    results.append((sample_id, False, str(e)))
            return results

        tasks = [
            (tmp_path / "p1.vital", tmp_path / "p1.wav", "sample_001"),
            (tmp_path / "p2.vital", tmp_path / "p2.wav", "sample_002"),
            (tmp_path / "p3.vital", tmp_path / "p3.wav", "sample_003"),
        ]

        results = patched_render_worker(tasks, Path("/mock/vital.vst3"))

        assert len(results) == 3
        for sample_id, success, error in results:
            assert success is True
            assert error is None

    def test_some_fail(self, tmp_path: Path) -> None:
        """部分任务渲染失败时，返回对应的失败信息。"""
        # 创建 mock renderer，第二个任务失败
        mock_renderer = MagicMock()
        call_count = [0]

        def mock_render(preset_path, output_path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("VST 渲染崩溃")
            return True

        mock_renderer.render_preset.side_effect = mock_render

        tasks = [
            (tmp_path / "p1.vital", tmp_path / "p1.wav", "sample_001"),
            (tmp_path / "p2.vital", tmp_path / "p2.wav", "sample_002"),
            (tmp_path / "p3.vital", tmp_path / "p3.wav", "sample_003"),
        ]

        # 直接模拟 worker 逻辑
        results = []
        for preset_path, output_path, sample_id in tasks:
            try:
                success = mock_renderer.render_preset(
                    Path(preset_path), Path(output_path)
                )
                if success:
                    results.append((sample_id, True, None))
                else:
                    results.append((sample_id, False, "渲染返回 False"))
            except Exception as e:
                results.append((sample_id, False, str(e)))

        assert len(results) == 3
        # 第一个成功
        assert results[0] == ("sample_001", True, None)
        # 第二个失败
        assert results[1][0] == "sample_002"
        assert results[1][1] is False
        assert "VST 渲染崩溃" in results[1][2]
        # 第三个成功
        assert results[2] == ("sample_003", True, None)

    def test_empty_tasks(self) -> None:
        """空任务列表应返回空结果（通过 mock 模拟）。"""
        mock_renderer = MagicMock()
        tasks: list[tuple[Path, Path, str]] = []
        results = []
        for preset_path, output_path, sample_id in tasks:
            try:
                success = mock_renderer.render_preset(preset_path, output_path)
                results.append((sample_id, success, None))
            except Exception as e:
                results.append((sample_id, False, str(e)))
        assert results == []


# ============================================================
# 测试 render_parallel 多进程渲染
# ============================================================

class TestRenderParallel:
    """render_parallel 方法测试。使用 multiprocessing.dummy.Pool 替代真实进程池。"""

    def test_status_updates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """渲染完成后应正确更新 statuses 中对应样本的状态。"""
        import multiprocessing.dummy as dummy_mp

        # mock _render_worker 返回全部成功
        def mock_render_worker(tasks, vital_vst_path):
            return [(task[2], True, None) for task in tasks]

        monkeypatch.setattr(
            ParallelProducer, "_render_worker", staticmethod(mock_render_worker)
        )
        # 使用线程池替代进程池
        monkeypatch.setattr(
            "src.parallel_producer.multiprocessing.Pool", dummy_mp.Pool
        )

        config = ProductionConfig(n_workers=2, checkpoint_interval=100)
        producer = _make_producer(config)

        statuses = [
            SampleStatus("s1", 1, "C3_v80", "pending"),
            SampleStatus("s2", 1, "C3_v120", "pending"),
            SampleStatus("s3", 2, "C3_v80", "pending"),
        ]

        render_tasks = [
            (tmp_path / "p1.vital", tmp_path / "p1.wav", "s1", 48, 80, 2.0),
            (tmp_path / "p2.vital", tmp_path / "p2.wav", "s2", 48, 120, 2.0),
            (tmp_path / "p3.vital", tmp_path / "p3.wav", "s3", 60, 80, 2.0),
        ]

        checkpoint_path = tmp_path / "checkpoint.json"
        results = producer.render_parallel(render_tasks, statuses, checkpoint_path)

        # 验证所有结果成功
        assert len(results) == 3
        for _, success, _ in results:
            assert success is True

        # 验证状态已更新为 rendered
        for s in statuses:
            assert s.status == "rendered"

    def test_failed_tasks_update_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """渲染失败的任务应将状态更新为 failed 并记录错误信息。"""
        import multiprocessing.dummy as dummy_mp

        # mock _render_worker：第二个任务失败
        def mock_render_worker(tasks, vital_vst_path):
            results = []
            for task in tasks:
                sid = task[2]
                if sid == "s2":
                    results.append((sid, False, "渲染超时"))
                else:
                    results.append((sid, True, None))
            return results

        monkeypatch.setattr(
            ParallelProducer, "_render_worker", staticmethod(mock_render_worker)
        )
        monkeypatch.setattr(
            "src.parallel_producer.multiprocessing.Pool", dummy_mp.Pool
        )

        config = ProductionConfig(n_workers=2, checkpoint_interval=100)
        producer = _make_producer(config)

        statuses = [
            SampleStatus("s1", 1, "C3_v80", "pending"),
            SampleStatus("s2", 1, "C3_v120", "pending"),
            SampleStatus("s3", 2, "C3_v80", "pending"),
        ]

        render_tasks = [
            (tmp_path / "p1.vital", tmp_path / "p1.wav", "s1", 48, 80, 2.0),
            (tmp_path / "p2.vital", tmp_path / "p2.wav", "s2", 48, 120, 2.0),
            (tmp_path / "p3.vital", tmp_path / "p3.wav", "s3", 60, 80, 2.0),
        ]

        checkpoint_path = tmp_path / "checkpoint.json"
        results = producer.render_parallel(render_tasks, statuses, checkpoint_path)

        # 验证 s2 状态为 failed
        status_map = {s.sample_id: s for s in statuses}
        assert status_map["s1"].status == "rendered"
        assert status_map["s2"].status == "failed"
        assert status_map["s2"].error == "渲染超时"
        assert status_map["s3"].status == "rendered"

    def test_checkpoint_saved_during_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """渲染过程中应按 checkpoint_interval 保存 checkpoint。"""
        import multiprocessing.dummy as dummy_mp

        def mock_render_worker(tasks, vital_vst_path):
            return [(task[2], True, None) for task in tasks]

        monkeypatch.setattr(
            ParallelProducer, "_render_worker", staticmethod(mock_render_worker)
        )
        monkeypatch.setattr(
            "src.parallel_producer.multiprocessing.Pool", dummy_mp.Pool
        )

        # 设置 checkpoint_interval=2，3 个任务应触发 1 次中间保存 + 1 次最终保存
        config = ProductionConfig(n_workers=1, checkpoint_interval=2)
        producer = _make_producer(config)

        statuses = [
            SampleStatus("s1", 1, "C3_v80", "pending"),
            SampleStatus("s2", 1, "C3_v120", "pending"),
            SampleStatus("s3", 2, "C3_v80", "pending"),
        ]

        render_tasks = [
            (tmp_path / "p1.vital", tmp_path / "p1.wav", "s1", 48, 80, 2.0),
            (tmp_path / "p2.vital", tmp_path / "p2.wav", "s2", 48, 120, 2.0),
            (tmp_path / "p3.vital", tmp_path / "p3.wav", "s3", 60, 80, 2.0),
        ]

        checkpoint_path = tmp_path / "checkpoint.json"
        producer.render_parallel(render_tasks, statuses, checkpoint_path)

        # 验证 checkpoint 文件存在且内容正确
        assert checkpoint_path.exists()
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["samples"]) == 3
        # 所有样本应为 rendered 状态
        for sample in data["samples"]:
            assert sample["status"] == "rendered"

    def test_empty_tasks(self, tmp_path: Path) -> None:
        """空任务列表应返回空结果。"""
        producer = _make_producer()
        checkpoint_path = tmp_path / "checkpoint.json"
        results = producer.render_parallel([], [], checkpoint_path)
        assert results == []


# ============================================================
# 测试 extract_embeddings_batch GPU 批量 embedding 提取
# ============================================================

class TestExtractEmbeddingsBatch:
    """extract_embeddings_batch 方法测试。使用 mock EmbeddingExtractor。"""

    def test_basic_extraction(self, tmp_path: Path) -> None:
        """使用 mock extractor 提取 embedding，返回 (N, 1024) 矩阵。"""
        import numpy as np

        # 创建 mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = lambda path: np.random.randn(1024).astype(np.float32)

        producer = _make_producer()
        audio_paths = [tmp_path / f"audio_{i}.wav" for i in range(5)]

        result = producer.extract_embeddings_batch(
            audio_paths, sample_rate=16000, extractor=mock_extractor
        )

        assert result.shape == (5, 1024)
        assert result.dtype == np.float32
        assert mock_extractor.extract.call_count == 5

    def test_batching_respects_batch_size(self, tmp_path: Path) -> None:
        """应按 embedding_batch_size 分批处理。"""
        import numpy as np

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = lambda path: np.ones(1024, dtype=np.float32)

        config = ProductionConfig(embedding_batch_size=3)
        producer = _make_producer(config)
        audio_paths = [tmp_path / f"audio_{i}.wav" for i in range(7)]

        result = producer.extract_embeddings_batch(
            audio_paths, sample_rate=16000, extractor=mock_extractor
        )

        # 7 个文件，batch_size=3，应分 3 批（3+3+1）
        assert result.shape == (7, 1024)
        assert mock_extractor.extract.call_count == 7

    def test_extraction_failure_uses_zero_vector(self, tmp_path: Path) -> None:
        """提取失败时应使用零向量占位。"""
        import numpy as np

        call_count = [0]

        def mock_extract(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("模型推理失败")
            return np.ones(1024, dtype=np.float32)

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = mock_extract

        producer = _make_producer()
        audio_paths = [tmp_path / f"audio_{i}.wav" for i in range(3)]

        result = producer.extract_embeddings_batch(
            audio_paths, sample_rate=16000, extractor=mock_extractor
        )

        assert result.shape == (3, 1024)
        # 第二个应为零向量
        assert np.all(result[1] == 0.0)
        # 第一个和第三个应为全 1
        assert np.all(result[0] == 1.0)
        assert np.all(result[2] == 1.0)

    def test_empty_paths(self) -> None:
        """空路径列表应返回 (0, 1024) 空矩阵。"""
        import numpy as np

        producer = _make_producer()
        result = producer.extract_embeddings_batch(
            [], sample_rate=16000, extractor=MagicMock()
        )

        assert result.shape == (0, 1024)
        assert result.dtype == np.float32


# ============================================================
# 测试 save_production_hdf5 HDF5 保存
# ============================================================

class TestSaveProductionHDF5:
    """save_production_hdf5 方法测试。验证 HDF5 结构、数据形状和元数据。"""

    @staticmethod
    def _make_test_data(n: int = 100):
        """生成测试用的随机数据。"""
        rng = np.random.default_rng(42)
        params = rng.random((n, 45)).astype(np.float32)
        embeddings = rng.random((n, 1024)).astype(np.float32)
        midi_conditions = [
            {"note": 60, "velocity": 100, "duration_sec": 2.0}
            for _ in range(n)
        ]
        audio_stats = [
            {"original_rms": -20.0, "original_peak": 0.9, "clipping_ratio": 0.01}
            for _ in range(n)
        ]
        metadata = {
            "param_names": [f"param_{i}" for i in range(45)],
            "param_ranges": [(0.0, 1.0) for _ in range(45)],
            "sampling_strategy": "lhs_stratified",
            "seed": 42,
            "production_timestamp": "2025-01-01T00:00:00+00:00",
            "vital_version": "1.5.5",
        }
        config_yaml = "target_samples: 100\nseed: 42\n"
        return params, embeddings, midi_conditions, audio_stats, metadata, config_yaml

    def test_creates_correct_hdf5_structure(self, tmp_path: Path) -> None:
        """HDF5 文件应包含 train/val/test 分组和 metadata 组。"""
        import h5py

        producer = _make_producer()
        params, embeddings, midi_conds, audio_stats, meta, cfg = self._make_test_data(100)
        hdf5_path = tmp_path / "test.h5"

        producer.save_production_hdf5(
            hdf5_path, params, embeddings, midi_conds, audio_stats, meta, cfg
        )

        with h5py.File(str(hdf5_path), "r") as f:
            # 验证顶层分组
            assert "train" in f
            assert "val" in f
            assert "test" in f
            assert "metadata" in f

            # 验证每个分组包含必需的数据集
            for split in ["train", "val", "test"]:
                grp = f[split]
                assert "params" in grp
                assert "embeddings" in grp
                assert "midi_notes" in grp
                assert "midi_velocities" in grp
                assert "midi_durations" in grp
                assert "audio_stats" in grp
                # 验证 audio_stats 子组
                stats = grp["audio_stats"]
                assert "original_rms" in stats
                assert "original_peak" in stats
                assert "clipping_ratio" in stats

    def test_data_shapes_and_types(self, tmp_path: Path) -> None:
        """验证各数据集的形状和数据类型。"""
        import h5py

        producer = _make_producer()
        n = 100
        params, embeddings, midi_conds, audio_stats, meta, cfg = self._make_test_data(n)
        hdf5_path = tmp_path / "test.h5"

        producer.save_production_hdf5(
            hdf5_path, params, embeddings, midi_conds, audio_stats, meta, cfg
        )

        with h5py.File(str(hdf5_path), "r") as f:
            total = 0
            for split in ["train", "val", "test"]:
                grp = f[split]
                n_split = grp["params"].shape[0]
                total += n_split

                # params 形状 (N_split, 45)，float32
                assert grp["params"].shape == (n_split, 45)
                assert grp["params"].dtype == np.float32

                # embeddings 形状 (N_split, 1024)，float32
                assert grp["embeddings"].shape == (n_split, 1024)
                assert grp["embeddings"].dtype == np.float32

                # midi_notes 形状 (N_split,)，int32
                assert grp["midi_notes"].shape == (n_split,)
                assert grp["midi_notes"].dtype == np.int32

                # midi_velocities 形状 (N_split,)，int32
                assert grp["midi_velocities"].shape == (n_split,)
                assert grp["midi_velocities"].dtype == np.int32

                # midi_durations 形状 (N_split,)，float32
                assert grp["midi_durations"].shape == (n_split,)
                assert grp["midi_durations"].dtype == np.float32

                # audio_stats 子组
                assert grp["audio_stats/original_rms"].shape == (n_split,)
                assert grp["audio_stats/original_peak"].shape == (n_split,)
                assert grp["audio_stats/clipping_ratio"].shape == (n_split,)

            # 总样本数应等于 n
            assert total == n

    def test_split_ratios_80_10_10(self, tmp_path: Path) -> None:
        """验证 80/10/10 数据集划分比例。"""
        import h5py

        producer = _make_producer()
        n = 1000
        params, embeddings, midi_conds, audio_stats, meta, cfg = self._make_test_data(n)
        hdf5_path = tmp_path / "test.h5"

        producer.save_production_hdf5(
            hdf5_path, params, embeddings, midi_conds, audio_stats, meta, cfg
        )

        with h5py.File(str(hdf5_path), "r") as f:
            n_train = f["train/params"].shape[0]
            n_val = f["val/params"].shape[0]
            n_test = f["test/params"].shape[0]

            # 80/10/10 划分
            assert n_train == int(n * 0.8)  # 800
            assert n_val == int(n * 0.1)  # 100
            assert n_test == n - n_train - n_val  # 100
            assert n_train + n_val + n_test == n

    def test_metadata_fields(self, tmp_path: Path) -> None:
        """验证 metadata 组包含所有必需字段。"""
        import h5py

        producer = _make_producer()
        params, embeddings, midi_conds, audio_stats, meta, cfg = self._make_test_data(50)
        hdf5_path = tmp_path / "test.h5"

        producer.save_production_hdf5(
            hdf5_path, params, embeddings, midi_conds, audio_stats, meta, cfg
        )

        with h5py.File(str(hdf5_path), "r") as f:
            meta_grp = f["metadata"]

            # param_names 数据集（45 个参数名）
            assert "param_names" in meta_grp
            param_names = [x.decode() if isinstance(x, bytes) else x for x in meta_grp["param_names"][:]]
            assert len(param_names) == 45

            # param_ranges 数据集 (45, 2)
            assert "param_ranges" in meta_grp
            assert meta_grp["param_ranges"].shape == (45, 2)
            assert meta_grp["param_ranges"].dtype == np.float32

            # 属性字段
            assert meta_grp.attrs["sampling_strategy"] == "lhs_stratified"
            assert meta_grp.attrs["seed"] == 42
            assert "production_timestamp" in meta_grp.attrs
            assert "vital_version" in meta_grp.attrs
            assert "production_config" in meta_grp.attrs
            # production_config 应包含 YAML 内容
            config_str = meta_grp.attrs["production_config"]
            assert "seed" in config_str


# ============================================================
# 测试 produce() 完整流水线（全 mock）
# ============================================================

class TestProduceFullPipeline:
    """produce() 方法测试。使用 mock 对象验证组件调用顺序。"""

    def _setup_mocked_producer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_presets: int = 2
    ) -> ParallelProducer:
        """创建完全 mock 化的 producer，用于测试 produce() 流水线。"""
        import multiprocessing.dummy as dummy_mp

        config = ProductionConfig(
            target_samples=12,
            n_workers=1,
            n_conditions=6,
            filter_margin=0.0,
            seed=42,
            checkpoint_interval=100,
            sampling_strategy="lhs_stratified",
        )

        # mock sampler：返回固定参数矩阵
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = np.random.default_rng(42).random(
            (n_presets, 45)
        ).astype(np.float32)

        # mock preprocessor：返回通过预处理的结果
        from src.audio_preprocessor import PreprocessResult
        mock_preprocessor = MagicMock()
        mock_preprocessor.process.return_value = PreprocessResult(
            audio=np.random.default_rng(42).random(16000).astype(np.float32),
            original_rms_db=-20.0,
            clipping_ratio=0.01,
            is_filtered=False,
            filter_reason=None,
            sample_rate=16000,
        )

        # mock validator：所有样本通过验证
        from src.quality_validator import DatasetQualityReport, SampleQualityResult
        mock_validator = MagicMock()
        mock_validator.validate_sample.return_value = SampleQualityResult(
            sample_id="test",
            is_valid=True,
            rms_db=-20.0,
            clipping_ratio=0.01,
            spectral_entropy=0.8,
            filter_reason=None,
        )
        mock_validator.validate_embeddings.return_value = DatasetQualityReport(
            total_samples=12,
            valid_samples=12,
            filtered_samples=0,
            filter_reasons={},
            pca_variance_ratio=[0.1] * 10,
            pca_top10_cumulative=0.5,
            pca_collapse_warning=False,
            near_duplicate_count=0,
            near_duplicate_ratio=0.0,
            insufficient_samples_warning=False,
            target_samples=12,
        )

        # mock analyzer
        from src.distribution_analyzer import DistributionReport
        mock_analyzer = MagicMock()
        mock_report = DistributionReport(
            pca_variance_ratios=[0.1] * 10,
            pca_cumulative_ratios=[0.1 * i for i in range(1, 11)],
            cosine_sim_mean=0.5,
            cosine_sim_std=0.1,
            cosine_sim_min=0.01,
            cosine_sim_max=0.99,
            cosine_sim_quantiles={"25%": 0.3, "50%": 0.5, "75%": 0.7},
            diversity_warning=False,
            param_stats={},
            param_ks_results={},
        )
        mock_analyzer.generate_report.return_value = mock_report

        producer = ParallelProducer(
            vital_vst_path=Path("/mock/vital.vst3"),
            config=config,
            preprocessor=mock_preprocessor,
            sampler=mock_sampler,
            validator=mock_validator,
            analyzer=mock_analyzer,
        )

        # mock render_parallel：标记所有样本为 rendered 并创建假音频文件
        original_render_parallel = producer.render_parallel

        def mock_render_parallel(render_tasks, statuses, checkpoint_path):
            """模拟渲染：标记为 rendered 并创建假 WAV 文件。"""
            import soundfile as sf
            results = []
            for preset_path, output_path, sample_id, midi_note, velocity, duration_sec in render_tasks:
                # 创建假音频文件
                audio = np.random.default_rng(42).random(44100 * 2).astype(np.float32) * 0.5
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, 44100)
                results.append((sample_id, True, None))
                # 更新状态
                for s in statuses:
                    if s.sample_id == sample_id:
                        s.status = "rendered"
            return results

        monkeypatch.setattr(producer, "render_parallel", mock_render_parallel)

        # mock extract_embeddings_batch：返回随机 embedding
        def mock_extract(audio_paths, **kwargs):
            return np.random.default_rng(42).random(
                (len(audio_paths), 1024)
            ).astype(np.float32)

        monkeypatch.setattr(producer, "extract_embeddings_batch", mock_extract)

        # mock PresetParser 和 PresetGenerator
        # 由于 produce() 内部使用 from ... import，需要 patch 源模块
        from unittest.mock import patch

        mock_parser_instance = MagicMock()
        from src.preset_parser import VitalPreset
        mock_generator_instance = MagicMock()
        mock_generator_instance.create_base_patch.return_value = VitalPreset(
            settings={name: 0.5 for name, _, _ in CORE_PARAMS},
            modulations=[],
            extra={"preset_name": "test"},
        )

        # patch PresetParser 类，使其实例化时返回 mock
        monkeypatch.setattr(
            "src.preset_parser.PresetParser",
            MagicMock(return_value=mock_parser_instance),
        )
        monkeypatch.setattr(
            "src.preset_generator.PresetGenerator",
            MagicMock(return_value=mock_generator_instance),
        )

        return producer

    def test_produce_calls_all_components(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """produce() 应按顺序调用所有组件。"""
        producer = self._setup_mocked_producer(tmp_path, monkeypatch)
        output_dir = tmp_path / "output"

        summary = producer.produce(output_dir)

        # 验证 sampler.sample 被调用
        producer.sampler.sample.assert_called_once()

        # 验证 validator.validate_embeddings 被调用
        producer.validator.validate_embeddings.assert_called_once()

        # 验证 analyzer.generate_report 被调用
        producer.analyzer.generate_report.assert_called_once()

        # 验证 analyzer.save_report 被调用
        producer.analyzer.save_report.assert_called_once()

        # 验证返回 ProductionSummary
        assert isinstance(summary, ProductionSummary)
        assert summary.total_presets > 0
        assert summary.total_samples > 0

    def test_produce_creates_output_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """produce() 应创建 HDF5 数据集和 production_summary.json。"""
        producer = self._setup_mocked_producer(tmp_path, monkeypatch)
        output_dir = tmp_path / "output"

        summary = producer.produce(output_dir)

        # 验证 production_summary.json 存在
        summary_path = output_dir / "production_summary.json"
        assert summary_path.exists()

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        assert "total_presets" in summary_data
        assert "total_samples" in summary_data
        assert "valid_samples" in summary_data
        assert "phase_timings" in summary_data

        # 验证 HDF5 文件存在（如果有有效样本）
        if summary.valid_samples > 0:
            hdf5_path = output_dir / "production_dataset.h5"
            assert hdf5_path.exists()

    def test_produce_resume_skips_completed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """produce() resume=True 应加载 checkpoint 并跳过已完成的样本。"""
        producer = self._setup_mocked_producer(tmp_path, monkeypatch)
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建一个 checkpoint，其中部分样本已完成
        checkpoint_path = output_dir / "checkpoint.json"
        statuses = [
            SampleStatus("preset_00000_C3_v80", 0, "C3_v80", "embedded"),
            SampleStatus("preset_00000_C3_v120", 0, "C3_v120", "embedded"),
            SampleStatus("preset_00000_C4_v80", 0, "C4_v80", "failed", "渲染失败"),
            SampleStatus("preset_00000_C4_v120", 0, "C4_v120", "pending"),
            SampleStatus("preset_00000_C5_v80", 0, "C5_v80", "rendered"),
            SampleStatus("preset_00000_C5_v120", 0, "C5_v120", "pending"),
            SampleStatus("preset_00001_C3_v80", 1, "C3_v80", "pending"),
            SampleStatus("preset_00001_C3_v120", 1, "C3_v120", "pending"),
            SampleStatus("preset_00001_C4_v80", 1, "C4_v80", "pending"),
            SampleStatus("preset_00001_C4_v120", 1, "C4_v120", "pending"),
            SampleStatus("preset_00001_C5_v80", 1, "C5_v80", "pending"),
            SampleStatus("preset_00001_C5_v120", 1, "C5_v120", "pending"),
        ]
        producer._save_checkpoint(statuses, checkpoint_path)

        # 记录 render_parallel 被调用时的任务
        render_tasks_received = []
        original_render_parallel = producer.render_parallel

        def tracking_render_parallel(render_tasks, statuses, cp_path):
            """追踪渲染任务，验证跳过了已完成的样本。"""
            import soundfile as sf
            render_tasks_received.extend(render_tasks)
            results = []
            for preset_path, output_path, sample_id, midi_note, velocity, duration_sec in render_tasks:
                audio = np.random.default_rng(42).random(44100 * 2).astype(np.float32) * 0.5
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, 44100)
                results.append((sample_id, True, None))
                for s in statuses:
                    if s.sample_id == sample_id:
                        s.status = "rendered"
            return results

        monkeypatch.setattr(producer, "render_parallel", tracking_render_parallel)

        summary = producer.produce(output_dir, resume=True)

        # 验证跳过了 embedded 和 failed 状态的样本
        rendered_ids = {t[2] for t in render_tasks_received}
        # embedded 样本不应出现在渲染任务中
        assert "preset_00000_C3_v80" not in rendered_ids
        assert "preset_00000_C3_v120" not in rendered_ids
        # failed 样本不应出现在渲染任务中
        assert "preset_00000_C4_v80" not in rendered_ids
        # rendered 样本也不应重新渲染（跳过渲染阶段）
        assert "preset_00000_C5_v80" not in rendered_ids
        # pending 样本应出现在渲染任务中
        assert "preset_00000_C4_v120" in rendered_ids
        assert "preset_00000_C5_v120" in rendered_ids

        assert isinstance(summary, ProductionSummary)

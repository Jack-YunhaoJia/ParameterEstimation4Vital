"""
并行生产模块。

协调多进程渲染、GPU 批量 embedding 提取、断点续传。
支持大规模数据集生产（100K+ 样本），提供资源估算和进度追踪。
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

import soundfile as sf

from src.audio_preprocessor import AudioPreprocessor
from src.batch_resampler import BatchResampler
from src.checkpoint_manager import CheckpointManager
from src.distribution_analyzer import DistributionAnalyzer
from src.multi_condition_renderer import DEFAULT_CONDITIONS, MidiCondition
from src.quality_validator import QualityValidator
from src.smart_sampler import SmartSampler
from src.training_data import CORE_PARAMS

logger = logging.getLogger(__name__)


def _inline_audio_filter(
    audio_path: Path,
    silence_threshold_db: float,
    clipping_threshold: float,
    clipping_ratio_limit: float,
) -> tuple[bool, str | None]:
    """对单个渲染音频执行静音/削波检测。

    复用 AudioPreprocessor 的静态方法，在渲染后立即判断音频是否有效。

    Args:
        audio_path: WAV 文件路径
        silence_threshold_db: 静音阈值 (dBFS)
        clipping_threshold: 削波检测绝对值阈值
        clipping_ratio_limit: 削波比例上限

    Returns:
        (is_valid, filter_reason) — True 表示通过, None 表示无过滤原因
    """
    try:
        audio_data, _sr = sf.read(str(audio_path), dtype="float32")
    except Exception:
        return False, "read_error"

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    rms_db = AudioPreprocessor.compute_rms_db(audio_data)
    if rms_db < silence_threshold_db:
        return False, "silence"

    clip_ratio = AudioPreprocessor.compute_clipping_ratio(audio_data, clipping_threshold)
    if clip_ratio > clipping_ratio_limit:
        return False, "clipping"

    return True, None


@dataclass
class ProductionConfig:
    """生产配置。"""

    target_samples: int = 100_000
    n_workers: int = 11  # M4 Pro: 12 cores - 1
    embedding_batch_size: int = 32
    embedding_device: str = "mps"  # MPS for M4 Pro
    checkpoint_interval: int = 100  # 每 N 个样本保存进度
    sampling_strategy: str = "lhs_stratified"
    seed: int = 42
    filter_margin: float = 0.02  # 2% 退化样本过滤余量
    n_conditions: int = 6  # 默认 6 个 MIDI 条件
    max_retries: int = 10  # 重试循环上限
    safety_margin: float = 0.02  # 动态过滤率安全余量
    resample_workers: int = 4  # 重采样并行线程数


@dataclass
class SampleStatus:
    """单个样本的处理状态。"""

    sample_id: str
    preset_index: int
    condition: str  # MIDI 条件标签
    status: str  # pending / rendered / render_passed / filtered / preprocessed / embedded / failed
    error: str | None = None


@dataclass
class RetryRoundStats:
    """单轮重试的统计信息。"""

    round_number: int
    n_new_presets: int
    n_rendered: int
    n_valid: int
    n_filtered: int
    filter_rate: float
    cumulative_valid: int


@dataclass
class ProductionSummary:
    """生产摘要。"""

    total_presets: int
    total_samples: int  # presets × conditions
    valid_samples: int
    filtered_samples: int
    failed_samples: int
    filter_reasons: dict[str, int] = field(default_factory=dict)
    total_time_sec: float = 0.0
    phase_timings: dict[str, float] = field(default_factory=dict)  # 阶段名 → 耗时
    storage_estimate: dict[str, str] = field(default_factory=dict)  # 存储项 → 大小估算
    retry_rounds: int = 0
    retry_stats: list[dict] = field(default_factory=list)


class ParallelProducer:
    """并行数据生产器。

    协调多进程渲染、GPU 批量 embedding 提取、断点续传。
    """

    def __init__(
        self,
        vital_vst_path: Path,
        config: ProductionConfig,
        preprocessor: AudioPreprocessor,
        sampler: SmartSampler,
        validator: QualityValidator,
        analyzer: DistributionAnalyzer,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        """初始化并行生产器。

        Args:
            vital_vst_path: Vital VST3 插件路径
            config: 生产配置
            preprocessor: 音频预处理器实例
            sampler: 智能采样器实例
            validator: 质量验证器实例
            analyzer: 分布分析器实例
            checkpoint_manager: 阶段级检查点管理器（可选）
        """
        self.vital_vst_path = Path(vital_vst_path)
        self.config = config
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.validator = validator
        self.analyzer = analyzer
        self.checkpoint_manager = checkpoint_manager
        # 计算所需预设数量
        self.n_presets = self._compute_n_presets()

    def _compute_n_presets(self) -> int:
        """计算所需预设数量。

        公式：ceil(target_samples / n_conditions / (1 - filter_margin))

        Returns:
            所需预设数量
        """
        return ceil(
            self.config.target_samples
            / self.config.n_conditions
            / (1 - self.config.filter_margin)
        )

    def _compute_effective_filter_rate(
        self, total_filtered: int, total_rendered: int
    ) -> float:
        """计算动态有效过滤率。

        Args:
            total_filtered: 累积过滤样本数
            total_rendered: 累积渲染样本数

        Returns:
            有效过滤率 = max(observed + safety_margin, filter_margin)
        """
        if total_rendered == 0:
            return self.config.filter_margin
        observed = total_filtered / total_rendered
        return max(observed + self.config.safety_margin, self.config.filter_margin)

    @staticmethod
    def _compute_retry_presets(
        deficit: int, n_conditions: int, effective_rate: float
    ) -> int:
        """计算补充预设数量。

        Args:
            deficit: 缺口样本数
            n_conditions: 每个预设的条件数
            effective_rate: 有效过滤率

        Returns:
            补充预设数量，至少为 1（当 deficit > 0 时）
        """
        if deficit <= 0:
            return 0
        n = ceil(deficit / n_conditions / (1 - effective_rate))
        return max(n, 1)

    @staticmethod
    def _cleanup_filtered_files(
        statuses: list[SampleStatus],
        presets_dir: Path,
        audio_dir: Path,
        n_conditions: int,
    ) -> None:
        """删除无效样本的音频文件和全条件失败的预设文件。

        Args:
            statuses: 样本状态列表
            presets_dir: 预设文件目录
            audio_dir: 音频文件目录
            n_conditions: 每个预设的条件数
        """
        # 删除 filtered 样本的 WAV 文件
        preset_filtered_count: dict[int, int] = {}
        for s in statuses:
            if s.status == "filtered":
                wav = audio_dir / f"{s.sample_id}.wav"
                try:
                    if wav.exists():
                        wav.unlink()
                except OSError as e:
                    logger.warning("删除 WAV 失败 %s: %s", wav, e)
                preset_filtered_count[s.preset_index] = (
                    preset_filtered_count.get(s.preset_index, 0) + 1
                )

        # 当某预设的所有条件均被过滤时，删除 .vital 文件
        for pi, count in preset_filtered_count.items():
            if count >= n_conditions:
                vital = presets_dir / f"preset_{pi:05d}.vital"
                try:
                    if vital.exists():
                        vital.unlink()
                except OSError as e:
                    logger.warning("删除预设失败 %s: %s", vital, e)

    @staticmethod
    def _lookup_params(
        all_params: list[np.ndarray], preset_index: int, initial_n_presets: int
    ) -> np.ndarray:
        """在多批次参数列表中查找指定预设的参数向量。

        Args:
            all_params: [initial_params, retry1_params, retry2_params, ...]
            preset_index: 全局预设索引
            initial_n_presets: 初始批次的预设数量

        Returns:
            参数向量 (45,)
        """
        if preset_index < initial_n_presets:
            return all_params[0][preset_index]
        # 在重试批次中查找
        offset = initial_n_presets
        for batch in all_params[1:]:
            if preset_index < offset + len(batch):
                return batch[preset_index - offset]
            offset += len(batch)
        raise IndexError(f"preset_index {preset_index} 超出所有参数批次范围")

    def _save_checkpoint(
        self,
        statuses: list[SampleStatus],
        path: Path,
        retry_state: dict | None = None,
    ) -> None:
        """保存进度状态文件（JSON）。

        Args:
            statuses: 样本状态列表
            path: 输出 JSON 文件路径
            retry_state: 重试循环状态（可选）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()

        created_at = now
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError):
                created_at = now

        checkpoint: dict[str, Any] = {
            "version": 2,
            "created_at": created_at,
            "updated_at": now,
            "total_presets": self.n_presets,
            "total_samples": self.n_presets * self.config.n_conditions,
            "config": {
                "target_samples": self.config.target_samples,
                "n_workers": self.config.n_workers,
                "sampling_strategy": self.config.sampling_strategy,
                "seed": self.config.seed,
            },
            "samples": [asdict(s) for s in statuses],
        }
        if retry_state is not None:
            checkpoint["retry_state"] = retry_state

        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        logger.info("Checkpoint 已保存至 %s（%d 个样本）", path, len(statuses))

    def _load_checkpoint(
        self, path: Path
    ) -> tuple[list[SampleStatus], dict | None]:
        """从 JSON 文件加载样本状态列表和重试状态。

        Args:
            path: checkpoint JSON 文件路径

        Returns:
            (样本状态列表, retry_state 或 None)。文件不存在或为空时返回 ([], None)。
        """
        path = Path(path)

        if not path.exists():
            logger.warning("Checkpoint 文件不存在: %s", path)
            return [], None

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning("Checkpoint 文件为空: %s", path)
                    return [], None
                data = json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Checkpoint 文件读取失败: %s, 错误: %s", path, e)
            return [], None

        samples = data.get("samples", [])
        statuses = []
        for s in samples:
            statuses.append(
                SampleStatus(
                    sample_id=s["sample_id"],
                    preset_index=s["preset_index"],
                    condition=s["condition"],
                    status=s["status"],
                    error=s.get("error"),
                )
            )

        retry_state = data.get("retry_state")

        logger.info("从 checkpoint 加载了 %d 个样本状态", len(statuses))
        return statuses, retry_state

    def estimate_resources(
        self, n_presets: int, n_conditions: int
    ) -> dict:
        """启动前输出存储和时间估算。

        计算公式：
        - WAV 存储：N_samples × sample_rate × duration × 4 bytes / 1e9 GB
        - HDF5 大小：N_valid × (45 + 1024) × 4 / 1e6 MB
        - 时间估算：N_samples × 3s / n_workers + N_samples × 0.1s

        Args:
            n_presets: 预设数量
            n_conditions: 每个预设的条件数

        Returns:
            包含 wav_size_gb, hdf5_size_mb, estimated_time_sec,
            estimated_hours 等的字典
        """
        n_samples = n_presets * n_conditions
        # 默认渲染参数：44100 Hz, 2 秒, 32-bit float (4 bytes)
        sample_rate = 44100
        duration_sec = 2.0

        # WAV 存储估算
        wav_size_gb = n_samples * sample_rate * duration_sec * 4 / 1e9

        # HDF5 大小估算（参数 45 维 + embedding 1024 维，float32）
        n_valid = int(n_samples * (1 - self.config.filter_margin))
        hdf5_size_mb = n_valid * (45 + 1024) * 4 / 1e6

        # 时间估算：渲染（3s/样本，多进程并行）+ embedding 提取（0.1s/样本）
        render_time_sec = n_samples * 3.0 / self.config.n_workers
        embed_time_sec = n_samples * 0.1
        estimated_time_sec = render_time_sec + embed_time_sec
        estimated_hours = estimated_time_sec / 3600

        estimate = {
            "n_presets": n_presets,
            "n_conditions": n_conditions,
            "n_samples": n_samples,
            "n_valid_estimate": n_valid,
            "wav_size_gb": round(wav_size_gb, 2),
            "hdf5_size_mb": round(hdf5_size_mb, 2),
            "render_time_sec": round(render_time_sec, 1),
            "embed_time_sec": round(embed_time_sec, 1),
            "estimated_time_sec": round(estimated_time_sec, 1),
            "estimated_hours": round(estimated_hours, 2),
        }

        logger.info(
            "资源估算: %d 样本, WAV %.2f GB, HDF5 %.2f MB, 预计 %.2f 小时",
            n_samples,
            wav_size_gb,
            hdf5_size_mb,
            estimated_hours,
        )

        return estimate

    def produce(
        self, output_dir: Path, resume: bool = False, resume_from: str | None = None,
    ) -> ProductionSummary:
        """执行完整生产流水线（渲染-过滤-重试循环架构）。

        流程：
        1. 参数采样（SmartSampler）
        2. 渲染-过滤-重试循环：预设生成 + 渲染 + 即时过滤 + 补充重试
        3. 音频预处理（仅对 render_passed 样本执行 DC/归一化/重采样/裁剪）
        4. Embedding 提取（GPU 批量）
        5. 质量验证
        6. 数据集保存（HDF5）
        7. 分布分析

        Args:
            output_dir: 输出目录
            resume: 是否从断点恢复
            resume_from: 指定恢复阶段名称（可选）

        Returns:
            ProductionSummary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / "checkpoint.json"
        presets_dir = output_dir / "presets"
        audio_dir = output_dir / "audio"
        presets_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 CheckpointManager（如果未通过 __init__ 提供）
        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(output_dir)

        # 确定需要执行的阶段
        phases_to_execute = None
        if resume_from is not None:
            self.checkpoint_manager.validate_resume_from(resume_from)
            phases_to_execute = set(self.checkpoint_manager.get_phases_to_execute(resume_from))
        elif resume and resume_from is None:
            auto_phase = self.checkpoint_manager.get_resume_phase()
            if auto_phase is not None:
                phases_to_execute = set(self.checkpoint_manager.get_phases_to_execute(auto_phase))

        def _should_execute(phase_name: str) -> bool:
            """判断是否需要执行指定阶段。"""
            if phases_to_execute is None:
                return True
            return phase_name in phases_to_execute

        phase_timings: dict[str, float] = {}
        total_start = time.time()

        target_samples = self.config.target_samples
        n_presets = self.n_presets
        n_conditions = self.config.n_conditions
        conditions = DEFAULT_CONDITIONS[:n_conditions]

        from src.preset_generator import PresetGenerator
        from src.preset_parser import PresetParser

        parser = PresetParser()
        generator = PresetGenerator(parser)
        cond_lookup = {c.label: c for c in conditions}

        # 过滤阈值（复用 AudioPreprocessor 默认值）
        from src.audio_preprocessor import PreprocessConfig
        _pp_cfg = PreprocessConfig()
        silence_threshold_db = _pp_cfg.silence_threshold_db
        clipping_threshold = _pp_cfg.clipping_threshold
        clipping_ratio_limit = _pp_cfg.clipping_ratio_limit

        # 多批次参数列表：[initial_params, retry1_params, ...]
        all_params_batches: list[np.ndarray] = []
        retry_stats_list: list[RetryRoundStats] = []
        filter_reasons: dict[str, int] = {}

        # ============================================================
        # 阶段 1：参数采样
        # ============================================================
        t0 = time.time()
        if _should_execute("sampling"):
            logger.info("阶段 1：参数采样（%d 个预设）", n_presets)
            params = self.sampler.sample(
                n_presets, strategy=self.config.sampling_strategy
            )
            all_params_batches.append(params)
            phase_timings["sampling"] = time.time() - t0
            logger.info("参数采样完成，耗时 %.1f 秒", phase_timings["sampling"])

            try:
                if self.checkpoint_manager:
                    self.checkpoint_manager.complete_phase("sampling", {"params": params})
            except Exception as e:
                logger.warning("sampling 检查点保存失败: %s", e)
        else:
            # 从检查点加载采样数据
            logger.info("阶段 1：跳过参数采样（从检查点加载）")
            try:
                sampling_data = self.checkpoint_manager.load_phase_data("sampling")
                params = sampling_data["params"]
                all_params_batches.append(params)
                n_presets = len(params)
                self.n_presets = n_presets
            except Exception as e:
                logger.error("加载 sampling 检查点失败: %s", e)
                raise
            phase_timings["sampling"] = time.time() - t0

        # ============================================================
        # 构建样本状态列表 / 断点恢复
        # ============================================================
        retry_round = 0
        next_preset_index = n_presets
        cumulative_filtered = 0
        cumulative_rendered = 0

        if resume and checkpoint_path.exists():
            statuses, saved_retry_state = self._load_checkpoint(checkpoint_path)
            logger.info("从断点恢复，加载 %d 个样本状态", len(statuses))
            if saved_retry_state:
                retry_round = saved_retry_state.get("current_round", 0)
                next_preset_index = saved_retry_state.get("next_preset_index", n_presets)
                cumulative_filtered = saved_retry_state.get("cumulative_filtered", 0)
                cumulative_rendered = saved_retry_state.get("cumulative_rendered", 0)
        else:
            statuses = []
            for pi in range(n_presets):
                for cond in conditions:
                    sid = f"preset_{pi:05d}_{cond.label}"
                    statuses.append(
                        SampleStatus(
                            sample_id=sid,
                            preset_index=pi,
                            condition=cond.label,
                            status="pending",
                        )
                    )

        # ============================================================
        # 阶段 2：渲染-过滤-重试循环
        # ============================================================
        t0 = time.time()
        if _should_execute("rendering"):
            logger.info("阶段 2：渲染-过滤-重试循环")

            def _build_render_tasks(batch_statuses):
                """为 pending 状态的样本构建渲染任务。"""
                tasks = []
                for s in batch_statuses:
                    if s.status not in ("pending",):
                        continue
                    preset_path = presets_dir / f"preset_{s.preset_index:05d}.vital"
                    if not preset_path.exists():
                        preset = generator.create_base_patch()
                        p_row = self._lookup_params(
                            all_params_batches, s.preset_index, n_presets
                        )
                        for col, (pname, _, _) in enumerate(CORE_PARAMS):
                            if pname in preset.settings:
                                preset.settings[pname] = float(p_row[col])
                        parser.serialize(preset, preset_path)
                    cond = cond_lookup.get(s.condition)
                    midi_note = cond.note if cond else 60
                    velocity = cond.velocity if cond else 100
                    duration_sec = cond.duration_sec if cond else 2.0
                    audio_path = audio_dir / f"{s.sample_id}.wav"
                    tasks.append((preset_path, audio_path, s.sample_id,
                                  midi_note, velocity, duration_sec))
                return tasks

            def _filter_rendered(all_statuses):
                """对 rendered 状态样本执行即时音频过滤，返回 (n_valid, n_silence, n_clipping)。"""
                n_valid = n_silence = n_clipping = n_err = 0
                for s in all_statuses:
                    if s.status != "rendered":
                        continue
                    ap = audio_dir / f"{s.sample_id}.wav"
                    ok, reason = _inline_audio_filter(
                        ap, silence_threshold_db, clipping_threshold, clipping_ratio_limit
                    )
                    if ok:
                        s.status = "render_passed"
                        n_valid += 1
                    else:
                        s.status = "filtered"
                        s.error = reason
                        if reason == "silence":
                            n_silence += 1
                        elif reason == "clipping":
                            n_clipping += 1
                        else:
                            n_err += 1
                        filter_reasons[reason or "unknown"] = (
                            filter_reasons.get(reason or "unknown", 0) + 1
                        )
                return n_valid, n_silence, n_clipping + n_err

            # --- Round 0：初始批次 ---
            render_tasks = _build_render_tasks(statuses)
            if render_tasks:
                self.render_parallel(render_tasks, statuses, checkpoint_path)

            round_valid, round_silence, round_clip = _filter_rendered(statuses)
            round_rendered = round_valid + round_silence + round_clip
            cumulative_rendered += round_rendered
            cumulative_filtered += round_silence + round_clip

            self._cleanup_filtered_files(statuses, presets_dir, audio_dir, n_conditions)
            self._save_checkpoint(statuses, checkpoint_path)

            valid_count = sum(1 for s in statuses if s.status in ("render_passed", "preprocessed", "embedded"))
            logger.info(
                "Round 0 完成：渲染 %d，有效 %d，静音 %d，削波 %d，累计有效 %d / %d",
                round_rendered, round_valid, round_silence, round_clip,
                valid_count, target_samples,
            )

            retry_stats_list.append(RetryRoundStats(
                round_number=0,
                n_new_presets=n_presets,
                n_rendered=round_rendered,
                n_valid=round_valid,
                n_filtered=round_silence + round_clip,
                filter_rate=((round_silence + round_clip) / round_rendered) if round_rendered > 0 else 0.0,
                cumulative_valid=valid_count,
            ))

            # --- 重试循环 Round 1 ~ max_retries ---
            consecutive_zero = 0
            while valid_count < target_samples and retry_round < self.config.max_retries:
                retry_round += 1
                deficit = target_samples - valid_count
                eff_rate = self._compute_effective_filter_rate(cumulative_filtered, cumulative_rendered)
                n_retry_presets = self._compute_retry_presets(deficit, n_conditions, eff_rate)

                logger.info(
                    "Round %d 开始：缺口 %d，有效过滤率 %.2f%%，补充 %d 个预设",
                    retry_round, deficit, eff_rate * 100, n_retry_presets,
                )

                # 采样新参数（临时修改 seed 以获得不同样本）
                original_seed = self.sampler.seed
                self.sampler.seed = self.config.seed + retry_round
                retry_params = self.sampler.sample(
                    n_retry_presets,
                    strategy=self.config.sampling_strategy,
                )
                self.sampler.seed = original_seed
                all_params_batches.append(retry_params)

                # 创建新样本状态
                new_statuses = []
                for ri in range(n_retry_presets):
                    pi = next_preset_index + ri
                    for cond in conditions:
                        sid = f"preset_{pi:05d}_{cond.label}"
                        new_s = SampleStatus(
                            sample_id=sid,
                            preset_index=pi,
                            condition=cond.label,
                            status="pending",
                        )
                        new_statuses.append(new_s)
                        statuses.append(new_s)
                next_preset_index += n_retry_presets

                # 渲染 + 过滤
                retry_render_tasks = _build_render_tasks(new_statuses)
                if retry_render_tasks:
                    self.render_parallel(retry_render_tasks, statuses, checkpoint_path)

                r_valid, r_silence, r_clip = _filter_rendered(statuses)
                r_rendered = r_valid + r_silence + r_clip
                cumulative_rendered += r_rendered
                cumulative_filtered += r_silence + r_clip

                self._cleanup_filtered_files(statuses, presets_dir, audio_dir, n_conditions)

                # 保存 checkpoint（含 retry_state）
                self._save_checkpoint(statuses, checkpoint_path, retry_state={
                    "current_round": retry_round,
                    "next_preset_index": next_preset_index,
                    "cumulative_filtered": cumulative_filtered,
                    "cumulative_rendered": cumulative_rendered,
                })

                valid_count = sum(1 for s in statuses if s.status in ("render_passed", "preprocessed", "embedded"))

                retry_stats_list.append(RetryRoundStats(
                    round_number=retry_round,
                    n_new_presets=n_retry_presets,
                    n_rendered=r_rendered,
                    n_valid=r_valid,
                    n_filtered=r_silence + r_clip,
                    filter_rate=((r_silence + r_clip) / r_rendered) if r_rendered > 0 else 0.0,
                    cumulative_valid=valid_count,
                ))

                logger.info(
                    "Round %d 完成：渲染 %d，有效 %d，过滤 %d，累计有效 %d / %d",
                    retry_round, r_rendered, r_valid, r_silence + r_clip,
                    valid_count, target_samples,
                )

                # 连续 3 轮零新增有效样本时提前终止
                if r_valid == 0:
                    consecutive_zero += 1
                    if consecutive_zero >= 3:
                        logger.warning("连续 %d 轮零新增有效样本，提前终止重试", consecutive_zero)
                        break
                else:
                    consecutive_zero = 0

            phase_timings["rendering"] = time.time() - t0
            logger.info("渲染-过滤-重试循环完成，耗时 %.1f 秒，共 %d 轮", phase_timings["rendering"], retry_round + 1)

            try:
                if self.checkpoint_manager:
                    self.checkpoint_manager.complete_phase("rendering", {
                        "statuses": [asdict(s) for s in statuses],
                        "retry_state": {
                            "current_round": retry_round,
                            "next_preset_index": next_preset_index,
                            "cumulative_filtered": cumulative_filtered,
                            "cumulative_rendered": cumulative_rendered,
                        },
                    })
            except Exception as e:
                logger.warning("rendering 检查点保存失败: %s", e)
        else:
            # 从检查点加载渲染数据
            logger.info("阶段 2：跳过渲染（从检查点加载）")
            try:
                rendering_data = self.checkpoint_manager.load_phase_data("rendering")
                statuses = []
                for s in rendering_data.get("statuses", []):
                    statuses.append(
                        SampleStatus(
                            sample_id=s["sample_id"],
                            preset_index=s["preset_index"],
                            condition=s["condition"],
                            status=s["status"],
                            error=s.get("error"),
                        )
                    )
                saved_retry = rendering_data.get("retry_state", {})
                retry_round = saved_retry.get("current_round", 0)
                next_preset_index = saved_retry.get("next_preset_index", n_presets)
                cumulative_filtered = saved_retry.get("cumulative_filtered", 0)
                cumulative_rendered = saved_retry.get("cumulative_rendered", 0)
            except Exception as e:
                logger.error("加载 rendering 检查点失败: %s", e)
                raise
            phase_timings["rendering"] = time.time() - t0

        # ============================================================
        # 阶段 3：音频预处理（仅 render_passed 样本，不再执行过滤判断）
        # ============================================================
        t0 = time.time()
        preprocess_results: dict[str, Any] = {}
        if _should_execute("preprocessing"):
            logger.info("阶段 3：音频预处理")

            for s in statuses:
                if s.status != "render_passed":
                    continue
                audio_path = audio_dir / f"{s.sample_id}.wav"
                if not audio_path.exists():
                    s.status = "failed"
                    s.error = "音频文件不存在"
                    continue
                try:
                    audio_data, sr = sf.read(str(audio_path), dtype="float32")
                    if audio_data.ndim > 1:
                        audio_data = audio_data[:, 0]
                    result = self.preprocessor.process(audio_data, sr)
                    preprocess_results[s.sample_id] = result
                    # 预处理阶段不再过滤（已在渲染后过滤），直接标记为 preprocessed
                    if result.is_filtered:
                        # 极少数情况：预处理器可能因其他原因过滤
                        s.status = "failed"
                        s.error = f"预处理过滤: {result.filter_reason}"
                        reason = result.filter_reason or "unknown"
                        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                    else:
                        s.status = "preprocessed"
                except Exception as e:
                    logger.error("预处理失败 %s: %s", s.sample_id, e)
                    s.status = "failed"
                    s.error = str(e)

            # 批量重采样 render_passed 音频文件
            render_passed_paths = [
                audio_dir / f"{s.sample_id}.wav"
                for s in statuses
                if s.status == "preprocessed"
            ]
            if render_passed_paths:
                try:
                    resampler = BatchResampler(
                        orig_sr=44100,
                        target_sr=16000,
                        n_workers=self.config.resample_workers,
                    )
                    resampler.resample_files(render_passed_paths)
                except Exception as e:
                    logger.warning("批量重采样失败: %s", e)

            self._save_checkpoint(statuses, checkpoint_path)
            phase_timings["preprocessing"] = time.time() - t0
            logger.info("预处理完成，耗时 %.1f 秒", phase_timings["preprocessing"])

            # 构建预处理统计并保存检查点
            try:
                if self.checkpoint_manager:
                    preprocess_stats: dict[str, dict] = {}
                    for sid, pr in preprocess_results.items():
                        preprocess_stats[sid] = {
                            "original_rms_db": pr.original_rms_db,
                            "clipping_ratio": pr.clipping_ratio,
                            "is_filtered": pr.is_filtered,
                            "filter_reason": pr.filter_reason,
                            "output_sample_rate": pr.sample_rate,
                            "resampled": True,
                        }
                    self.checkpoint_manager.complete_phase("preprocessing", preprocess_stats)
            except Exception as e:
                logger.warning("preprocessing 检查点保存失败: %s", e)
        else:
            # 从检查点加载预处理数据
            logger.info("阶段 3：跳过预处理（从检查点加载）")
            try:
                preprocess_data = self.checkpoint_manager.load_phase_data("preprocessing")
                # 将检查点中的预处理统计恢复为 preprocess_results 的简化形式
                # 同时更新 statuses 中的状态
                from types import SimpleNamespace
                for sid, stats in preprocess_data.items():
                    pr = SimpleNamespace(
                        audio=None,
                        original_rms_db=stats.get("original_rms_db", 0.0),
                        clipping_ratio=stats.get("clipping_ratio", 0.0),
                        is_filtered=stats.get("is_filtered", False),
                        filter_reason=stats.get("filter_reason"),
                        sample_rate=stats.get("output_sample_rate", 16000),
                    )
                    preprocess_results[sid] = pr
                # 更新 statuses 中 render_passed 样本为 preprocessed
                for s in statuses:
                    if s.sample_id in preprocess_data and s.status == "render_passed":
                        if not preprocess_data[s.sample_id].get("is_filtered", False):
                            s.status = "preprocessed"
            except Exception as e:
                logger.error("加载 preprocessing 检查点失败: %s", e)
                raise
            phase_timings["preprocessing"] = time.time() - t0

        # ============================================================
        # 阶段 4：Embedding 提取（GPU 批量）
        # ============================================================
        t0 = time.time()
        embed_sample_ids: list[str] = []
        embed_audio_paths: list[Path] = []
        embeddings_map: dict[str, np.ndarray] = {}
        emb_matrix = np.empty((0, 1024), dtype=np.float32)

        if _should_execute("embedding"):
            logger.info("阶段 4：Embedding 提取")

            for s in statuses:
                if s.status == "preprocessed":
                    embed_sample_ids.append(s.sample_id)
                    embed_audio_paths.append(audio_dir / f"{s.sample_id}.wav")

            # 支持增量提取：加载已有部分 Embedding，仅对缺失样本提取
            existing_ids: list[str] = []
            existing_emb = np.empty((0, 1024), dtype=np.float32)
            if self.checkpoint_manager:
                try:
                    missing_ids = self.checkpoint_manager.get_missing_sample_ids(embed_sample_ids)
                    if len(missing_ids) < len(embed_sample_ids):
                        existing_ids, existing_emb = self.checkpoint_manager.load_partial_embeddings()
                        logger.info(
                            "增量 Embedding 提取：已有 %d，缺失 %d",
                            len(existing_ids), len(missing_ids),
                        )
                except Exception as e:
                    logger.warning("加载部分 Embedding 失败，将全量提取: %s", e)
                    missing_ids = embed_sample_ids
            else:
                missing_ids = embed_sample_ids

            # 提取缺失样本的 Embedding
            if missing_ids:
                missing_paths = [audio_dir / f"{sid}.wav" for sid in missing_ids]
                batch_size = self.config.embedding_batch_size

                # 按批次提取并保存
                for batch_start in range(0, len(missing_paths), batch_size):
                    batch_end = min(batch_start + batch_size, len(missing_paths))
                    batch_paths = missing_paths[batch_start:batch_end]
                    batch_ids = missing_ids[batch_start:batch_end]

                    batch_emb = self.extract_embeddings_batch(batch_paths)

                    try:
                        if self.checkpoint_manager:
                            self.checkpoint_manager.save_embedding_batch(batch_ids, batch_emb)
                    except Exception as e:
                        logger.warning("Embedding 批次保存失败: %s", e)

                # 合并已有和新提取的 Embedding
                try:
                    if self.checkpoint_manager:
                        self.checkpoint_manager.finalize_embeddings()
                except Exception as e:
                    logger.warning("Embedding 合并失败: %s", e)

            # 构建完整的 embeddings_map
            if self.checkpoint_manager:
                try:
                    all_ids, all_emb = self.checkpoint_manager.load_partial_embeddings()
                    for i, sid in enumerate(all_ids):
                        embeddings_map[sid] = all_emb[i]
                    embed_sample_ids = all_ids
                    emb_matrix = all_emb
                except Exception:
                    pass

            # 如果 checkpoint_manager 加载失败，回退到内存中的数据
            if not embeddings_map and embed_audio_paths:
                emb_matrix = self.extract_embeddings_batch(embed_audio_paths)
                for i, sid in enumerate(embed_sample_ids):
                    embeddings_map[sid] = emb_matrix[i]

            status_map = {s.sample_id: s for s in statuses}
            for sid in embeddings_map:
                if sid in status_map:
                    status_map[sid].status = "embedded"

            self._save_checkpoint(statuses, checkpoint_path)
            phase_timings["embedding"] = time.time() - t0
            logger.info("Embedding 提取完成，耗时 %.1f 秒", phase_timings["embedding"])

            try:
                if self.checkpoint_manager:
                    self.checkpoint_manager.complete_phase("embedding", {
                        "sample_ids": np.array(embed_sample_ids, dtype=str),
                        "embeddings": emb_matrix,
                    })
            except Exception as e:
                logger.warning("embedding 检查点保存失败: %s", e)
        else:
            # 从检查点加载 Embedding 数据
            logger.info("阶段 4：跳过 Embedding 提取（从检查点加载）")
            try:
                embedding_data = self.checkpoint_manager.load_phase_data("embedding")
                loaded_ids = list(np.asarray(embedding_data["sample_ids"]).astype(str))
                loaded_emb = np.asarray(embedding_data["embeddings"]).astype(np.float32)
                embed_sample_ids = loaded_ids
                emb_matrix = loaded_emb
                for i, sid in enumerate(loaded_ids):
                    embeddings_map[sid] = loaded_emb[i]
                # 更新 statuses
                status_map = {s.sample_id: s for s in statuses}
                for sid in loaded_ids:
                    if sid in status_map:
                        status_map[sid].status = "embedded"
            except Exception as e:
                logger.error("加载 embedding 检查点失败: %s", e)
                raise
            phase_timings["embedding"] = time.time() - t0

        # ============================================================
        # 阶段 5：质量验证
        # ============================================================
        t0 = time.time()
        valid_sample_ids: list[str] = []
        valid_params_list: list[np.ndarray] = []
        valid_embeddings_list: list[np.ndarray] = []
        valid_midi_conditions: list[dict] = []
        valid_audio_stats: list[dict] = []
        cond_map = {c.label: c for c in conditions}

        if _should_execute("validation"):
            logger.info("阶段 5：质量验证")

            for s in statuses:
                if s.status != "embedded":
                    continue
                if s.sample_id not in embeddings_map:
                    continue

                pr = preprocess_results.get(s.sample_id)
                if pr is not None and pr.audio is not None:
                    qr = self.validator.validate_sample(
                        pr.audio, pr.sample_rate, s.sample_id
                    )
                    if not qr.is_valid:
                        reason = qr.filter_reason or "quality"
                        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                        continue

                valid_sample_ids.append(s.sample_id)
                valid_params_list.append(
                    self._lookup_params(all_params_batches, s.preset_index, n_presets)
                )
                valid_embeddings_list.append(embeddings_map[s.sample_id])

                cond = cond_map.get(s.condition)
                if cond:
                    valid_midi_conditions.append({
                        "note": cond.note,
                        "velocity": cond.velocity,
                        "duration_sec": cond.duration_sec,
                    })
                else:
                    valid_midi_conditions.append({"note": 60, "velocity": 100, "duration_sec": 2.0})

                pr = preprocess_results.get(s.sample_id)
                if pr:
                    valid_audio_stats.append({
                        "original_rms": pr.original_rms_db,
                        "original_peak": float(np.max(np.abs(pr.audio)) if pr.audio is not None else 0.0),
                        "clipping_ratio": pr.clipping_ratio,
                    })
                else:
                    valid_audio_stats.append({"original_rms": 0.0, "original_peak": 0.0, "clipping_ratio": 0.0})

            final_valid_count = len(valid_sample_ids)
            if valid_embeddings_list:
                all_embeddings = np.stack(valid_embeddings_list, axis=0)
                valid_mask = np.ones(final_valid_count, dtype=bool)
                quality_report = self.validator.validate_embeddings(
                    all_embeddings, target_samples, valid_mask
                )
            else:
                all_embeddings = np.empty((0, 1024), dtype=np.float32)
                quality_report = None

            valid_params = (
                np.stack(valid_params_list, axis=0)
                if valid_params_list
                else np.empty((0, 45), dtype=np.float32)
            )

            phase_timings["validation"] = time.time() - t0
            logger.info("质量验证完成，耗时 %.1f 秒", phase_timings["validation"])

            try:
                if self.checkpoint_manager:
                    quality_report_dict = {}
                    if quality_report is not None:
                        try:
                            quality_report_dict = asdict(quality_report)
                        except Exception:
                            quality_report_dict = {"summary": str(quality_report)}
                    self.checkpoint_manager.complete_phase("validation", {
                        "valid_sample_ids": valid_sample_ids,
                        "quality_report": quality_report_dict,
                    })
            except Exception as e:
                logger.warning("validation 检查点保存失败: %s", e)
        else:
            # 从检查点加载验证数据
            logger.info("阶段 5：跳过质量验证（从检查点加载）")
            try:
                validation_data = self.checkpoint_manager.load_phase_data("validation")
                valid_sample_ids = validation_data.get("valid_sample_ids", [])
                # 重建 valid_params, valid_embeddings 等
                for sid in valid_sample_ids:
                    s_match = next((s for s in statuses if s.sample_id == sid), None)
                    if s_match:
                        valid_params_list.append(
                            self._lookup_params(all_params_batches, s_match.preset_index, n_presets)
                        )
                        if sid in embeddings_map:
                            valid_embeddings_list.append(embeddings_map[sid])
                        cond = cond_map.get(s_match.condition)
                        if cond:
                            valid_midi_conditions.append({
                                "note": cond.note,
                                "velocity": cond.velocity,
                                "duration_sec": cond.duration_sec,
                            })
                        else:
                            valid_midi_conditions.append({"note": 60, "velocity": 100, "duration_sec": 2.0})
                        pr = preprocess_results.get(sid)
                        if pr:
                            valid_audio_stats.append({
                                "original_rms": pr.original_rms_db,
                                "original_peak": float(np.max(np.abs(pr.audio)) if pr.audio is not None else 0.0),
                                "clipping_ratio": pr.clipping_ratio,
                            })
                        else:
                            valid_audio_stats.append({"original_rms": 0.0, "original_peak": 0.0, "clipping_ratio": 0.0})
            except Exception as e:
                logger.error("加载 validation 检查点失败: %s", e)
                raise

            final_valid_count = len(valid_sample_ids)
            if valid_embeddings_list:
                all_embeddings = np.stack(valid_embeddings_list, axis=0)
            else:
                all_embeddings = np.empty((0, 1024), dtype=np.float32)
            valid_params = (
                np.stack(valid_params_list, axis=0)
                if valid_params_list
                else np.empty((0, 45), dtype=np.float32)
            )
            quality_report = None
            phase_timings["validation"] = time.time() - t0

        # ============================================================
        # 阶段 6：数据集保存（HDF5）
        # ============================================================
        t0 = time.time()
        hdf5_path = output_dir / "production_dataset.h5"

        if _should_execute("saving"):
            logger.info("阶段 6：保存 HDF5 数据集")

            metadata = {
                "param_names": [name for name, _, _ in CORE_PARAMS],
                "param_ranges": [(lo, hi) for _, lo, hi in CORE_PARAMS],
                "sampling_strategy": self.config.sampling_strategy,
                "seed": self.config.seed,
                "production_timestamp": datetime.now(timezone.utc).isoformat(),
                "vital_version": str(self.vital_vst_path),
            }
            config_dict = asdict(self.config)
            config_yaml = yaml.dump(config_dict, default_flow_style=False)

            if final_valid_count > 0:
                self.save_production_hdf5(
                    output_path=hdf5_path,
                    params=valid_params,
                    sample_ids=valid_sample_ids,
                    embeddings=all_embeddings,
                    midi_conditions=valid_midi_conditions,
                    audio_stats=valid_audio_stats,
                    metadata=metadata,
                    config_yaml=config_yaml,
                )

            phase_timings["saving"] = time.time() - t0
            logger.info("HDF5 保存完成，耗时 %.1f 秒", phase_timings["saving"])

            # 计算数据集划分统计
            splits_info = {}
            if final_valid_count > 0:
                n_train = int(final_valid_count * 0.8)
                n_val = int(final_valid_count * 0.1)
                n_test = final_valid_count - n_train - n_val
                splits_info = {"train": n_train, "val": n_val, "test": n_test}

            try:
                if self.checkpoint_manager:
                    self.checkpoint_manager.complete_phase("saving", {
                        "hdf5_path": str(hdf5_path),
                        "splits": splits_info,
                    })
            except Exception as e:
                logger.warning("saving 检查点保存失败: %s", e)
        else:
            logger.info("阶段 6：跳过 HDF5 保存（从检查点加载）")
            phase_timings["saving"] = time.time() - t0

        # ============================================================
        # 阶段 7：分布分析
        # ============================================================
        t0 = time.time()
        dist_report = None

        if _should_execute("analysis"):
            logger.info("阶段 7：分布分析")

            if final_valid_count >= 2:
                dist_report = self.analyzer.generate_report(all_embeddings, valid_params)
                report_path = output_dir / "distribution_report.json"
                self.analyzer.save_report(dist_report, report_path)

            phase_timings["analysis"] = time.time() - t0
            logger.info("分布分析完成，耗时 %.1f 秒", phase_timings["analysis"])

            try:
                if self.checkpoint_manager:
                    report_dict = {}
                    if dist_report is not None:
                        try:
                            report_dict = asdict(dist_report)
                        except Exception:
                            report_dict = {"summary": str(dist_report)}
                    self.checkpoint_manager.complete_phase("analysis", {
                        "report": report_dict,
                    })
            except Exception as e:
                logger.warning("analysis 检查点保存失败: %s", e)
        else:
            logger.info("阶段 7：跳过分布分析（从检查点加载）")
            phase_timings["analysis"] = time.time() - t0

        # ============================================================
        # 生成 ProductionSummary
        # ============================================================
        total_time = time.time() - total_start
        failed_count = sum(1 for s in statuses if s.status == "failed")
        filtered_count = sum(1 for s in statuses if s.status == "filtered")

        # 计算实际总预设数（含重试）
        total_presets_all = next_preset_index
        total_samples_all = len(statuses)

        estimate = self.estimate_resources(total_presets_all, n_conditions)

        summary = ProductionSummary(
            total_presets=total_presets_all,
            total_samples=total_samples_all,
            valid_samples=final_valid_count,
            filtered_samples=filtered_count,
            failed_samples=failed_count,
            filter_reasons=filter_reasons,
            total_time_sec=total_time,
            phase_timings=phase_timings,
            storage_estimate={
                "wav_total_gb": str(estimate["wav_size_gb"]),
                "hdf5_size_mb": str(estimate["hdf5_size_mb"]),
            },
            retry_rounds=retry_round,
            retry_stats=[asdict(rs) for rs in retry_stats_list],
        )

        # 保存 production_summary.json
        summary_dict = asdict(summary)
        summary_dict["dataset_splits"] = {}
        if final_valid_count > 0:
            n_train = int(final_valid_count * 0.8)
            n_val = int(final_valid_count * 0.1)
            n_test = final_valid_count - n_train - n_val
            summary_dict["dataset_splits"] = {
                "train": n_train, "val": n_val, "test": n_test,
            }
        if dist_report is not None:
            summary_dict["embedding_distribution"] = {
                "pca_top10_cumulative_variance": (
                    sum(dist_report.pca_variance_ratios[:10])
                    if len(dist_report.pca_variance_ratios) >= 10
                    else sum(dist_report.pca_variance_ratios)
                ),
                "mean_cosine_similarity": dist_report.cosine_sim_mean,
                "diversity_warning": dist_report.diversity_warning,
            }

        summary_path = output_dir / "production_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)

        logger.info(
            "生产完成：%d 有效样本 / %d 总样本（%d 轮重试），耗时 %.1f 秒",
            final_valid_count, total_samples_all, retry_round, total_time,
        )

        return summary

    def save_production_hdf5(
        self,
        output_path: Path,
        params: np.ndarray,
        sample_ids: list[str],
        embeddings: np.ndarray,
        midi_conditions: list[dict],
        audio_stats: list[dict],
        metadata: dict,
        config_yaml: str,
    ) -> None:
        """保存生产数据集为 HDF5。

        将数据按 80/10/10 比例划分为 train/val/test 三个分组，
        每个分组包含 params、embeddings、midi_notes、midi_velocities、
        midi_durations、sample_ids 和 audio_stats 子组。metadata 组保存元数据。

        Args:
            output_path: HDF5 输出路径
            params: (N, 45) 参数矩阵 float32
            sample_ids: 长度 N 的样本标识符字符串列表
            embeddings: (N, 1024) embedding 矩阵 float32
            midi_conditions: 长度 N 的字典列表，每个包含 note/velocity/duration_sec
            audio_stats: 长度 N 的字典列表，每个包含 original_rms/original_peak/clipping_ratio
            metadata: 元数据字典
            config_yaml: 完整生产配置 YAML 字符串
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(params)
        if n == 0:
            logger.warning("无有效样本，跳过 HDF5 保存")
            return

        # 构建 sample_ids numpy 数组
        sample_ids_array = np.array(sample_ids, dtype=object)

        # 构建 MIDI 条件数组
        midi_notes = np.array(
            [c["note"] for c in midi_conditions], dtype=np.int32
        )
        midi_velocities = np.array(
            [c["velocity"] for c in midi_conditions], dtype=np.int32
        )
        midi_durations = np.array(
            [c["duration_sec"] for c in midi_conditions], dtype=np.float32
        )

        # 构建音频统计数组
        original_rms = np.array(
            [s["original_rms"] for s in audio_stats], dtype=np.float32
        )
        original_peak = np.array(
            [s["original_peak"] for s in audio_stats], dtype=np.float32
        )
        clipping_ratio = np.array(
            [s["clipping_ratio"] for s in audio_stats], dtype=np.float32
        )

        # 80/10/10 数据集划分
        rng = np.random.default_rng(self.config.seed)
        indices = np.arange(n)
        rng.shuffle(indices)

        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        # test 取剩余部分
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        with h5py.File(str(output_path), "w") as f:
            # 写入 train/val/test 分组
            for split_name, idx in splits.items():
                grp = f.create_group(split_name)
                grp.create_dataset(
                    "params",
                    data=params[idx].astype(np.float32),
                )
                grp.create_dataset(
                    "embeddings",
                    data=embeddings[idx].astype(np.float32),
                )
                # sample_ids 数据集
                split_ids = sample_ids_array[idx].tolist()
                dt = h5py.string_dtype()
                grp.create_dataset("sample_ids", data=split_ids, dtype=dt)

                grp.create_dataset(
                    "midi_notes",
                    data=midi_notes[idx],
                )
                grp.create_dataset(
                    "midi_velocities",
                    data=midi_velocities[idx],
                )
                grp.create_dataset(
                    "midi_durations",
                    data=midi_durations[idx],
                )
                # audio_stats 子组
                stats_grp = grp.create_group("audio_stats")
                stats_grp.create_dataset(
                    "original_rms",
                    data=original_rms[idx],
                )
                stats_grp.create_dataset(
                    "original_peak",
                    data=original_peak[idx],
                )
                stats_grp.create_dataset(
                    "clipping_ratio",
                    data=clipping_ratio[idx],
                )

            # 写入 metadata 组
            meta_grp = f.create_group("metadata")

            # param_names: 字符串列表
            param_names = metadata["param_names"]
            dt = h5py.string_dtype()
            meta_grp.create_dataset(
                "param_names", data=param_names, dtype=dt
            )

            # param_ranges: (45, 2) float32
            param_ranges = np.array(
                metadata["param_ranges"], dtype=np.float32
            )
            meta_grp.create_dataset("param_ranges", data=param_ranges)

            # 标量字符串属性
            meta_grp.attrs["sampling_strategy"] = metadata[
                "sampling_strategy"
            ]
            meta_grp.attrs["seed"] = metadata["seed"]
            meta_grp.attrs["production_timestamp"] = metadata[
                "production_timestamp"
            ]
            meta_grp.attrs["vital_version"] = metadata.get(
                "vital_version", ""
            )
            meta_grp.attrs["production_config"] = config_yaml

        logger.info(
            "HDF5 数据集已保存至 %s（train=%d, val=%d, test=%d）",
            output_path,
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

    @staticmethod
    def _render_worker(
        tasks: list[tuple[Path, Path, str, int, int, float]],
        vital_vst_path: Path,
    ) -> list[tuple[str, bool, str | None]]:
        """渲染 worker 进程函数。

        每个 worker 加载独立的 Vital VST3 实例，逐个渲染任务列表中的预设。
        在函数体内导入 AudioRenderer 以避免子进程导入问题。

        Args:
            tasks: [(preset_path, output_path, sample_id, midi_note, velocity, duration_sec), ...]
            vital_vst_path: VST3 插件路径

        Returns:
            [(sample_id, success, error_msg), ...]
        """
        # 在 worker 内部导入，避免子进程序列化问题
        from src.audio_renderer import AudioRenderer, RenderConfig

        results: list[tuple[str, bool, str | None]] = []

        # 每个 worker 加载独立的 VST3 实例（使用默认配置初始化）
        try:
            renderer = AudioRenderer(vital_vst_path, RenderConfig())
        except Exception as e:
            # 插件加载失败，所有任务标记为失败
            logger.error("Worker 加载 VST3 失败: %s", e)
            for task in tasks:
                results.append((task[2], False, f"VST3 加载失败: {e}"))
            return results

        # 逐个渲染任务，每个任务使用对应的 MIDI 条件
        for preset_path, output_path, sample_id, midi_note, velocity, duration_sec in tasks:
            try:
                # 更新渲染配置为当前任务的 MIDI 条件
                renderer._config.midi_note = midi_note
                renderer._config.velocity = velocity
                renderer._config.duration_sec = duration_sec

                success = renderer.render_preset(
                    Path(preset_path), Path(output_path)
                )
                if success:
                    results.append((sample_id, True, None))
                else:
                    results.append((sample_id, False, "渲染返回 False"))
            except Exception as e:
                logger.error("渲染失败 %s: %s", sample_id, e)
                results.append((sample_id, False, str(e)))

        return results

    def render_parallel(
        self,
        render_tasks: list[tuple[Path, Path, str, int, int, float]],
        statuses: list[SampleStatus],
        checkpoint_path: Path,
    ) -> list[tuple[str, bool, str | None]]:
        """多进程并行渲染。

        使用 multiprocessing.Pool 将渲染任务均匀分配给 n_workers 个 worker，
        收集结果并更新样本状态，每 checkpoint_interval 个样本保存一次 checkpoint。

        Args:
            render_tasks: [(preset_path, output_path, sample_id, midi_note, velocity, duration_sec), ...]
            statuses: 样本状态列表，渲染完成后更新对应样本的状态
            checkpoint_path: checkpoint 文件路径

        Returns:
            所有 worker 的结果列表 [(sample_id, success, error_msg), ...]
        """
        n_workers = self.config.n_workers
        if not render_tasks:
            return []

        # 将任务均匀分配给 n_workers 个 worker
        chunks: list[list[tuple[Path, Path, str, int, int, float]]] = [
            [] for _ in range(min(n_workers, len(render_tasks)))
        ]
        for i, task in enumerate(render_tasks):
            chunks[i % len(chunks)].append(task)

        # 构建 sample_id → status 索引，用于快速更新
        status_map: dict[str, SampleStatus] = {
            s.sample_id: s for s in statuses
        }

        # 使用进程池并行渲染
        all_results: list[tuple[str, bool, str | None]] = []
        try:
            worker_args = [
                (chunk, self.vital_vst_path) for chunk in chunks if chunk
            ]
            pool = multiprocessing.Pool(processes=len(worker_args))
            try:
                chunk_results = pool.starmap(
                    ParallelProducer._render_worker, worker_args
                )
            finally:
                pool.close()
                pool.join()

            # 收集结果并更新状态
            completed_count = 0
            for results in chunk_results:
                for sample_id, success, error_msg in results:
                    all_results.append((sample_id, success, error_msg))
                    # 更新对应样本的状态
                    if sample_id in status_map:
                        if success:
                            status_map[sample_id].status = "rendered"
                        else:
                            status_map[sample_id].status = "failed"
                            status_map[sample_id].error = error_msg
                    completed_count += 1
                    # 每 checkpoint_interval 个样本保存一次 checkpoint
                    if completed_count % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(statuses, checkpoint_path)
                        logger.info(
                            "渲染进度: %d/%d 完成",
                            completed_count,
                            len(render_tasks),
                        )

        except Exception as e:
            # 进程崩溃时记录错误
            logger.error("多进程渲染异常: %s", e)
            # 将未完成的任务标记为失败
            completed_ids = {r[0] for r in all_results}
            for task in render_tasks:
                sample_id = task[2]
                if sample_id not in completed_ids:
                    all_results.append(
                        (sample_id, False, f"进程崩溃: {e}")
                    )
                    if sample_id in status_map:
                        status_map[sample_id].status = "failed"
                        status_map[sample_id].error = f"进程崩溃: {e}"

        # 最终保存一次 checkpoint
        self._save_checkpoint(statuses, checkpoint_path)
        logger.info(
            "渲染完成: 共 %d 个任务, 成功 %d, 失败 %d",
            len(all_results),
            sum(1 for _, s, _ in all_results if s),
            sum(1 for _, s, _ in all_results if not s),
        )

        return all_results

    def extract_embeddings_batch(
        self,
        audio_paths: list[Path],
        sample_rate: int = 16000,
        extractor: Any = None,
    ) -> np.ndarray:
        """GPU 批量 embedding 提取。

        将音频文件按 embedding_batch_size 分批，使用 EmbeddingExtractor
        逐个提取 embedding，返回 (N, 1024) embedding 矩阵。

        Args:
            audio_paths: 音频文件路径列表
            sample_rate: 音频采样率（默认 16000）
            extractor: EmbeddingExtractor 实例（可选，用于依赖注入/测试）

        Returns:
            (N, 1024) embedding 矩阵，N 为音频文件数量
        """
        if not audio_paths:
            return np.empty((0, 1024), dtype=np.float32)

        # 如果未提供 extractor，创建默认实例
        if extractor is None:
            from src.embedding_extractor import EmbeddingExtractor
            extractor = EmbeddingExtractor(
                device=self.config.embedding_device
            )

        batch_size = self.config.embedding_batch_size
        all_embeddings: list[np.ndarray] = []

        # 按 batch_size 分批处理
        import torch

        for batch_start in range(0, len(audio_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(audio_paths))
            batch_paths = audio_paths[batch_start:batch_end]

            for audio_path in batch_paths:
                try:
                    embedding = extractor.extract(Path(audio_path))
                    all_embeddings.append(embedding)
                except Exception as e:
                    # 提取失败时使用零向量占位
                    logger.error("Embedding 提取失败 %s: %s", audio_path, e)
                    all_embeddings.append(
                        np.zeros(1024, dtype=np.float32)
                    )

            # 每个 batch 结束后释放 MPS GPU 缓存，防止内存累积
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()

            logger.info(
                "Embedding 提取进度: %d/%d",
                min(batch_end, len(audio_paths)),
                len(audio_paths),
            )

        return np.stack(all_embeddings, axis=0)

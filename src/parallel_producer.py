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

from src.audio_preprocessor import AudioPreprocessor
from src.distribution_analyzer import DistributionAnalyzer
from src.multi_condition_renderer import DEFAULT_CONDITIONS, MidiCondition
from src.quality_validator import QualityValidator
from src.smart_sampler import SmartSampler
from src.training_data import CORE_PARAMS

logger = logging.getLogger(__name__)


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


@dataclass
class SampleStatus:
    """单个样本的处理状态。"""

    sample_id: str
    preset_index: int
    condition: str  # MIDI 条件标签
    status: str  # pending / rendered / preprocessed / embedded / failed
    error: str | None = None


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
    ) -> None:
        """初始化并行生产器。

        Args:
            vital_vst_path: Vital VST3 插件路径
            config: 生产配置
            preprocessor: 音频预处理器实例
            sampler: 智能采样器实例
            validator: 质量验证器实例
            analyzer: 分布分析器实例
        """
        self.vital_vst_path = Path(vital_vst_path)
        self.config = config
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.validator = validator
        self.analyzer = analyzer
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

    def _save_checkpoint(
        self, statuses: list[SampleStatus], path: Path
    ) -> None:
        """保存进度状态文件（JSON）。

        格式参考设计文档中的 checkpoint.json 格式，包含版本号、
        时间戳、配置信息和样本状态列表。

        Args:
            statuses: 样本状态列表
            path: 输出 JSON 文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()

        # 尝试读取已有 checkpoint 的 created_at
        created_at = now
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError):
                created_at = now

        checkpoint = {
            "version": 1,
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

        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        logger.info("Checkpoint 已保存至 %s（%d 个样本）", path, len(statuses))

    def _load_checkpoint(self, path: Path) -> list[SampleStatus]:
        """从 JSON 文件加载样本状态列表。

        Args:
            path: checkpoint JSON 文件路径

        Returns:
            样本状态列表。文件不存在或为空时返回空列表。
        """
        path = Path(path)

        if not path.exists():
            logger.warning("Checkpoint 文件不存在: %s", path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning("Checkpoint 文件为空: %s", path)
                    return []
                data = json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Checkpoint 文件读取失败: %s, 错误: %s", path, e)
            return []

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

        logger.info("从 checkpoint 加载了 %d 个样本状态", len(statuses))
        return statuses

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
        self, output_dir: Path, resume: bool = False
    ) -> ProductionSummary:
        """执行完整生产流水线。

        流程：
        1. 参数采样（SmartSampler）
        2. 预设生成 + 多条件渲染（多进程，使用 render_parallel）
        3. 音频预处理（AudioPreprocessor.process）
        4. Embedding 提取（GPU 批量，使用 extract_embeddings_batch）
        5. 质量验证（QualityValidator.validate_sample + validate_embeddings）
        6. 数据集保存（save_production_hdf5）
        7. 分布分析（DistributionAnalyzer.generate_report + save_report）

        Args:
            output_dir: 输出目录
            resume: 是否从断点恢复

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

        phase_timings: dict[str, float] = {}
        total_start = time.time()

        n_presets = self.n_presets
        n_conditions = self.config.n_conditions
        conditions = DEFAULT_CONDITIONS[:n_conditions]

        # ============================================================
        # 阶段 1：参数采样
        # ============================================================
        t0 = time.time()
        logger.info("阶段 1：参数采样（%d 个预设）", n_presets)
        params = self.sampler.sample(
            n_presets, strategy=self.config.sampling_strategy
        )
        phase_timings["sampling"] = time.time() - t0
        logger.info("参数采样完成，耗时 %.1f 秒", phase_timings["sampling"])

        # ============================================================
        # 构建样本状态列表 / 断点恢复
        # ============================================================
        if resume and checkpoint_path.exists():
            statuses = self._load_checkpoint(checkpoint_path)
            logger.info("从断点恢复，加载 %d 个样本状态", len(statuses))
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
        # 阶段 2：预设生成 + 多条件渲染（多进程）
        # ============================================================
        t0 = time.time()
        logger.info("阶段 2：预设生成 + 多条件渲染")

        # 生成 .vital 预设文件
        from src.preset_generator import PresetGenerator
        from src.preset_parser import PresetParser

        parser = PresetParser()
        generator = PresetGenerator(parser)

        # 构建条件标签 → MidiCondition 映射
        cond_lookup = {c.label: c for c in conditions}

        # 为每个预设生成 .vital 文件并构建渲染任务
        # 任务格式: (preset_path, audio_path, sample_id, midi_note, velocity, duration_sec)
        render_tasks: list[tuple[Path, Path, str, int, int, float]] = []
        for s in statuses:
            # 断点恢复：跳过已完成、已预处理和失败的样本
            if s.status in ("embedded", "preprocessed", "failed"):
                continue
            if s.status == "rendered":
                # 已渲染但未提取 embedding，跳过渲染阶段
                continue

            # 生成预设文件（如果尚未生成）
            preset_path = presets_dir / f"preset_{s.preset_index:05d}.vital"
            if not preset_path.exists():
                preset = generator.create_base_patch()
                # 将采样参数写入预设
                for col, (pname, _, _) in enumerate(CORE_PARAMS):
                    if pname in preset.settings:
                        preset.settings[pname] = float(
                            params[s.preset_index, col]
                        )
                parser.serialize(preset, preset_path)

            # 从条件标签解析 MIDI 参数
            cond = cond_lookup.get(s.condition)
            midi_note = cond.note if cond else 60
            velocity = cond.velocity if cond else 100
            duration_sec = cond.duration_sec if cond else 2.0

            # 构建渲染输出路径
            audio_path = audio_dir / f"{s.sample_id}.wav"
            render_tasks.append((preset_path, audio_path, s.sample_id, midi_note, velocity, duration_sec))

        # 执行多进程渲染
        if render_tasks:
            self.render_parallel(render_tasks, statuses, checkpoint_path)

        phase_timings["rendering"] = time.time() - t0
        logger.info("渲染完成，耗时 %.1f 秒", phase_timings["rendering"])

        # ============================================================
        # 阶段 3：音频预处理
        # ============================================================
        t0 = time.time()
        logger.info("阶段 3：音频预处理")

        # 收集需要预处理的样本（status == "rendered"）
        preprocess_results: dict[str, Any] = {}
        filter_reasons: dict[str, int] = {}

        for s in statuses:
            if s.status != "rendered":
                continue
            audio_path = audio_dir / f"{s.sample_id}.wav"
            if not audio_path.exists():
                s.status = "failed"
                s.error = "音频文件不存在"
                continue

            try:
                # 读取 WAV 文件
                import soundfile as sf

                audio_data, sr = sf.read(str(audio_path), dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]  # 取第一个声道

                result = self.preprocessor.process(audio_data, sr)
                preprocess_results[s.sample_id] = result

                if result.is_filtered:
                    s.status = "failed"
                    s.error = f"过滤: {result.filter_reason}"
                    reason = result.filter_reason or "unknown"
                    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                else:
                    s.status = "preprocessed"
            except Exception as e:
                logger.error("预处理失败 %s: %s", s.sample_id, e)
                s.status = "failed"
                s.error = str(e)

        self._save_checkpoint(statuses, checkpoint_path)
        phase_timings["preprocessing"] = time.time() - t0
        logger.info("预处理完成，耗时 %.1f 秒", phase_timings["preprocessing"])

        # ============================================================
        # 阶段 4：Embedding 提取（GPU 批量）
        # ============================================================
        t0 = time.time()
        logger.info("阶段 4：Embedding 提取")

        # 收集需要提取 embedding 的样本
        embed_sample_ids: list[str] = []
        embed_audio_paths: list[Path] = []
        for s in statuses:
            if s.status == "preprocessed":
                audio_path = audio_dir / f"{s.sample_id}.wav"
                embed_sample_ids.append(s.sample_id)
                embed_audio_paths.append(audio_path)

        embeddings_map: dict[str, np.ndarray] = {}
        if embed_audio_paths:
            emb_matrix = self.extract_embeddings_batch(embed_audio_paths)
            for i, sid in enumerate(embed_sample_ids):
                embeddings_map[sid] = emb_matrix[i]
                # 更新状态
                status_map = {s.sample_id: s for s in statuses}
                if sid in status_map:
                    status_map[sid].status = "embedded"

        self._save_checkpoint(statuses, checkpoint_path)
        phase_timings["embedding"] = time.time() - t0
        logger.info("Embedding 提取完成，耗时 %.1f 秒", phase_timings["embedding"])

        # ============================================================
        # 阶段 5：质量验证
        # ============================================================
        t0 = time.time()
        logger.info("阶段 5：质量验证")

        # 收集所有有效样本的数据
        valid_sample_ids: list[str] = []
        valid_params_list: list[np.ndarray] = []
        valid_embeddings_list: list[np.ndarray] = []
        valid_midi_conditions: list[dict] = []
        valid_audio_stats: list[dict] = []

        # 构建条件标签到 MidiCondition 的映射
        cond_map = {c.label: c for c in conditions}

        for s in statuses:
            if s.status != "embedded":
                continue
            if s.sample_id not in embeddings_map:
                continue

            # 质量验证（对预处理后的音频）
            pr = preprocess_results.get(s.sample_id)
            if pr is not None and pr.audio is not None:
                qr = self.validator.validate_sample(
                    pr.audio, pr.sample_rate, s.sample_id
                )
                if not qr.is_valid:
                    reason = qr.filter_reason or "quality"
                    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                    continue

            # 收集有效样本数据
            valid_sample_ids.append(s.sample_id)
            valid_params_list.append(params[s.preset_index])
            valid_embeddings_list.append(embeddings_map[s.sample_id])

            # MIDI 条件
            cond = cond_map.get(s.condition)
            if cond:
                valid_midi_conditions.append({
                    "note": cond.note,
                    "velocity": cond.velocity,
                    "duration_sec": cond.duration_sec,
                })
            else:
                valid_midi_conditions.append({
                    "note": 60, "velocity": 100, "duration_sec": 2.0,
                })

            # 音频统计
            pr = preprocess_results.get(s.sample_id)
            if pr:
                valid_audio_stats.append({
                    "original_rms": pr.original_rms_db,
                    "original_peak": float(
                        np.max(np.abs(pr.audio)) if pr.audio is not None else 0.0
                    ),
                    "clipping_ratio": pr.clipping_ratio,
                })
            else:
                valid_audio_stats.append({
                    "original_rms": 0.0,
                    "original_peak": 0.0,
                    "clipping_ratio": 0.0,
                })

        # Embedding 矩阵质量验证
        valid_count = len(valid_sample_ids)
        if valid_embeddings_list:
            all_embeddings = np.stack(valid_embeddings_list, axis=0)
            valid_mask = np.ones(valid_count, dtype=bool)
            quality_report = self.validator.validate_embeddings(
                all_embeddings, self.config.target_samples, valid_mask
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

        # ============================================================
        # 阶段 6：数据集保存（HDF5）
        # ============================================================
        t0 = time.time()
        logger.info("阶段 6：保存 HDF5 数据集")

        hdf5_path = output_dir / "production_dataset.h5"

        # 构建元数据
        metadata = {
            "param_names": [name for name, _, _ in CORE_PARAMS],
            "param_ranges": [(lo, hi) for _, lo, hi in CORE_PARAMS],
            "sampling_strategy": self.config.sampling_strategy,
            "seed": self.config.seed,
            "production_timestamp": datetime.now(timezone.utc).isoformat(),
            "vital_version": str(self.vital_vst_path),
        }

        # 序列化生产配置为 YAML
        config_dict = asdict(self.config)
        config_yaml = yaml.dump(config_dict, default_flow_style=False)

        if valid_count > 0:
            self.save_production_hdf5(
                output_path=hdf5_path,
                params=valid_params,
                embeddings=all_embeddings,
                midi_conditions=valid_midi_conditions,
                audio_stats=valid_audio_stats,
                metadata=metadata,
                config_yaml=config_yaml,
            )

        phase_timings["saving"] = time.time() - t0
        logger.info("HDF5 保存完成，耗时 %.1f 秒", phase_timings["saving"])

        # ============================================================
        # 阶段 7：分布分析
        # ============================================================
        t0 = time.time()
        logger.info("阶段 7：分布分析")

        dist_report = None
        if valid_count >= 2:
            dist_report = self.analyzer.generate_report(
                all_embeddings, valid_params
            )
            report_path = output_dir / "distribution_report.json"
            self.analyzer.save_report(dist_report, report_path)

        phase_timings["analysis"] = time.time() - t0
        logger.info("分布分析完成，耗时 %.1f 秒", phase_timings["analysis"])

        # ============================================================
        # 生成 ProductionSummary 和 production_summary.json
        # ============================================================
        total_time = time.time() - total_start

        # 统计失败样本数
        failed_count = sum(1 for s in statuses if s.status == "failed")
        filtered_count = sum(filter_reasons.values())

        # 存储估算
        estimate = self.estimate_resources(n_presets, n_conditions)

        summary = ProductionSummary(
            total_presets=n_presets,
            total_samples=n_presets * n_conditions,
            valid_samples=valid_count,
            filtered_samples=filtered_count,
            failed_samples=failed_count,
            filter_reasons=filter_reasons,
            total_time_sec=total_time,
            phase_timings=phase_timings,
            storage_estimate={
                "wav_total_gb": str(estimate["wav_size_gb"]),
                "hdf5_size_mb": str(estimate["hdf5_size_mb"]),
            },
        )

        # 保存人类可读的 production_summary.json
        summary_dict = asdict(summary)
        # 添加额外信息
        summary_dict["dataset_splits"] = {}
        if valid_count > 0:
            n_train = int(valid_count * 0.8)
            n_val = int(valid_count * 0.1)
            n_test = valid_count - n_train - n_val
            summary_dict["dataset_splits"] = {
                "train": n_train,
                "val": n_val,
                "test": n_test,
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
            "生产完成：%d 有效样本 / %d 总样本，耗时 %.1f 秒",
            valid_count,
            n_presets * n_conditions,
            total_time,
        )

        return summary

    def save_production_hdf5(
        self,
        output_path: Path,
        params: np.ndarray,
        embeddings: np.ndarray,
        midi_conditions: list[dict],
        audio_stats: list[dict],
        metadata: dict,
        config_yaml: str,
    ) -> None:
        """保存生产数据集为 HDF5。

        将数据按 80/10/10 比例划分为 train/val/test 三个分组，
        每个分组包含 params、embeddings、midi_notes、midi_velocities、
        midi_durations 和 audio_stats 子组。metadata 组保存元数据。

        Args:
            output_path: HDF5 输出路径
            params: (N, 45) 参数矩阵 float32
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

            logger.info(
                "Embedding 提取进度: %d/%d",
                min(batch_end, len(audio_paths)),
                len(audio_paths),
            )

        return np.stack(all_embeddings, axis=0)

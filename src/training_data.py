"""
训练数据集生成模块。

通过随机采样参数空间生成训练数据集，每个样本包含参数向量和对应的
渲染音频 MuQ embedding。支持 HDF5 格式存储和 80/10/10 数据集划分。
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from src.audio_renderer import AudioRenderer
from src.embedding_extractor import EmbeddingExtractor
from src.preset_generator import PresetGenerator
from src.preset_parser import VitalPreset

logger = logging.getLogger(__name__)


# 45 core parameters: (name, min, max)
CORE_PARAMS: list[tuple[str, float, float]] = [
    # osc_1 core (6)
    ("osc_1_level", 0.0, 1.0),
    ("osc_1_transpose", -48.0, 48.0),
    ("osc_1_tune", -1.0, 1.0),
    ("osc_1_wave_frame", 0.0, 256.0),
    ("osc_1_unison_voices", 1.0, 16.0),
    ("osc_1_unison_detune", 0.0, 100.0),
    # filter_1 core (6)
    ("filter_1_cutoff", 8.0, 136.0),
    ("filter_1_resonance", 0.0, 1.0),
    ("filter_1_drive", 0.0, 1.0),
    ("filter_1_mix", 0.0, 1.0),
    ("filter_1_model", 0.0, 5.0),
    ("filter_1_style", 0.0, 3.0),
    # env_1 ADSR (4)
    ("env_1_attack", 0.0, 4.0),
    ("env_1_decay", 0.0, 4.0),
    ("env_1_sustain", 0.0, 1.0),
    ("env_1_release", 0.0, 4.0),
    # 9 Effect Switches
    ("chorus_on", 0.0, 1.0),
    ("compressor_on", 0.0, 1.0),
    ("delay_on", 0.0, 1.0),
    ("distortion_on", 0.0, 1.0),
    ("eq_on", 0.0, 1.0),
    ("flanger_on", 0.0, 1.0),
    ("phaser_on", 0.0, 1.0),
    ("reverb_on", 0.0, 1.0),
    ("filter_fx_on", 0.0, 1.0),
    # 9 dry_wet / mix params
    ("chorus_dry_wet", 0.0, 1.0),
    ("delay_dry_wet", 0.0, 1.0),
    ("flanger_dry_wet", 0.0, 1.0),
    ("phaser_dry_wet", 0.0, 1.0),
    ("reverb_dry_wet", 0.0, 1.0),
    ("distortion_mix", 0.0, 1.0),
    ("compressor_mix", 0.0, 1.0),
    ("filter_fx_mix", 0.0, 1.0),
    ("reverb_decay_time", 0.0, 1.0),
    # Other effect core params (7) to reach 45
    ("distortion_drive", 0.0, 1.0),
    ("delay_frequency", 0.0, 10.0),
    ("chorus_frequency", 0.0, 10.0),
    ("chorus_feedback", -1.0, 1.0),
    ("flanger_frequency", 0.0, 10.0),
    ("phaser_frequency", 0.0, 10.0),
    ("reverb_size", 0.0, 1.0),
    ("delay_feedback", -1.0, 1.0),
    ("compressor_attack", 0.0, 1.0),
    ("compressor_release", 0.0, 1.0),
    ("eq_low_gain", -6.0, 6.0),
]

# Names of effect switch parameters (binary 0/1)
EFFECT_SWITCH_NAMES: set[str] = {
    "chorus_on", "compressor_on", "delay_on", "distortion_on",
    "eq_on", "flanger_on", "phaser_on", "reverb_on", "filter_fx_on",
}

# Indices of effect switches within CORE_PARAMS
EFFECT_SWITCH_INDICES: list[int] = [
    i for i, (name, _, _) in enumerate(CORE_PARAMS)
    if name in EFFECT_SWITCH_NAMES
]

NUM_PARAMS = len(CORE_PARAMS)  # Should be 45


@dataclass
class DatasetMetadata:
    """训练数据集元数据。

    Attributes:
        param_ranges: 参数名到 (min, max) 值域的映射
        param_names: 参数名称列表（有序）
        total_samples: 成功生成的总样本数
        split_ratio: 训练/验证/测试划分比例
        failed_samples: 失败跳过的样本数
    """

    param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    param_names: list[str] = field(default_factory=list)
    total_samples: int = 0
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    failed_samples: int = 0


def split_dataset(
    n: int,
    ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[int, int, int]:
    """Compute train/val/test split sizes from total count.

    Ensures train + val + test == n and each size is within 1 of the
    ideal proportion.

    Args:
        n: Total number of samples.
        ratio: (train, val, test) proportions summing to 1.0.

    Returns:
        (n_train, n_val, n_test) tuple.
    """
    n_test = round(n * ratio[2])
    n_val = round(n * ratio[1])
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


class TrainingDataGenerator:
    """训练数据集生成器。

    通过随机采样参数空间、渲染音频、提取 embedding 生成
    (参数向量, embedding) 配对数据集。
    """

    SPLIT_RATIO: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __init__(
        self,
        generator: PresetGenerator,
        renderer: AudioRenderer,
        extractor: EmbeddingExtractor,
    ) -> None:
        """初始化训练数据生成器。

        Args:
            generator: 预设生成器，用于创建 Vital 预设
            renderer: 音频渲染器，用于将预设渲染为音频
            extractor: embedding 提取器，用于从音频提取 MuQ embedding
        """
        self._generator = generator
        self._renderer = renderer
        self._extractor = extractor

    def sample_parameters(self, n: int) -> np.ndarray:
        """在 45 个核心参数的有效值域内均匀随机采样。

        Effect switch 参数采样后四舍五入为 0.0 或 1.0（二值）。

        Args:
            n: 采样数量

        Returns:
            (n, 45) 的参数矩阵，float32
        """
        params = np.empty((n, NUM_PARAMS), dtype=np.float32)

        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            params[:, col] = np.random.uniform(lo, hi, size=n)

        # Round effect switches to binary 0/1
        for idx in EFFECT_SWITCH_INDICES:
            params[:, idx] = np.round(params[:, idx])

        return params

    def _params_to_preset(self, param_vector: np.ndarray) -> VitalPreset:
        """Convert a single parameter vector to a VitalPreset.

        Starts from the generator's base patch and overrides the 45 core
        parameters with the sampled values.

        Args:
            param_vector: 1-D array of length 45.

        Returns:
            VitalPreset with overridden parameters.
        """
        preset = self._generator.create_base_patch()
        for i, (name, _, _) in enumerate(CORE_PARAMS):
            preset.settings[name] = float(param_vector[i])
        return preset


    @staticmethod
    def save_hdf5(
        output_path: Path,
        train_params: np.ndarray,
        train_embeddings: np.ndarray,
        val_params: np.ndarray,
        val_embeddings: np.ndarray,
        test_params: np.ndarray,
        test_embeddings: np.ndarray,
        metadata: DatasetMetadata,
    ) -> None:
        """Save dataset splits and metadata to HDF5 file.

        HDF5 structure:
            train/params, train/embeddings
            val/params, val/embeddings
            test/params, test/embeddings
            metadata/param_names, metadata/param_ranges, metadata/generation_log

        Args:
            output_path: Path to the .h5 output file.
            train_params, train_embeddings: Training split arrays.
            val_params, val_embeddings: Validation split arrays.
            test_params, test_embeddings: Test split arrays.
            metadata: DatasetMetadata with parameter info.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            # Data splits
            for split_name, p, e in [
                ("train", train_params, train_embeddings),
                ("val", val_params, val_embeddings),
                ("test", test_params, test_embeddings),
            ]:
                grp = f.create_group(split_name)
                grp.create_dataset("params", data=p.astype(np.float32))
                grp.create_dataset("embeddings", data=e.astype(np.float32))

            # Metadata
            meta_grp = f.create_group("metadata")
            # param_names as variable-length strings
            dt = h5py.string_dtype()
            meta_grp.create_dataset(
                "param_names",
                data=metadata.param_names,
                dtype=dt,
            )
            # param_ranges as (45, 2) float32
            ranges = np.array(
                [metadata.param_ranges[n] for n in metadata.param_names],
                dtype=np.float32,
            )
            meta_grp.create_dataset("param_ranges", data=ranges)
            # generation log as JSON string
            log_info = {
                "total_samples": metadata.total_samples,
                "failed_samples": metadata.failed_samples,
                "split_ratio": list(metadata.split_ratio),
                "sampling_distribution": "uniform",
            }
            meta_grp.create_dataset(
                "generation_log",
                data=json.dumps(log_info),
                dtype=dt,
            )

        logger.info("Saved HDF5 dataset to %s", output_path)

    @staticmethod
    def load_hdf5(filepath: Path) -> dict:
        """Load dataset from HDF5 file.

        Returns:
            Dictionary with keys: train_params, train_embeddings,
            val_params, val_embeddings, test_params, test_embeddings,
            param_names, param_ranges, generation_log.
        """
        filepath = Path(filepath)
        result = {}
        with h5py.File(filepath, "r") as f:
            for split in ("train", "val", "test"):
                result[f"{split}_params"] = f[split]["params"][:]
                result[f"{split}_embeddings"] = f[split]["embeddings"][:]
            result["param_names"] = [
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in f["metadata"]["param_names"][:]
            ]
            result["param_ranges"] = f["metadata"]["param_ranges"][:]
            raw_log = f["metadata"]["generation_log"][()]
            if isinstance(raw_log, bytes):
                raw_log = raw_log.decode("utf-8")
            result["generation_log"] = json.loads(raw_log)
        return result


    def generate_dataset(
        self, n_samples: int, output_dir: Path
    ) -> DatasetMetadata:
        """Generate complete training dataset.

        Pipeline: sample parameters → create presets → render audio →
        extract embeddings → split 80/10/10 → save HDF5.

        Failed samples are skipped and logged. The final dataset contains
        no missing values.

        Args:
            n_samples: Number of parameter configurations to sample.
            output_dir: Directory to save the HDF5 dataset file.

        Returns:
            DatasetMetadata with generation statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating dataset with %d samples...", n_samples)

        # Step 1: Sample parameters
        all_params = self.sample_parameters(n_samples)

        # Step 2-3: For each sample, create preset → render → extract embedding
        successful_params: list[np.ndarray] = []
        successful_embeddings: list[np.ndarray] = []
        failed_count = 0

        for i in range(n_samples):
            try:
                param_vec = all_params[i]
                preset = self._params_to_preset(param_vec)

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    preset_path = tmp_path / f"sample_{i}.vital"
                    audio_path = tmp_path / f"sample_{i}.wav"

                    # Serialize preset
                    self._generator._parser.serialize(preset, preset_path)

                    # Render audio
                    success = self._renderer.render_preset(
                        preset_path, audio_path
                    )
                    if not success:
                        raise RuntimeError(
                            f"Render failed for sample {i}"
                        )

                    # Extract embedding
                    embedding = self._extractor.extract(audio_path)

                successful_params.append(param_vec)
                successful_embeddings.append(embedding)

                if (i + 1) % 100 == 0:
                    logger.info(
                        "Progress: %d/%d samples processed (%d failed)",
                        i + 1,
                        n_samples,
                        failed_count,
                    )

            except Exception as e:
                failed_count += 1
                logger.error("Sample %d failed: %s", i, e)
                continue

        total_successful = len(successful_params)
        if total_successful == 0:
            logger.error("All samples failed! No dataset generated.")
            return DatasetMetadata(
                param_ranges={
                    name: (lo, hi) for name, lo, hi in CORE_PARAMS
                },
                param_names=[name for name, _, _ in CORE_PARAMS],
                total_samples=0,
                split_ratio=self.SPLIT_RATIO,
                failed_samples=failed_count,
            )

        # Stack into matrices
        params_matrix = np.stack(successful_params)  # (N, 45)
        embeddings_matrix = np.stack(successful_embeddings)  # (N, embed_dim)

        logger.info(
            "Successfully generated %d/%d samples (%d failed)",
            total_successful,
            n_samples,
            failed_count,
        )

        # Step 4: Split 80/10/10
        n_train, n_val, n_test = split_dataset(
            total_successful, self.SPLIT_RATIO
        )

        # Shuffle indices for random split
        indices = np.random.permutation(total_successful)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_params = params_matrix[train_idx]
        train_embeddings = embeddings_matrix[train_idx]
        val_params = params_matrix[val_idx]
        val_embeddings = embeddings_matrix[val_idx]
        test_params = params_matrix[test_idx]
        test_embeddings = embeddings_matrix[test_idx]

        # Build metadata
        metadata = DatasetMetadata(
            param_ranges={
                name: (lo, hi) for name, lo, hi in CORE_PARAMS
            },
            param_names=[name for name, _, _ in CORE_PARAMS],
            total_samples=total_successful,
            split_ratio=self.SPLIT_RATIO,
            failed_samples=failed_count,
        )

        # Step 5: Save HDF5
        hdf5_path = output_dir / "dataset.h5"
        self.save_hdf5(
            hdf5_path,
            train_params,
            train_embeddings,
            val_params,
            val_embeddings,
            test_params,
            test_embeddings,
            metadata,
        )

        # Save metadata as JSON too for easy inspection
        meta_json_path = output_dir / "metadata.json"
        meta_json = {
            "param_names": metadata.param_names,
            "param_ranges": {
                k: list(v) for k, v in metadata.param_ranges.items()
            },
            "total_samples": metadata.total_samples,
            "split_ratio": list(metadata.split_ratio),
            "failed_samples": metadata.failed_samples,
            "sampling_distribution": "uniform",
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        }
        meta_json_path.write_text(
            json.dumps(meta_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Dataset saved: %d train, %d val, %d test -> %s",
            n_train,
            n_val,
            n_test,
            hdf5_path,
        )

        return metadata

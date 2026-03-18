"""
MuQ Embedding 提取模块。

调用 MuQ 模型从音频文件中提取固定维度的 embedding 向量。
支持单个文件提取和批量提取，包含采样率自动重采样和错误处理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# MuQ 默认配置
DEFAULT_MODEL_REPO = "OpenMuQ/MuQ-large-msd-iter"
DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_EMBEDDING_DIM = 1024


class ModelLoadError(Exception):
    """MuQ 模型加载失败错误，包含模型路径和失败原因。"""

    def __init__(self, model_path: str | None, message: str) -> None:
        self.model_path = model_path
        self.message = message
        super().__init__(
            f"Failed to load MuQ model"
            f"{f' from {model_path!r}' if model_path else ''}: {message}"
        )


@dataclass
class EmbeddingResult:
    """Embedding 提取结果。

    Attributes:
        embeddings: 文件名到 embedding 向量的映射
        dimension: embedding 向量维度
    """

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    dimension: int = DEFAULT_EMBEDDING_DIM


class EmbeddingExtractor:
    """MuQ 音频 embedding 提取器。

    加载 MuQ 预训练模型，从 WAV 音频文件中提取固定维度的 embedding 向量。
    支持自动重采样至模型要求的采样率。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    ) -> None:
        """加载 MuQ 模型。

        Args:
            model_path: 模型路径或 HuggingFace repo ID。
                None 时使用默认预训练权重。
            device: 推理设备（"cpu"、"cuda"、"mps"）
            target_sample_rate: MuQ 要求的输入采样率

        Raises:
            ModelLoadError: 模型加载失败
        """
        self._device = device
        self._target_sample_rate = target_sample_rate
        self._model_repo = model_path or DEFAULT_MODEL_REPO

        try:
            from muq import MuQ

            self._model = MuQ.from_pretrained(self._model_repo)
            self._model = self._model.to(device).eval()
            logger.info(
                "Loaded MuQ model from %s on device %s",
                self._model_repo,
                device,
            )
        except Exception as e:
            raise ModelLoadError(self._model_repo, str(e)) from e

    def extract(self, audio_path: Path) -> np.ndarray:
        """从单个 WAV 文件提取 embedding。

        读取音频文件，检查采样率并在不匹配时自动重采样，
        通过 MuQ 模型提取 embedding 向量（对时间维度取平均）。

        Args:
            audio_path: WAV 音频文件路径

        Returns:
            embedding 向量，shape 为 (embedding_dim,)

        Raises:
            FileNotFoundError: 音频文件不存在
            ValueError: 音频文件格式不兼容
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 读取音频
        try:
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception as e:
            raise ValueError(
                f"Incompatible audio format for '{audio_path}': {e}"
            ) from e

        # 转为单声道（如果是多声道）
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # 检查采样率，不匹配时自动重采样
        if sample_rate != self._target_sample_rate:
            logger.warning(
                "Sample rate mismatch for '%s': got %d Hz, expected %d Hz. "
                "Auto-resampling...",
                audio_path.name,
                sample_rate,
                self._target_sample_rate,
            )
            audio_data = self._resample(
                audio_data, sample_rate, self._target_sample_rate
            )

        # 转为 torch tensor 并添加 batch 维度
        waveform = torch.tensor(
            audio_data, dtype=torch.float32
        ).unsqueeze(0).to(self._device)

        # 提取 embedding
        with torch.no_grad():
            output = self._model(waveform, output_hidden_states=True)
            # output.last_hidden_state shape: (1, T, hidden_dim)
            # 对时间维度取平均得到固定维度向量
            embedding = output.last_hidden_state.mean(dim=1).squeeze(0)

        embedding_np = embedding.cpu().numpy()

        # 显式释放 GPU tensor，减少 MPS 内存压力
        del waveform, output, embedding
        logger.debug(
            "Extracted embedding from '%s': shape %s",
            audio_path.name,
            embedding_np.shape,
        )
        return embedding_np

    def extract_waveform(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """从已加载的音频数据提取 embedding（跳过文件 IO）。

        Args:
            audio_data: 单声道 float32 音频数据
            sample_rate: 音频采样率

        Returns:
            embedding 向量，shape 为 (embedding_dim,)
        """
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != self._target_sample_rate:
            audio_data = self._resample(audio_data, sample_rate, self._target_sample_rate)

        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(waveform, output_hidden_states=True)
            embedding = output.last_hidden_state.mean(dim=1).squeeze(0)

        embedding_np = embedding.cpu().numpy()
        del waveform, output, embedding
        return embedding_np

    def extract_batch(self, audio_dir: Path) -> EmbeddingResult:
        """批量提取目录下所有 WAV 文件的 embedding。

        遍历目录中的所有 .wav 文件，逐个提取 embedding，
        收集到 EmbeddingResult 中。

        Args:
            audio_dir: 包含 WAV 文件的目录路径

        Returns:
            EmbeddingResult 包含所有文件名到 embedding 的映射
        """
        audio_dir = Path(audio_dir)
        wav_files = sorted(audio_dir.glob("*.wav"))

        if not wav_files:
            logger.warning("No .wav files found in %s", audio_dir)
            return EmbeddingResult(embeddings={}, dimension=DEFAULT_EMBEDDING_DIM)

        logger.info(
            "Starting batch extraction: %d WAV files from %s",
            len(wav_files),
            audio_dir,
        )

        embeddings: dict[str, np.ndarray] = {}
        dimension = DEFAULT_EMBEDDING_DIM

        for wav_path in wav_files:
            try:
                embedding = self.extract(wav_path)
                embeddings[wav_path.name] = embedding
                dimension = embedding.shape[0]
                logger.info("Extracted: %s", wav_path.name)
            except Exception as e:
                logger.error(
                    "Failed to extract embedding from '%s': %s",
                    wav_path.name,
                    e,
                )

        logger.info(
            "Batch extraction complete: %d/%d successful",
            len(embeddings),
            len(wav_files),
        )

        return EmbeddingResult(embeddings=embeddings, dimension=dimension)

    def save(self, result: EmbeddingResult, output_path: Path) -> None:
        """保存 embedding 为 .npz 文件。

        使用 np.savez 保存文件名到向量的映射，
        每个文件名作为 key，对应的 embedding 向量作为 value。

        Args:
            result: EmbeddingResult 对象
            output_path: 输出 .npz 文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(output_path, **result.embeddings)
        logger.info(
            "Saved %d embeddings to %s",
            len(result.embeddings),
            output_path,
        )

    @staticmethod
    def load(filepath: Path) -> EmbeddingResult:
        """从 .npz 文件加载 embedding。

        Args:
            filepath: .npz 文件路径

        Returns:
            EmbeddingResult 对象
        """
        filepath = Path(filepath)
        data = np.load(filepath)
        embeddings = {key: data[key] for key in data.files}
        dimension = next(iter(embeddings.values())).shape[0] if embeddings else DEFAULT_EMBEDDING_DIM
        return EmbeddingResult(embeddings=embeddings, dimension=dimension)

    @staticmethod
    def _resample(
        audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """重采样音频到目标采样率。

        优先使用 soxr（快 5-10x），回退到 scipy resample_poly。

        Args:
            audio: 输入音频数据（1D float array）
            orig_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            重采样后的音频数据
        """
        if orig_sr == target_sr:
            return audio

        try:
            import soxr
            resampled = soxr.resample(audio, orig_sr, target_sr, quality="HQ")
            return resampled.astype(np.float32)
        except ImportError:
            pass

        from math import gcd
        from scipy.signal import resample_poly

        common = gcd(orig_sr, target_sr)
        up = target_sr // common
        down = orig_sr // common
        resampled = resample_poly(audio, up, down)
        return resampled.astype(np.float32)

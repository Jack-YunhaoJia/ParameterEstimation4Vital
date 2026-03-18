"""
高效批量重采样模块。

使用预计算多相滤波器系数 + 多线程并行执行批量音频重采样，
支持增量重采样（跳过已是目标采样率的文件）。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, upfirdn

logger = logging.getLogger(__name__)


class BatchResampler:
    """高效批量重采样器。

    初始化时预计算多相滤波器系数，使用多线程并行处理多个文件。
    """

    def __init__(
        self,
        orig_sr: int = 44100,
        target_sr: int = 16000,
        n_workers: int = 4,
    ) -> None:
        """初始化时预计算多相滤波器系数。

        Args:
            orig_sr: 原始采样率
            target_sr: 目标采样率
            n_workers: 并行线程数
        """
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.n_workers = n_workers

        # 计算 up/down 比率
        common = gcd(orig_sr, target_sr)
        self.up = target_sr // common
        self.down = orig_sr // common

        # 预计算多相滤波器系数
        # 截止频率为 min(1/up, 1/down) * Nyquist
        max_rate = max(self.up, self.down)
        cutoff = 1.0 / max_rate
        # 滤波器长度：经验值，每个多相分支约 10 个系数
        n_taps = 2 * 10 * max_rate + 1
        self._filter = firwin(n_taps, cutoff, window="hamming") * self.up

        logger.info(
            "BatchResampler initialized: %dHz -> %dHz (up=%d, down=%d, "
            "filter_taps=%d, workers=%d)",
            orig_sr,
            target_sr,
            self.up,
            self.down,
            n_taps,
            n_workers,
        )

    def resample_single(self, audio_path: Path) -> bool:
        """重采样单个文件。检查当前采样率，跳过已完成的文件。

        Args:
            audio_path: 音频文件路径

        Returns:
            True 表示成功（含跳过），False 表示失败
        """
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")

            # 多声道转单声道
            if audio.ndim > 1:
                audio = audio[:, 0]

            # 已是目标采样率，跳过
            if sr == self.target_sr:
                logger.debug("Skipping %s: already at %dHz", audio_path.name, sr)
                return True

            # 使用预计算滤波器执行重采样
            resampled = upfirdn(self._filter, audio, self.up, self.down)

            # upfirdn 会引入滤波器延迟，需要裁剪
            # 滤波器延迟 = (len(filter) - 1) / 2
            delay = (len(self._filter) - 1) // 2
            # 计算期望输出长度
            expected_len = int(np.ceil(len(audio) * self.up / self.down))
            # 从延迟位置开始截取期望长度
            start = delay // self.down
            resampled = resampled[start : start + expected_len]

            resampled = resampled.astype(np.float32)

            # 覆盖写回原文件
            sf.write(str(audio_path), resampled, self.target_sr)
            logger.debug("Resampled %s: %dHz -> %dHz", audio_path.name, sr, self.target_sr)
            return True

        except Exception as e:
            logger.error("Failed to resample %s: %s", audio_path, e)
            return False

    def resample_files(
        self,
        audio_paths: list[Path],
        statuses: dict[str, str] | None = None,
    ) -> dict[str, bool]:
        """批量重采样音频文件（原地覆盖写回）。

        跳过已经是目标采样率的文件（增量重采样）。

        Args:
            audio_paths: 音频文件路径列表
            statuses: sample_id → 重采样状态映射（可选，用于增量跳过）

        Returns:
            sample_id → 是否成功 的映射
        """
        results: dict[str, bool] = {}

        if not audio_paths:
            return results

        # 过滤已完成的文件
        paths_to_process: list[Path] = []
        for path in audio_paths:
            sample_id = path.stem  # e.g. "preset_00000_C3_v80"
            if statuses and statuses.get(sample_id) == "completed":
                logger.debug("Skipping %s: already completed", sample_id)
                results[sample_id] = True
                continue
            paths_to_process.append(path)

        logger.info(
            "Resampling %d files (%d skipped) with %d workers",
            len(paths_to_process),
            len(audio_paths) - len(paths_to_process),
            self.n_workers,
        )

        # 多线程并行执行
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.resample_single, path): path
                for path in paths_to_process
            }
            for future in futures:
                path = futures[future]
                sample_id = path.stem
                try:
                    success = future.result()
                    results[sample_id] = success
                    if not success:
                        logger.error("Resample failed for %s", sample_id)
                except Exception as e:
                    logger.error(
                        "Resample failed for %s: %s", sample_id, e
                    )
                    results[sample_id] = False

        return results

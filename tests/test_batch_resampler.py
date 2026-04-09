"""
BatchResampler 属性测试与单元测试。

Feature: robust-checkpoint-and-resampling
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib import Path

from src.batch_resampler import BatchResampler


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Random audio duration between 0.1s and 0.5s (short for speed)
audio_duration_st = st.floats(min_value=0.1, max_value=0.5)


def _make_wav(path: Path, sr: int, duration: float) -> Path:
    """Create a WAV file with random audio data at the given sample rate."""
    n_samples = max(int(sr * duration), 1)
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(n_samples).astype(np.float32) * 0.5
    sf.write(str(path), audio, sr)
    return path


# ---------------------------------------------------------------------------
# Property 9: 重采样输出采样率正确性
# ---------------------------------------------------------------------------


class TestProperty9ResampleOutputSampleRate:
    """Feature: robust-checkpoint-and-resampling, Property 9: 重采样输出采样率正确性

    **Validates: Requirements 5.3**
    """

    @given(duration=st.floats(min_value=0.1, max_value=2.0))
    @settings(max_examples=100, deadline=None)
    def test_resampled_file_has_target_sample_rate(self, tmp_path_factory, duration):
        """对任意长度的 44100Hz WAV 文件重采样后，输出采样率应为 16000Hz。

        **Validates: Requirements 5.3**
        """
        tmp_path = tmp_path_factory.mktemp("resample_sr")
        wav_path = tmp_path / "test.wav"

        # Generate random-length 44100Hz audio
        n_samples = max(int(44100 * duration), 1)
        rng = np.random.default_rng()
        audio = rng.standard_normal(n_samples).astype(np.float32) * 0.5
        sf.write(str(wav_path), audio, 44100)

        # Resample
        resampler = BatchResampler(orig_sr=44100, target_sr=16000, n_workers=1)
        success = resampler.resample_single(wav_path)

        assert success, "Resampling should succeed"

        # Verify output sample rate
        info = sf.info(str(wav_path))
        assert info.samplerate == 16000, (
            f"Expected 16000Hz, got {info.samplerate}Hz"
        )


# ---------------------------------------------------------------------------
# Property 10: 重采样错误隔离
# ---------------------------------------------------------------------------


class TestProperty10ResampleErrorIsolation:
    """Feature: robust-checkpoint-and-resampling, Property 10: 重采样错误隔离

    **Validates: Requirements 5.4**
    """

    @given(
        n_valid=st.integers(min_value=1, max_value=5),
        n_invalid=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_files_succeed_despite_invalid_files(
        self, tmp_path_factory, n_valid, n_invalid
    ):
        """混合有效和无效文件时，有效文件应全部成功，无效文件标记为失败。

        **Validates: Requirements 5.4**
        """
        tmp_path = tmp_path_factory.mktemp("resample_err")

        # Create valid WAV files
        valid_paths: list[Path] = []
        for i in range(n_valid):
            wav_path = tmp_path / f"valid_{i:03d}.wav"
            n_samples = 4410  # 0.1s at 44100Hz
            audio = np.zeros(n_samples, dtype=np.float32)
            sf.write(str(wav_path), audio, 44100)
            valid_paths.append(wav_path)

        # Create invalid file paths (non-existent or not WAV)
        invalid_paths: list[Path] = []
        for i in range(n_invalid):
            invalid_path = tmp_path / f"invalid_{i:03d}.wav"
            # Don't create the file — it doesn't exist
            invalid_paths.append(invalid_path)

        all_paths = valid_paths + invalid_paths

        resampler = BatchResampler(orig_sr=44100, target_sr=16000, n_workers=2)
        results = resampler.resample_files(all_paths)

        # All valid files should succeed
        for vp in valid_paths:
            sample_id = vp.stem
            assert results.get(sample_id) is True, (
                f"Valid file {sample_id} should succeed, got {results.get(sample_id)}"
            )

        # All invalid files should be marked as failed
        for ip in invalid_paths:
            sample_id = ip.stem
            assert results.get(sample_id) is False, (
                f"Invalid file {sample_id} should fail, got {results.get(sample_id)}"
            )


# ---------------------------------------------------------------------------
# Property 11: 增量重采样跳过已完成文件
# ---------------------------------------------------------------------------


class TestProperty11IncrementalResampleSkip:
    """Feature: robust-checkpoint-and-resampling, Property 11: 增量重采样跳过已完成文件

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @given(
        n_done=st.integers(min_value=1, max_value=5),
        n_todo=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_skips_already_target_sr_files(
        self, tmp_path_factory, n_done, n_todo
    ):
        """已是 16000Hz 的文件应被跳过，仅对 44100Hz 文件执行重采样。

        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        tmp_path = tmp_path_factory.mktemp("resample_skip")

        # Files already at 16000Hz (should be skipped)
        done_paths: list[Path] = []
        for i in range(n_done):
            wav_path = tmp_path / f"done_{i:03d}.wav"
            n_samples = 1600  # 0.1s at 16000Hz
            audio = np.ones(n_samples, dtype=np.float32) * 0.3
            sf.write(str(wav_path), audio, 16000)
            done_paths.append(wav_path)

        # Files at 44100Hz (should be resampled)
        todo_paths: list[Path] = []
        for i in range(n_todo):
            wav_path = tmp_path / f"todo_{i:03d}.wav"
            n_samples = 4410  # 0.1s at 44100Hz
            audio = np.ones(n_samples, dtype=np.float32) * 0.3
            sf.write(str(wav_path), audio, 44100)
            todo_paths.append(wav_path)

        # Record original content of done files for comparison
        done_originals = {}
        for dp in done_paths:
            data, sr = sf.read(str(dp), dtype="float32")
            done_originals[dp.stem] = (data.copy(), sr)

        all_paths = done_paths + todo_paths

        resampler = BatchResampler(orig_sr=44100, target_sr=16000, n_workers=2)
        results = resampler.resample_files(all_paths)

        # All files should succeed
        for p in all_paths:
            assert results.get(p.stem) is True, (
                f"File {p.stem} should succeed"
            )

        # Done files should remain unchanged (still 16000Hz, same data)
        for dp in done_paths:
            data, sr = sf.read(str(dp), dtype="float32")
            assert sr == 16000, (
                f"Done file {dp.stem} should still be 16000Hz, got {sr}"
            )
            orig_data, _ = done_originals[dp.stem]
            np.testing.assert_array_equal(
                data, orig_data,
                err_msg=f"Done file {dp.stem} content should be unchanged",
            )

        # Todo files should now be 16000Hz
        for tp in todo_paths:
            info = sf.info(str(tp))
            assert info.samplerate == 16000, (
                f"Todo file {tp.stem} should be 16000Hz after resample, "
                f"got {info.samplerate}"
            )


# ---------------------------------------------------------------------------
# Unit Tests: BatchResampler 边界情况 (Task 7.5)
# ---------------------------------------------------------------------------


class TestBatchResamplerUnit:
    """BatchResampler 边界情况单元测试。

    Requirements: 5.1, 5.2
    """

    # -- 1. 空文件列表 --

    def test_empty_file_list_returns_empty_dict(self):
        """空文件列表应返回空字典。"""
        resampler = BatchResampler(orig_sr=44100, target_sr=16000, n_workers=2)
        results = resampler.resample_files([])
        assert results == {}

    # -- 2. 预计算滤波器系数正确性 --

    def test_up_down_ratio_correct(self):
        """up/down 比率应正确反映采样率转换。

        44100 -> 16000: gcd(44100, 16000) = 100
        up = 16000/100 = 160, down = 44100/100 = 441
        """
        resampler = BatchResampler(orig_sr=44100, target_sr=16000)
        assert resampler.up == 160
        assert resampler.down == 441

    def test_up_down_ratio_upsample(self):
        """上采样时 up/down 比率应正确。

        16000 -> 44100: gcd(16000, 44100) = 100
        up = 44100/100 = 441, down = 16000/100 = 160
        """
        resampler = BatchResampler(orig_sr=16000, target_sr=44100)
        assert resampler.up == 441
        assert resampler.down == 160

    def test_filter_length_positive(self):
        """滤波器长度应大于 0。"""
        resampler = BatchResampler(orig_sr=44100, target_sr=16000)
        assert len(resampler._filter) > 0

    def test_filter_is_numpy_array(self):
        """滤波器系数应为 numpy 数组。"""
        resampler = BatchResampler(orig_sr=44100, target_sr=16000)
        assert isinstance(resampler._filter, np.ndarray)

    def test_filter_coefficients_finite(self):
        """滤波器系数应全部为有限值。"""
        resampler = BatchResampler(orig_sr=44100, target_sr=16000)
        assert np.all(np.isfinite(resampler._filter))

    # -- 3. 单线程和多线程模式一致性 --

    def test_single_vs_multi_thread_consistency(self, tmp_path):
        """单线程和多线程模式应产生相同的重采样结果。"""
        # Create test WAV files
        n_files = 4
        paths_single = []
        paths_multi = []

        rng = np.random.default_rng(123)
        for i in range(n_files):
            audio = rng.standard_normal(4410).astype(np.float32) * 0.5

            # Single-thread copy
            p1 = tmp_path / "single" / f"file_{i:03d}.wav"
            p1.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p1), audio, 44100)
            paths_single.append(p1)

            # Multi-thread copy
            p2 = tmp_path / "multi" / f"file_{i:03d}.wav"
            p2.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p2), audio, 44100)
            paths_multi.append(p2)

        # Resample with single thread
        resampler_single = BatchResampler(orig_sr=44100, target_sr=16000, n_workers=1)
        results_single = resampler_single.resample_files(paths_single)

        # Resample with multiple threads
        resampler_multi = BatchResampler(orig_sr=44100, target_sr=16000, n_workers=4)
        results_multi = resampler_multi.resample_files(paths_multi)

        # Both should succeed
        for i in range(n_files):
            sid = f"file_{i:03d}"
            assert results_single[sid] is True
            assert results_multi[sid] is True

        # Output audio should be identical
        for p1, p2 in zip(paths_single, paths_multi):
            data1, sr1 = sf.read(str(p1), dtype="float32")
            data2, sr2 = sf.read(str(p2), dtype="float32")
            assert sr1 == sr2 == 16000
            np.testing.assert_array_equal(
                data1, data2,
                err_msg=f"Single vs multi thread mismatch for {p1.stem}",
            )

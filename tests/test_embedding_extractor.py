"""
EmbeddingExtractor 单元测试。

使用 mock 对象替代实际的 MuQ 模型，测试核心逻辑：
- EmbeddingResult dataclass
- ModelLoadError 异常
- extract 方法（含采样率重采样）
- extract_batch 方法
- save/load round-trip
- 错误处理
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.embedding_extractor import (
    EmbeddingExtractor,
    EmbeddingResult,
    ModelLoadError,
    DEFAULT_EMBEDDING_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1024
TIME_FRAMES = 50


def _make_mock_output(batch_size: int = 1, time_frames: int = TIME_FRAMES):
    """Create a mock MuQ model output with last_hidden_state."""
    import torch

    hidden = torch.randn(batch_size, time_frames, EMBEDDING_DIM)
    output = MagicMock()
    output.last_hidden_state = hidden
    output.hidden_states = [hidden]
    return output


def _create_wav(path: Path, sample_rate: int = 44100, duration: float = 1.0):
    """Write a short sine-wave WAV file."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sample_rate)
    return path


def _create_mock_extractor(
    device: str = "cpu",
    target_sample_rate: int = 16000,
) -> EmbeddingExtractor:
    """Create an EmbeddingExtractor with a mocked MuQ model (bypasses __init__)."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.side_effect = lambda *args, **kwargs: _make_mock_output()

    extractor = EmbeddingExtractor.__new__(EmbeddingExtractor)
    extractor._device = device
    extractor._target_sample_rate = target_sample_rate
    extractor._model_repo = "OpenMuQ/MuQ-large-msd-iter"
    extractor._model = mock_model

    return extractor


# ---------------------------------------------------------------------------
# Test: EmbeddingResult dataclass
# ---------------------------------------------------------------------------

class TestEmbeddingResult:
    def test_creation_with_defaults(self):
        result = EmbeddingResult()
        assert result.embeddings == {}
        assert result.dimension == DEFAULT_EMBEDDING_DIM

    def test_creation_with_values(self):
        emb = {"file1.wav": np.zeros(1024), "file2.wav": np.ones(1024)}
        result = EmbeddingResult(embeddings=emb, dimension=1024)
        assert len(result.embeddings) == 2
        assert result.dimension == 1024
        assert "file1.wav" in result.embeddings
        assert "file2.wav" in result.embeddings

    def test_no_shared_state_between_instances(self):
        r1 = EmbeddingResult()
        r2 = EmbeddingResult()
        r1.embeddings["test.wav"] = np.zeros(10)
        assert "test.wav" not in r2.embeddings


# ---------------------------------------------------------------------------
# Test: ModelLoadError exception
# ---------------------------------------------------------------------------

class TestModelLoadError:
    def test_with_model_path(self):
        err = ModelLoadError("/path/to/model", "connection timeout")
        assert "/path/to/model" in str(err)
        assert "connection timeout" in str(err)
        assert err.model_path == "/path/to/model"
        assert err.message == "connection timeout"

    def test_without_model_path(self):
        err = ModelLoadError(None, "out of memory")
        assert "out of memory" in str(err)
        assert err.model_path is None

    def test_is_exception(self):
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("repo", "test error")


# ---------------------------------------------------------------------------
# Test: EmbeddingExtractor.__init__ (model loading)
# ---------------------------------------------------------------------------

class TestEmbeddingExtractorInit:
    def test_model_load_failure_raises_model_load_error(self):
        """ModelLoadError should be raised when model loading fails."""
        mock_muq_module = MagicMock()
        mock_muq_module.MuQ.from_pretrained.side_effect = RuntimeError(
            "download failed"
        )

        with patch.dict("sys.modules", {"muq": mock_muq_module}):
            with pytest.raises(ModelLoadError, match="download failed"):
                EmbeddingExtractor(model_path="bad/model", device="cpu")

    def test_successful_init_with_mock(self):
        """Extractor should initialize successfully with a mock model."""
        mock_muq_module = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_muq_module.MuQ.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {"muq": mock_muq_module}):
            extractor = EmbeddingExtractor(device="cpu")
            assert extractor._device == "cpu"
            assert extractor._model is mock_model


# ---------------------------------------------------------------------------
# Test: extract returns correct shape
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_correct_shape(self, tmp_path: Path):
        """extract should return a 1D array of shape (embedding_dim,)."""
        wav_path = _create_wav(tmp_path / "test.wav", sample_rate=16000)
        extractor = _create_mock_extractor(target_sample_rate=16000)

        embedding = extractor.extract(wav_path)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (EMBEDDING_DIM,)

    def test_handles_sample_rate_mismatch(self, tmp_path: Path):
        """extract should auto-resample when sample rate doesn't match."""
        # Create WAV at 44100 Hz, extractor expects 16000 Hz
        wav_path = _create_wav(tmp_path / "test_44k.wav", sample_rate=44100)
        extractor = _create_mock_extractor(target_sample_rate=16000)

        embedding = extractor.extract(wav_path)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (EMBEDDING_DIM,)

        # Verify the model was called (resampling happened internally)
        extractor._model.assert_called_once()

    def test_matching_sample_rate_no_resample(self, tmp_path: Path):
        """extract should not resample when sample rates match."""
        wav_path = _create_wav(tmp_path / "test_16k.wav", sample_rate=16000)
        extractor = _create_mock_extractor(target_sample_rate=16000)

        embedding = extractor.extract(wav_path)
        assert embedding.shape == (EMBEDDING_DIM,)

    def test_file_not_found_raises(self):
        """extract should raise FileNotFoundError for missing files."""
        extractor = _create_mock_extractor()
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            extractor.extract(Path("/nonexistent/audio.wav"))

    def test_invalid_audio_raises_value_error(self, tmp_path: Path):
        """extract should raise ValueError for incompatible audio."""
        bad_file = tmp_path / "bad.wav"
        bad_file.write_text("not audio data")
        extractor = _create_mock_extractor()

        with pytest.raises(ValueError, match="Incompatible audio format"):
            extractor.extract(bad_file)


# ---------------------------------------------------------------------------
# Test: extract_batch
# ---------------------------------------------------------------------------

class TestExtractBatch:
    def test_processes_all_wav_files(self, tmp_path: Path):
        """extract_batch should process all .wav files in directory."""
        for i in range(3):
            _create_wav(tmp_path / f"audio_{i}.wav", sample_rate=16000)

        extractor = _create_mock_extractor(target_sample_rate=16000)
        result = extractor.extract_batch(tmp_path)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        for name in ["audio_0.wav", "audio_1.wav", "audio_2.wav"]:
            assert name in result.embeddings
            assert result.embeddings[name].shape == (EMBEDDING_DIM,)

    def test_empty_directory(self, tmp_path: Path):
        """extract_batch should return empty result for empty directory."""
        extractor = _create_mock_extractor()
        result = extractor.extract_batch(tmp_path)

        assert len(result.embeddings) == 0
        assert result.dimension == DEFAULT_EMBEDDING_DIM

    def test_ignores_non_wav_files(self, tmp_path: Path):
        """extract_batch should only process .wav files."""
        _create_wav(tmp_path / "audio.wav", sample_rate=16000)
        (tmp_path / "readme.txt").write_text("not audio")
        (tmp_path / "preset.vital").write_text("{}")

        extractor = _create_mock_extractor(target_sample_rate=16000)
        result = extractor.extract_batch(tmp_path)

        assert len(result.embeddings) == 1
        assert "audio.wav" in result.embeddings

    def test_continues_on_individual_failure(self, tmp_path: Path):
        """extract_batch should skip failed files and continue."""
        _create_wav(tmp_path / "good.wav", sample_rate=16000)
        bad_file = tmp_path / "bad.wav"
        bad_file.write_text("not audio data")
        _create_wav(tmp_path / "also_good.wav", sample_rate=16000)

        extractor = _create_mock_extractor(target_sample_rate=16000)
        result = extractor.extract_batch(tmp_path)

        # bad.wav should be skipped, the other two should succeed
        assert len(result.embeddings) == 2
        assert "good.wav" in result.embeddings
        assert "also_good.wav" in result.embeddings
        assert "bad.wav" not in result.embeddings

    def test_result_dimension_matches_embedding(self, tmp_path: Path):
        """EmbeddingResult dimension should match actual embedding size."""
        _create_wav(tmp_path / "test.wav", sample_rate=16000)
        extractor = _create_mock_extractor(target_sample_rate=16000)
        result = extractor.extract_batch(tmp_path)

        assert result.dimension == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Test: save and load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_npz_file(self, tmp_path: Path):
        """save should create a .npz file."""
        result = EmbeddingResult(
            embeddings={"a.wav": np.random.randn(1024).astype(np.float32)},
            dimension=1024,
        )
        output_path = tmp_path / "embeddings.npz"

        extractor = _create_mock_extractor()
        extractor.save(result, output_path)

        # np.savez appends .npz if not present; our path already has it
        assert output_path.exists() or (tmp_path / "embeddings.npz.npz").exists()

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """save should create parent directories if they don't exist."""
        result = EmbeddingResult(
            embeddings={"a.wav": np.zeros(1024)},
            dimension=1024,
        )
        output_path = tmp_path / "nested" / "dir" / "embeddings.npz"

        extractor = _create_mock_extractor()
        extractor.save(result, output_path)

        assert output_path.parent.exists()

    def test_round_trip_preserves_data(self, tmp_path: Path):
        """save then load should preserve all embedding data."""
        original = EmbeddingResult(
            embeddings={
                "file1.wav": np.random.randn(1024).astype(np.float32),
                "file2.wav": np.random.randn(1024).astype(np.float32),
                "file3.wav": np.random.randn(1024).astype(np.float32),
            },
            dimension=1024,
        )
        output_path = tmp_path / "test_roundtrip.npz"

        extractor = _create_mock_extractor()
        extractor.save(original, output_path)

        loaded = EmbeddingExtractor.load(output_path)

        assert len(loaded.embeddings) == len(original.embeddings)
        assert loaded.dimension == original.dimension
        for key in original.embeddings:
            assert key in loaded.embeddings
            np.testing.assert_array_almost_equal(
                loaded.embeddings[key], original.embeddings[key]
            )

    def test_round_trip_with_many_files(self, tmp_path: Path):
        """Round-trip should work with many embedding entries."""
        embs = {
            f"audio_{i:03d}.wav": np.random.randn(1024).astype(np.float32)
            for i in range(20)
        }
        original = EmbeddingResult(embeddings=embs, dimension=1024)
        output_path = tmp_path / "many.npz"

        extractor = _create_mock_extractor()
        extractor.save(original, output_path)

        loaded = EmbeddingExtractor.load(output_path)
        assert len(loaded.embeddings) == 20
        for key in original.embeddings:
            np.testing.assert_array_almost_equal(
                loaded.embeddings[key], original.embeddings[key]
            )

    def test_save_with_correct_keys(self, tmp_path: Path):
        """Saved .npz should contain the correct filename keys."""
        result = EmbeddingResult(
            embeddings={
                "chorus_on_1.0.wav": np.zeros(1024),
                "chorus_on_0.0.wav": np.ones(1024),
            },
            dimension=1024,
        )
        output_path = tmp_path / "keys_test.npz"

        extractor = _create_mock_extractor()
        extractor.save(result, output_path)

        data = np.load(output_path)
        assert set(data.files) == {"chorus_on_1.0.wav", "chorus_on_0.0.wav"}


# ---------------------------------------------------------------------------
# Test: Resampling
# ---------------------------------------------------------------------------

class TestResample:
    def test_resample_changes_length(self):
        """Resampling from 44100 to 16000 should change array length."""
        audio = np.random.randn(44100).astype(np.float32)  # 1 second at 44100
        resampled = EmbeddingExtractor._resample(audio, 44100, 16000)

        expected_length = 16000  # 1 second at 16000
        # Allow small tolerance due to filter effects
        assert abs(len(resampled) - expected_length) <= 10

    def test_resample_same_rate_returns_same(self):
        """Resampling with same rate should return identical array."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = EmbeddingExtractor._resample(audio, 16000, 16000)

        np.testing.assert_array_equal(resampled, audio)

    def test_resample_output_is_float32(self):
        """Resampled audio should be float32."""
        audio = np.random.randn(44100).astype(np.float32)
        resampled = EmbeddingExtractor._resample(audio, 44100, 16000)

        assert resampled.dtype == np.float32

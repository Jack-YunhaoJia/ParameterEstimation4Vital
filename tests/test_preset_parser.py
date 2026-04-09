"""
PresetParser 单元测试。

验证 VitalPreset 数据类、PresetParser 的 parse/serialize/validate_effect_switches 方法，
以及 PresetParseError 自定义异常。
"""

import json
from pathlib import Path

import pytest

from src.preset_parser import PresetParseError, PresetParser, VitalPreset


@pytest.fixture
def parser() -> PresetParser:
    return PresetParser()


@pytest.fixture
def sample_vital_json() -> dict:
    """构造一个最小但完整的 Vital 预设 JSON 结构。"""
    settings = {
        "chorus_on": 0.0,
        "compressor_on": 0.0,
        "delay_on": 0.0,
        "distortion_on": 0.0,
        "eq_on": 0.0,
        "flanger_on": 0.0,
        "phaser_on": 0.0,
        "reverb_on": 0.0,
        "filter_fx_on": 0.0,
        "osc_1_on": 1.0,
        "osc_1_level": 0.707,
        "filter_1_on": 1.0,
        "filter_1_cutoff": 60.0,
        "modulations": [
            {
                "source": "env_2",
                "destination": "filter_1_cutoff",
            },
            {
                "source": "",
                "destination": "",
            },
        ],
        "wavetables": [{"name": "Init"}],
    }
    return {
        "author": "Test",
        "comments": "",
        "preset_name": "Test Preset",
        "settings": settings,
    }


@pytest.fixture
def sample_vital_file(tmp_path: Path, sample_vital_json: dict) -> Path:
    """写入一个临时 .vital 文件。"""
    filepath = tmp_path / "test.vital"
    filepath.write_text(json.dumps(sample_vital_json), encoding="utf-8")
    return filepath


class TestVitalPreset:
    def test_dataclass_creation(self):
        preset = VitalPreset(settings={"osc_1_on": 1.0})
        assert preset.settings == {"osc_1_on": 1.0}
        assert preset.modulations == []
        assert preset.extra == {}

    def test_dataclass_with_all_fields(self):
        preset = VitalPreset(
            settings={"chorus_on": 0.0},
            modulations=[{"source": "env_2", "destination": "filter_1_cutoff"}],
            extra={"author": "Test"},
        )
        assert preset.settings["chorus_on"] == 0.0
        assert len(preset.modulations) == 1
        assert preset.extra["author"] == "Test"


class TestPresetParserParse:
    def test_parse_valid_file(
        self, parser: PresetParser, sample_vital_file: Path
    ):
        preset = parser.parse(sample_vital_file)

        # settings 不应包含 modulations 和 wavetables
        assert "modulations" not in preset.settings
        assert "wavetables" not in preset.settings

        # 参数应在 settings 中
        assert preset.settings["osc_1_on"] == 1.0
        assert preset.settings["chorus_on"] == 0.0
        assert preset.settings["filter_1_cutoff"] == 60.0

    def test_parse_extracts_modulations(
        self, parser: PresetParser, sample_vital_file: Path
    ):
        preset = parser.parse(sample_vital_file)
        assert len(preset.modulations) == 2
        assert preset.modulations[0]["source"] == "env_2"

    def test_parse_extracts_wavetables_to_extra(
        self, parser: PresetParser, sample_vital_file: Path
    ):
        preset = parser.parse(sample_vital_file)
        assert "wavetables" in preset.extra
        assert preset.extra["wavetables"] == [{"name": "Init"}]

    def test_parse_extracts_toplevel_keys_to_extra(
        self, parser: PresetParser, sample_vital_file: Path
    ):
        preset = parser.parse(sample_vital_file)
        assert preset.extra["author"] == "Test"
        assert preset.extra["preset_name"] == "Test Preset"

    def test_parse_invalid_json(self, parser: PresetParser, tmp_path: Path):
        filepath = tmp_path / "bad.vital"
        filepath.write_text("not json at all {{{", encoding="utf-8")
        with pytest.raises(PresetParseError) as exc_info:
            parser.parse(filepath)
        assert "Invalid JSON" in str(exc_info.value)
        assert str(filepath) in str(exc_info.value)

    def test_parse_missing_settings_key(
        self, parser: PresetParser, tmp_path: Path
    ):
        filepath = tmp_path / "no_settings.vital"
        filepath.write_text(json.dumps({"author": "Test"}), encoding="utf-8")
        with pytest.raises(PresetParseError) as exc_info:
            parser.parse(filepath)
        assert "settings" in str(exc_info.value)

    def test_parse_nonexistent_file(self, parser: PresetParser, tmp_path: Path):
        filepath = tmp_path / "nonexistent.vital"
        with pytest.raises(PresetParseError) as exc_info:
            parser.parse(filepath)
        assert "Cannot read file" in str(exc_info.value)

    def test_parse_json_root_not_object(
        self, parser: PresetParser, tmp_path: Path
    ):
        filepath = tmp_path / "array.vital"
        filepath.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(PresetParseError) as exc_info:
            parser.parse(filepath)
        assert "object" in str(exc_info.value)

    def test_parse_settings_not_dict(
        self, parser: PresetParser, tmp_path: Path
    ):
        filepath = tmp_path / "bad_settings.vital"
        filepath.write_text(
            json.dumps({"settings": "not a dict"}), encoding="utf-8"
        )
        with pytest.raises(PresetParseError) as exc_info:
            parser.parse(filepath)
        assert "dictionary" in str(exc_info.value)


class TestPresetParserSerialize:
    def test_serialize_creates_file(
        self, parser: PresetParser, tmp_path: Path
    ):
        preset = VitalPreset(
            settings={"osc_1_on": 1.0, "chorus_on": 0.0},
            modulations=[{"source": "env_2", "destination": "filter_1_cutoff"}],
            extra={"author": "Test", "wavetables": [{"name": "Init"}]},
        )
        filepath = tmp_path / "output.vital"
        parser.serialize(preset, filepath)
        assert filepath.exists()

    def test_serialize_produces_valid_json(
        self, parser: PresetParser, tmp_path: Path
    ):
        preset = VitalPreset(
            settings={"osc_1_on": 1.0},
            modulations=[],
            extra={"author": "Test"},
        )
        filepath = tmp_path / "output.vital"
        parser.serialize(preset, filepath)
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert "settings" in raw

    def test_serialize_puts_modulations_inside_settings(
        self, parser: PresetParser, tmp_path: Path
    ):
        preset = VitalPreset(
            settings={"osc_1_on": 1.0},
            modulations=[{"source": "lfo_1", "destination": "osc_1_level"}],
            extra={},
        )
        filepath = tmp_path / "output.vital"
        parser.serialize(preset, filepath)
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        assert "modulations" in raw["settings"]
        assert len(raw["settings"]["modulations"]) == 1

    def test_serialize_puts_wavetables_inside_settings(
        self, parser: PresetParser, tmp_path: Path
    ):
        preset = VitalPreset(
            settings={"osc_1_on": 1.0},
            modulations=[],
            extra={"wavetables": [{"name": "Saw"}]},
        )
        filepath = tmp_path / "output.vital"
        parser.serialize(preset, filepath)
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        assert "wavetables" in raw["settings"]
        assert raw["settings"]["wavetables"] == [{"name": "Saw"}]

    def test_serialize_creates_parent_dirs(
        self, parser: PresetParser, tmp_path: Path
    ):
        preset = VitalPreset(settings={"osc_1_on": 1.0})
        filepath = tmp_path / "nested" / "dir" / "output.vital"
        parser.serialize(preset, filepath)
        assert filepath.exists()


class TestPresetParserRoundTrip:
    def test_round_trip_preserves_settings(
        self, parser: PresetParser, sample_vital_file: Path, tmp_path: Path
    ):
        """parse → serialize → parse 应产生等价结果。"""
        original = parser.parse(sample_vital_file)
        roundtrip_path = tmp_path / "roundtrip.vital"
        parser.serialize(original, roundtrip_path)
        restored = parser.parse(roundtrip_path)

        assert original.settings == restored.settings

    def test_round_trip_preserves_modulations(
        self, parser: PresetParser, sample_vital_file: Path, tmp_path: Path
    ):
        original = parser.parse(sample_vital_file)
        roundtrip_path = tmp_path / "roundtrip.vital"
        parser.serialize(original, roundtrip_path)
        restored = parser.parse(roundtrip_path)

        assert original.modulations == restored.modulations

    def test_round_trip_preserves_extra(
        self, parser: PresetParser, sample_vital_file: Path, tmp_path: Path
    ):
        original = parser.parse(sample_vital_file)
        roundtrip_path = tmp_path / "roundtrip.vital"
        parser.serialize(original, roundtrip_path)
        restored = parser.parse(roundtrip_path)

        assert original.extra == restored.extra


class TestPresetParserValidateEffectSwitches:
    def test_validate_all_switches_present(self, parser: PresetParser):
        settings = {switch: 0.0 for switch in PresetParser.EFFECT_SWITCHES}
        preset = VitalPreset(settings=settings)
        assert parser.validate_effect_switches(preset) is True

    def test_validate_missing_one_switch(self, parser: PresetParser):
        settings = {switch: 0.0 for switch in PresetParser.EFFECT_SWITCHES}
        del settings["chorus_on"]
        preset = VitalPreset(settings=settings)
        assert parser.validate_effect_switches(preset) is False

    def test_validate_empty_settings(self, parser: PresetParser):
        preset = VitalPreset(settings={})
        assert parser.validate_effect_switches(preset) is False

    def test_validate_with_extra_keys(self, parser: PresetParser):
        settings = {switch: 1.0 for switch in PresetParser.EFFECT_SWITCHES}
        settings["osc_1_on"] = 1.0
        preset = VitalPreset(settings=settings)
        assert parser.validate_effect_switches(preset) is True


class TestPresetParseError:
    def test_error_contains_filepath(self):
        err = PresetParseError("/path/to/file.vital", "bad format")
        assert "/path/to/file.vital" in str(err)
        assert err.filepath == "/path/to/file.vital"

    def test_error_contains_message(self):
        err = PresetParseError("test.vital", "Missing settings key")
        assert "Missing settings key" in str(err)
        assert err.message == "Missing settings key"

    def test_error_is_exception(self):
        err = PresetParseError("test.vital", "test")
        assert isinstance(err, Exception)

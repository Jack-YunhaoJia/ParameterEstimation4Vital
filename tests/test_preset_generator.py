"""
PresetGenerator 单元测试。

验证预设生成的核心功能：Base_Patch 创建、效果器变体生成、
批量生成、以及无效输入处理。
"""

import pytest
from pathlib import Path

from src.preset_parser import PresetParser, VitalPreset
from src.preset_generator import PresetGenerator, _load_default_template


@pytest.fixture
def parser():
    return PresetParser()


@pytest.fixture
def template():
    return _load_default_template()


@pytest.fixture
def generator(parser, template):
    return PresetGenerator(parser, template)


class TestCreateBasePatch:
    """验证 create_base_patch 方法。"""

    def test_base_patch_osc_1_on(self, generator):
        """需求 1.2: osc_1_on 应为 1.0"""
        base = generator.create_base_patch()
        assert base.settings["osc_1_on"] == 1.0

    def test_base_patch_filter_1_on(self, generator):
        """需求 1.2: filter_1_on 应为 1.0"""
        base = generator.create_base_patch()
        assert base.settings["filter_1_on"] == 1.0

    def test_base_patch_all_effects_off(self, generator):
        """需求 1.2: Base_Patch 中所有效果器应关闭"""
        base = generator.create_base_patch()
        for switch in PresetParser.EFFECT_SWITCHES:
            assert base.settings[switch] == 0.0, (
                f"{switch} should be 0.0 in base patch"
            )

    def test_base_patch_has_settings(self, generator):
        """需求 1.1: 生成的预设应包含 settings 字典"""
        base = generator.create_base_patch()
        assert isinstance(base, VitalPreset)
        assert isinstance(base.settings, dict)
        assert len(base.settings) > 0

    def test_base_patch_contains_all_effect_switches(self, generator):
        """需求 1.1: settings 应包含所有 9 个 Effect_Switch 键"""
        base = generator.create_base_patch()
        for switch in PresetParser.EFFECT_SWITCHES:
            assert switch in base.settings

    def test_base_patch_has_772_params(self, generator):
        """需求 1.1: settings 应包含 772 个参数"""
        base = generator.create_base_patch()
        assert len(base.settings) == 772


class TestCreateEffectVariant:
    """验证 create_effect_variant 方法。"""

    def test_variant_modifies_only_target_switch(self, generator):
        """需求 1.3, 1.5: 变体仅在目标 Effect_Switch 上与 Base_Patch 不同"""
        base = generator.create_base_patch()
        variant = generator.create_effect_variant("chorus_on", 1.0)

        for key, value in base.settings.items():
            if key == "chorus_on":
                assert variant.settings[key] == 1.0
            else:
                assert variant.settings[key] == value, (
                    f"Parameter '{key}' should be unchanged"
                )

    def test_variant_off_state(self, generator):
        """需求 1.3: 关闭状态变体"""
        variant = generator.create_effect_variant("reverb_on", 0.0)
        assert variant.settings["reverb_on"] == 0.0

    def test_variant_on_state(self, generator):
        """需求 1.3: 开启状态变体"""
        variant = generator.create_effect_variant("delay_on", 1.0)
        assert variant.settings["delay_on"] == 1.0

    def test_variant_is_deep_copy(self, generator):
        """变体应是深拷贝，修改不影响后续生成"""
        v1 = generator.create_effect_variant("chorus_on", 1.0)
        v2 = generator.create_effect_variant("chorus_on", 0.0)
        v1.settings["chorus_on"] = 999.0
        assert v2.settings["chorus_on"] == 0.0

    def test_all_effect_switches_accepted(self, generator):
        """需求 1.3: 所有 9 个效果器开关名都应被接受"""
        for switch in PresetParser.EFFECT_SWITCHES:
            variant = generator.create_effect_variant(switch, 1.0)
            assert variant.settings[switch] == 1.0

    def test_invalid_effect_name_raises_value_error(self, generator):
        """需求 1.6: 无效效果器名应抛出 ValueError"""
        with pytest.raises(ValueError) as exc_info:
            generator.create_effect_variant("invalid_effect", 1.0)

        error_msg = str(exc_info.value)
        assert "invalid_effect" in error_msg
        # 错误信息应包含有效列表
        for switch in PresetParser.EFFECT_SWITCHES:
            assert switch in error_msg

    def test_invalid_effect_name_empty_string(self, generator):
        """需求 1.6: 空字符串应抛出 ValueError"""
        with pytest.raises(ValueError):
            generator.create_effect_variant("", 1.0)

    def test_invalid_effect_name_similar_name(self, generator):
        """需求 1.6: 类似但不完全匹配的名称应抛出 ValueError"""
        with pytest.raises(ValueError):
            generator.create_effect_variant("chorus", 1.0)


class TestGenerateAllVariants:
    """验证 generate_all_variants 方法。"""

    def test_generates_19_files(self, generator, tmp_path):
        """需求 1.4: 应生成 18 个效果器变体 + 1 个 base_patch = 19 个文件"""
        files = generator.generate_all_variants(tmp_path / "presets")
        assert len(files) == 19

    def test_generates_base_patch_file(self, generator, tmp_path):
        """应生成 base_patch.vital"""
        output_dir = tmp_path / "presets"
        files = generator.generate_all_variants(output_dir)
        base_path = output_dir / "base_patch.vital"
        assert base_path in files
        assert base_path.exists()

    def test_generates_all_effect_variant_files(self, generator, tmp_path):
        """需求 1.4: 应为每个效果器生成开/关两个文件"""
        output_dir = tmp_path / "presets"
        files = generator.generate_all_variants(output_dir)

        for switch in PresetParser.EFFECT_SWITCHES:
            on_path = output_dir / f"{switch}_1.0.vital"
            off_path = output_dir / f"{switch}_0.0.vital"
            assert on_path.exists(), f"Missing: {on_path.name}"
            assert off_path.exists(), f"Missing: {off_path.name}"
            assert on_path in files
            assert off_path in files

    def test_generated_files_are_valid_json(self, generator, parser, tmp_path):
        """生成的文件应可被 PresetParser 解析"""
        output_dir = tmp_path / "presets"
        files = generator.generate_all_variants(output_dir)

        for filepath in files:
            preset = parser.parse(filepath)
            assert isinstance(preset, VitalPreset)

    def test_creates_output_directory(self, generator, tmp_path):
        """输出目录不存在时应自动创建"""
        output_dir = tmp_path / "nested" / "deep" / "presets"
        assert not output_dir.exists()
        generator.generate_all_variants(output_dir)
        assert output_dir.exists()

    def test_variant_files_differ_only_in_target_switch(
        self, generator, parser, tmp_path
    ):
        """需求 1.5: 变体文件与 base_patch 仅在目标开关上不同"""
        output_dir = tmp_path / "presets"
        generator.generate_all_variants(output_dir)

        base = parser.parse(output_dir / "base_patch.vital")

        for switch in PresetParser.EFFECT_SWITCHES:
            on_preset = parser.parse(output_dir / f"{switch}_1.0.vital")
            for key, value in base.settings.items():
                if key == switch:
                    assert on_preset.settings[key] == 1.0
                else:
                    assert on_preset.settings[key] == value


class TestPresetGeneratorInit:
    """验证 PresetGenerator 初始化。"""

    def test_default_template_loading(self, parser):
        """无模板时应使用内置默认模板"""
        gen = PresetGenerator(parser)
        base = gen.create_base_patch()
        assert base.settings["osc_1_on"] == 1.0
        assert len(base.settings) == 772

    def test_custom_template(self, parser):
        """自定义模板应被使用"""
        custom = {"osc_1_on": 0.5, "filter_1_on": 0.5}
        for switch in PresetParser.EFFECT_SWITCHES:
            custom[switch] = 0.0
        gen = PresetGenerator(parser, custom)
        base = gen.create_base_patch()
        # create_base_patch 会强制设置 osc_1_on=1.0
        assert base.settings["osc_1_on"] == 1.0
        assert base.settings["filter_1_on"] == 1.0

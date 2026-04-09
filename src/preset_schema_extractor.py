"""
语料驱动的参数 Schema 提取模块。

从真实 .vital 预设语料中自动提取参数 schema，包括参数分类、
编码方式、默认值、统计量和稳定排序。所有可训练字段统一落到
单个 float32 稠密监督矩阵中。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

from src.preset_parser import VitalPreset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binary / Categorical classification patterns
# ---------------------------------------------------------------------------

_BINARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".*_on$"),
    re.compile(r".*_bypass$"),
    re.compile(r"^legato$"),
    re.compile(r".*_smooth_interpolation$"),
    re.compile(r".*_midi_track$"),
    re.compile(r".*_random_phase$"),
]

_CATEGORICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".*_destination$"),
    re.compile(r".*_model$"),
    re.compile(r".*_style$"),
    re.compile(r".*_type$"),
    re.compile(r".*_mode$"),
    re.compile(r"^effect_chain_order$"),
    re.compile(r"^stereo_routing$"),
    re.compile(r".*_sync_type$"),
    re.compile(r".*_filter_input$"),
    re.compile(r".*_wavetable_id$"),
    re.compile(r".*_wavetable_source_type$"),
]

# Modulation slot field suffixes (k=1..64)
_MOD_SLOT_SUFFIXES: list[str] = [
    "source",
    "destination",
    "amount",
    "bypass",
    "bipolar",
    "power",
    "stereo",
]

# Wavetable fields per oscillator (i=1..3)
_WAVETABLE_FIELD_TEMPLATES: list[str] = [
    "osc_{i}_wavetable_id",
    "osc_{i}_wavetable_source_type",
]


def _classify_param(name: str) -> str:
    """Classify a parameter name as binary, categorical, or continuous."""
    for pat in _BINARY_PATTERNS:
        if pat.match(name):
            return "binary"
    for pat in _CATEGORICAL_PATTERNS:
        if pat.match(name):
            return "categorical"
    return "continuous"



@dataclass
class CorpusSchema:
    """语料驱动的参数 schema。

    Attributes:
        param_names: 稳定排序的参数名列表
        param_types: param_name → "binary" | "categorical" | "continuous"
        param_value_encoding: param_name → "identity" | "category_id"
        default_values: param_name → 编码后默认值
        corpus_min: param_name → 编码域中的语料最小值
        corpus_max: param_name → 编码域中的语料最大值
        presence_ratio: param_name → 出现比例
        category_values: categorical param → 稳定排序后的类别词表
    """

    param_names: list[str] = field(default_factory=list)
    param_types: dict[str, str] = field(default_factory=dict)
    param_value_encoding: dict[str, str] = field(default_factory=dict)
    default_values: dict[str, float] = field(default_factory=dict)
    corpus_min: dict[str, float] = field(default_factory=dict)
    corpus_max: dict[str, float] = field(default_factory=dict)
    presence_ratio: dict[str, float] = field(default_factory=dict)
    category_values: dict[str, list[str]] = field(default_factory=dict)


class PresetSchemaExtractor:
    """从预设语料中自动提取参数 schema。"""

    def __init__(self, inventory_path: Path | None = None) -> None:
        """初始化，可选加载 vital_param_inventory.json 作为上界参考。

        Args:
            inventory_path: vital_param_inventory.json 路径，为 None 则不使用 inventory。
        """
        self._inventory_order: list[str] | None = None
        self._inventory_set: set[str] | None = None

        if inventory_path is not None:
            inventory_path = Path(inventory_path)
            if inventory_path.exists():
                with open(inventory_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._inventory_order = list(data.get("continuous_params", []))
                self._inventory_set = set(self._inventory_order)
                logger.info(
                    "Loaded inventory with %d continuous params from %s",
                    len(self._inventory_order),
                    inventory_path,
                )
            else:
                logger.warning("Inventory path %s does not exist, skipping.", inventory_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, presets: list[VitalPreset]) -> CorpusSchema:
        """从预设列表中提取 CorpusSchema。

        1. 取所有 settings keys 的并集
        2. 分类每个 key 为 binary/categorical/continuous
        3. 为 categorical 字段构建稳定 category_values 顺序，映射到 category_id
        4. 在统一 float 域中计算 default_value, corpus_min, corpus_max, presence_ratio
        5. 追加 modulation slot 字段和 wavetable 字段
        6. 与 inventory 对比，log 不在 inventory 中的 keys
        7. 按稳定排序输出
        """
        n_presets = len(presets)

        # Step 1: collect all settings keys union
        all_keys: set[str] = set()
        for p in presets:
            all_keys.update(p.settings.keys())

        # Step 2: classify each key
        param_types: dict[str, str] = {}
        for key in all_keys:
            param_types[key] = _classify_param(key)

        # Step 3: collect per-param values across corpus
        param_values_map: dict[str, list[float]] = {k: [] for k in all_keys}
        param_presence_count: dict[str, int] = {k: 0 for k in all_keys}
        # For categorical: collect raw string representations
        categorical_raw_values: dict[str, list[str]] = {
            k: [] for k in all_keys if param_types[k] == "categorical"
        }

        for p in presets:
            for key in all_keys:
                if key in p.settings:
                    param_presence_count[key] += 1
                    raw_val = p.settings[key]
                    if param_types[key] == "categorical":
                        categorical_raw_values[key].append(str(raw_val))
                    else:
                        # binary / continuous: store as float
                        if isinstance(raw_val, (int, float)):
                            param_values_map[key].append(float(raw_val))
                        else:
                            # non-numeric in a binary/continuous slot — treat as 0.0
                            param_values_map[key].append(0.0)

        # Step 3b: build stable category_values for categorical params
        category_values: dict[str, list[str]] = {}
        for key in sorted(categorical_raw_values.keys()):
            raw_vals = categorical_raw_values[key]
            # Stable order: sorted unique values
            unique_sorted = sorted(set(raw_vals))
            category_values[key] = unique_sorted

        # Build category_id lookup for each categorical param
        category_id_map: dict[str, dict[str, int]] = {}
        for key, cats in category_values.items():
            category_id_map[key] = {v: i for i, v in enumerate(cats)}

        # Now compute encoded values for categorical params
        for key in categorical_raw_values:
            id_lookup = category_id_map[key]
            for raw_str in categorical_raw_values[key]:
                param_values_map[key].append(float(id_lookup[raw_str]))

        # Step 4: compute statistics in unified float domain
        default_values: dict[str, float] = {}
        corpus_min: dict[str, float] = {}
        corpus_max: dict[str, float] = {}
        presence_ratio: dict[str, float] = {}

        for key in all_keys:
            vals = param_values_map[key]
            ptype = param_types[key]

            # presence_ratio
            presence_ratio[key] = (
                param_presence_count[key] / n_presets if n_presets > 0 else 0.0
            )

            if not vals:
                default_values[key] = 0.0
                corpus_min[key] = 0.0
                corpus_max[key] = 0.0
                continue

            corpus_min[key] = min(vals)
            corpus_max[key] = max(vals)

            if ptype == "continuous":
                # default = corpus median
                default_values[key] = float(median(vals))
            elif ptype == "binary":
                # default = mode (most common value)
                counter = Counter(vals)
                default_values[key] = float(counter.most_common(1)[0][0])
            elif ptype == "categorical":
                # default = category_id of corpus mode
                raw_vals = categorical_raw_values[key]
                counter = Counter(raw_vals)
                mode_str = counter.most_common(1)[0][0]
                default_values[key] = float(category_id_map[key][mode_str])

        # Step 5: determine encoding
        param_value_encoding: dict[str, str] = {}
        for key in all_keys:
            if param_types[key] == "categorical":
                param_value_encoding[key] = "category_id"
            else:
                param_value_encoding[key] = "identity"

        # Step 5b: append modulation slot fields (k=1..64)
        mod_fields: list[str] = []
        for k in range(1, 65):
            for suffix in _MOD_SLOT_SUFFIXES:
                field_name = f"modulation_{k}_{suffix}"
                mod_fields.append(field_name)
                if field_name not in param_types:
                    if suffix in ("source", "destination"):
                        param_types[field_name] = "categorical"
                        param_value_encoding[field_name] = "category_id"
                        category_values[field_name] = []
                    elif suffix == "bypass":
                        param_types[field_name] = "binary"
                        param_value_encoding[field_name] = "identity"
                    else:
                        param_types[field_name] = "continuous"
                        param_value_encoding[field_name] = "identity"
                    default_values[field_name] = 0.0
                    corpus_min[field_name] = 0.0
                    corpus_max[field_name] = 0.0
                    presence_ratio[field_name] = 0.0

        # Step 5c: append wavetable fields (i=1..3)
        wt_fields: list[str] = []
        for i in range(1, 4):
            for tmpl in _WAVETABLE_FIELD_TEMPLATES:
                field_name = tmpl.format(i=i)
                wt_fields.append(field_name)
                if field_name not in param_types:
                    param_types[field_name] = "categorical"
                    param_value_encoding[field_name] = "category_id"
                    category_values[field_name] = []
                    default_values[field_name] = 0.0
                    corpus_min[field_name] = 0.0
                    corpus_max[field_name] = 0.0
                    presence_ratio[field_name] = 0.0

        # Step 6: cross-reference with inventory, log keys not in inventory
        if self._inventory_set is not None:
            corpus_only = all_keys - self._inventory_set
            if corpus_only:
                for key in sorted(corpus_only):
                    logger.info("Corpus key not in inventory: %s", key)

        # Step 7: stable ordering
        param_names = self._build_stable_order(
            corpus_keys=all_keys,
            mod_fields=mod_fields,
            wt_fields=wt_fields,
        )

        return CorpusSchema(
            param_names=param_names,
            param_types={k: param_types[k] for k in param_names},
            param_value_encoding={k: param_value_encoding[k] for k in param_names},
            default_values={k: default_values[k] for k in param_names},
            corpus_min={k: corpus_min[k] for k in param_names},
            corpus_max={k: corpus_max[k] for k in param_names},
            presence_ratio={k: presence_ratio[k] for k in param_names},
            category_values={
                k: category_values[k]
                for k in param_names
                if k in category_values
            },
        )

    def save_schema(self, schema: CorpusSchema, output_path: Path) -> None:
        """将 schema 保存为 JSON 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(schema)
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Schema saved to %s (%d params)", output_path, len(schema.param_names))

    def load_schema(self, path: Path) -> CorpusSchema:
        """从 JSON 文件加载 schema。"""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        schema = CorpusSchema(
            param_names=data["param_names"],
            param_types=data["param_types"],
            param_value_encoding=data["param_value_encoding"],
            default_values={k: float(v) for k, v in data["default_values"].items()},
            corpus_min={k: float(v) for k, v in data["corpus_min"].items()},
            corpus_max={k: float(v) for k, v in data["corpus_max"].items()},
            presence_ratio={k: float(v) for k, v in data["presence_ratio"].items()},
            category_values=data.get("category_values", {}),
        )
        logger.info("Schema loaded from %s (%d params)", path, len(schema.param_names))
        return schema

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_stable_order(
        self,
        corpus_keys: set[str],
        mod_fields: list[str],
        wt_fields: list[str],
    ) -> list[str]:
        """Build a stable parameter ordering.

        Order:
        1. Params that appear in inventory, in inventory order
           (excluding modulation slot fields and wavetable fields that
            are handled separately)
        2. Modulation slot fields: modulation_k_* for k=1..64
        3. Wavetable fields: osc_i_wavetable_* for i=1..3
        4. Corpus keys not in inventory, in alphabetical order
        """
        mod_field_set = set(mod_fields)
        wt_field_set = set(wt_fields)

        ordered: list[str] = []
        seen: set[str] = set()

        # 1. Inventory order (if available)
        if self._inventory_order is not None:
            for name in self._inventory_order:
                if name in mod_field_set or name in wt_field_set:
                    continue
                # Include if it's a corpus key OR if it's a modulation numeric
                # param that exists in inventory (like modulation_1_amount etc.)
                # We include inventory params that are in corpus_keys or that
                # are modulation/wavetable fields handled separately
                if name in corpus_keys and name not in seen:
                    ordered.append(name)
                    seen.add(name)

        # 2. Modulation slot fields in canonical order
        for name in mod_fields:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        # 3. Wavetable fields in canonical order
        for name in wt_fields:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        # 4. Remaining corpus keys not yet placed, alphabetical
        remaining = sorted(corpus_keys - seen)
        for name in remaining:
            ordered.append(name)
            seen.add(name)

        return ordered

"""
路由掩码增强模块。

对路由图中激活的可掩码边做变异，生成多个 masked variants。
variant_0 始终为原始未掩码预设。K<=8 时有界穷举，K>8 时随机采样。

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Literal

from src.preset_parser import VitalPreset
from src.route_graph_builder import RouteEdge, RouteGraph, RouteGraphBuilder

logger = logging.getLogger(__name__)

RunMode = Literal["pilot", "canary", "full"]


@dataclass
class RouteMask:
    """路由掩码。

    Attributes:
        mask_vector: 与 maskable edges 对齐的 0/1 向量，1=保留 0=禁用
        masked_edge_names: 被禁用的边名称列表
    """

    mask_vector: list[int]
    masked_edge_names: list[str]


@dataclass
class MaskMetadata:
    """掩码变异元数据。

    Attributes:
        base_preset_id: 基础预设标识
        variant_id: 变体编号
        num_masked_edges: 被掩码的边数量
        total_active_edges: 激活边总数
        maskable_edge_names: 所有可掩码边的有序名称列表
    """

    base_preset_id: str
    variant_id: int
    num_masked_edges: int
    total_active_edges: int
    maskable_edge_names: list[str]


@dataclass
class MaskedVariant:
    """一个掩码变异结果。

    Attributes:
        preset: 变异后的 VitalPreset（variant_0 为原始预设的深拷贝）
        route_mask: 路由掩码
        metadata: 掩码元数据
    """

    preset: VitalPreset
    route_mask: RouteMask
    metadata: MaskMetadata


def _edge_name(edge: RouteEdge) -> str:
    """Generate a stable name for a route edge."""
    return f"{edge.edge_type}:{edge.source}->{edge.destination}"


class RouteMaskAugmenter:
    """对路由图中激活的可掩码边做变异，生成多个 variants。

    算法：
    1. 构建 RouteGraph
    2. 提取 maskable edges 列表（is_maskable=True and is_active=True）
    3. variant_0 = 原始未掩码预设（mask_vector 全 1）
    4. K<=8: 有界穷举所有 2^K 掩码（排除全 1），过滤有效掩码，cap at effective_max
    5. K>8: 随机采样有效掩码，上限 effective_max
    6. 对每个有效掩码，apply mutation_rule 生成 mutated VitalPreset
    7. 返回 [variant_0, variant_1, ...]
    """

    # Sources that can produce sound
    _SOUND_SOURCES = {"osc_1", "osc_2", "osc_3", "sample"}

    # On-key mapping for sources
    _SOURCE_ON_KEYS = {
        "osc_1": "osc_1_on",
        "osc_2": "osc_2_on",
        "osc_3": "osc_3_on",
        "sample": "sample_on",
    }

    # Destination key mapping for sources
    _SOURCE_DEST_KEYS = {
        "osc_1": "osc_1_destination",
        "osc_2": "osc_2_destination",
        "osc_3": "osc_3_destination",
        "sample": "sample_destination",
    }

    # Filter on-key mapping
    _FILTER_ON_KEYS = {
        "filter_1": "filter_1_on",
        "filter_2": "filter_2_on",
        "filter_fx": "filter_fx_on",
    }

    def __init__(
        self,
        graph_builder: RouteGraphBuilder,
        min_variants: int = 16,
        max_variants: int = 64,
    ) -> None:
        self._graph_builder = graph_builder
        self._min_variants = min_variants
        self._max_variants = max_variants

    def augment(
        self,
        preset: VitalPreset,
        base_preset_id: str,
        run_mode: RunMode = "full",
        pilot_max_variants_per_base: int = 8,
        canary_max_variants_per_base: int = 24,
    ) -> list[MaskedVariant]:
        """为一个 base preset 生成所有 masked variants。

        Args:
            preset: 原始预设
            base_preset_id: 基础预设标识
            run_mode: 运行模式 ("pilot", "canary", "full")
            pilot_max_variants_per_base: pilot 模式下每个 base preset 的最大 variant 数
            canary_max_variants_per_base: canary 模式下每个 base preset 的最大 variant 数

        Returns:
            MaskedVariant 列表，variant_0 始终为原始未掩码预设
        """
        # 1. Build RouteGraph
        graph = self._graph_builder.build(preset)

        # 2. Extract maskable edges (is_maskable=True and is_active=True)
        maskable_edges = [
            e for e in graph.edges if e.is_maskable and e.is_active
        ]
        K = len(maskable_edges)
        edge_names = [_edge_name(e) for e in maskable_edges]

        # 3. Compute effective_max based on run_mode
        effective_max = self._effective_max_variants(
            run_mode, pilot_max_variants_per_base, canary_max_variants_per_base
        )

        # 4. variant_0 = original unmasked preset
        variant_0 = MaskedVariant(
            preset=copy.deepcopy(preset),
            route_mask=RouteMask(
                mask_vector=[1] * K,
                masked_edge_names=[],
            ),
            metadata=MaskMetadata(
                base_preset_id=base_preset_id,
                variant_id=0,
                num_masked_edges=0,
                total_active_edges=graph.total_active_edges,
                maskable_edge_names=list(edge_names),
            ),
        )
        variants: list[MaskedVariant] = [variant_0]

        # 5. If no maskable edges, return only variant_0
        if K == 0:
            logger.warning(
                "No maskable edges for preset '%s'; returning only variant_0",
                base_preset_id,
            )
            return variants

        # 6. Generate masked variants
        if K <= 8:
            variants = self._exhaustive_enumerate(
                preset, base_preset_id, graph, maskable_edges,
                edge_names, K, effective_max, variants,
            )
        else:
            variants = self._random_sample(
                preset, base_preset_id, graph, maskable_edges,
                edge_names, K, effective_max, variants,
            )

        # If no additional variants were generated beyond variant_0, log warning
        if len(variants) == 1:
            logger.warning(
                "No valid masks found for preset '%s'; returning only variant_0",
                base_preset_id,
            )

        return variants

    def _effective_max_variants(
        self,
        run_mode: RunMode,
        pilot_max: int,
        canary_max: int,
    ) -> int:
        """Compute the effective max variant count (excluding variant_0) for the run mode."""
        if run_mode == "pilot":
            return pilot_max
        elif run_mode == "canary":
            return canary_max
        else:  # full
            return self._max_variants

    def _exhaustive_enumerate(
        self,
        preset: VitalPreset,
        base_preset_id: str,
        graph: RouteGraph,
        maskable_edges: list[RouteEdge],
        edge_names: list[str],
        K: int,
        effective_max: int,
        variants: list[MaskedVariant],
    ) -> list[MaskedVariant]:
        """Bounded exhaustive enumeration for K<=8."""
        for bits in itertools.product([0, 1], repeat=K):
            mask = list(bits)
            # Skip the all-1s mask (that's variant_0)
            if all(b == 1 for b in mask):
                continue
            # Cap check: variants already includes variant_0
            if len(variants) >= 1 + effective_max:
                break
            if self._is_valid_mask(mask, maskable_edges, preset):
                mutated = self._apply_mask(preset, mask, maskable_edges)
                masked_names = [
                    edge_names[i] for i in range(K) if mask[i] == 0
                ]
                variants.append(MaskedVariant(
                    preset=mutated,
                    route_mask=RouteMask(
                        mask_vector=list(mask),
                        masked_edge_names=masked_names,
                    ),
                    metadata=MaskMetadata(
                        base_preset_id=base_preset_id,
                        variant_id=len(variants),
                        num_masked_edges=mask.count(0),
                        total_active_edges=graph.total_active_edges,
                        maskable_edge_names=list(edge_names),
                    ),
                ))
        return variants

    def _random_sample(
        self,
        preset: VitalPreset,
        base_preset_id: str,
        graph: RouteGraph,
        maskable_edges: list[RouteEdge],
        edge_names: list[str],
        K: int,
        effective_max: int,
        variants: list[MaskedVariant],
    ) -> list[MaskedVariant]:
        """Random sampling for K>8."""
        sampled = 0
        attempts = 0
        max_attempts = effective_max * 10
        seen: set[tuple[int, ...]] = set()

        while sampled < effective_max and attempts < max_attempts:
            attempts += 1
            mask = [random.randint(0, 1) for _ in range(K)]
            # Skip all-1s mask
            if all(b == 1 for b in mask):
                continue
            # Skip duplicates
            mask_tuple = tuple(mask)
            if mask_tuple in seen:
                continue
            seen.add(mask_tuple)

            if self._is_valid_mask(mask, maskable_edges, preset):
                mutated = self._apply_mask(preset, mask, maskable_edges)
                masked_names = [
                    edge_names[i] for i in range(K) if mask[i] == 0
                ]
                variants.append(MaskedVariant(
                    preset=mutated,
                    route_mask=RouteMask(
                        mask_vector=list(mask),
                        masked_edge_names=masked_names,
                    ),
                    metadata=MaskMetadata(
                        base_preset_id=base_preset_id,
                        variant_id=len(variants),
                        num_masked_edges=mask.count(0),
                        total_active_edges=graph.total_active_edges,
                        maskable_edge_names=list(edge_names),
                    ),
                ))
                sampled += 1

        return variants

    def _is_valid_mask(
        self,
        mask: list[int],
        edges: list[RouteEdge],
        preset: VitalPreset,
    ) -> bool:
        """Check if a mask retains at least one source-to-output reachable path.

        Sound path reachability:
        - Sources (osc_1/2/3, sample) → optional filters (filter_1/2) → output
        - filter_fx is on the output path
        - A source reaches output if:
          1. It's on (osc_i_on=1) AND
          2. Either its destination is "direct" (bypasses filters) OR
             its destination filter is on (filter_X_on=1)

        The mask may disable source edges (set_on_to_0) or filter edges (set_on_to_0).
        We simulate the effect of the mask on the preset state to check reachability.
        """
        settings = dict(preset.settings)

        # Build a map of edge index to edge for quick lookup
        # Simulate the mask: for each disabled edge, determine what gets turned off
        disabled_sources: set[str] = set()
        disabled_filters: set[str] = set()

        for i, edge in enumerate(edges):
            if mask[i] == 0:
                if edge.edge_type == "signal" and edge.mutation_rule == "set_on_to_0":
                    if edge.source in self._SOUND_SOURCES:
                        disabled_sources.add(edge.source)
                    elif edge.source in self._FILTER_ON_KEYS:
                        disabled_filters.add(edge.source)

        # Check each sound source for reachability to output
        for source in self._SOUND_SOURCES:
            on_key = self._SOURCE_ON_KEYS[source]
            dest_key = self._SOURCE_DEST_KEYS[source]

            # Check if source is on (considering mask disabling it)
            if source in disabled_sources:
                continue
            source_on = settings.get(on_key, 0)
            if not self._is_on(source_on):
                continue

            # Source is on — check if it can reach output
            dest_value = settings.get(dest_key, 0)
            dest_label = RouteGraphBuilder._resolve_source_destination(
                source, dest_value
            )

            if dest_label == "direct":
                # Direct path to output — always reachable
                return True

            # Destination is a filter (or filter_1+filter_2)
            # Check if the destination filter(s) are on and not disabled
            if dest_label == "filter_1+filter_2":
                filter_targets = ["filter_1", "filter_2"]
            else:
                filter_targets = [dest_label]

            for filt in filter_targets:
                if filt in disabled_filters:
                    continue
                filt_on_key = self._FILTER_ON_KEYS.get(filt)
                if filt_on_key is None:
                    continue
                filt_on = settings.get(filt_on_key, 0)
                if self._is_on(filt_on):
                    return True

        return False

    def _apply_mask(
        self,
        preset: VitalPreset,
        mask: list[int],
        edges: list[RouteEdge],
    ) -> VitalPreset:
        """Apply a mask to a preset, returning a deep-copied mutated preset.

        For each mask[i]==0:
        - "set_on_to_0": find the corresponding *_on key in settings and set to 0
        - "set_bypass_to_1": find the corresponding modulation slot and set bypass=1
        """
        mutated = VitalPreset(
            settings=copy.deepcopy(preset.settings),
            modulations=copy.deepcopy(preset.modulations),
            extra=copy.deepcopy(preset.extra),
        )

        for i, edge in enumerate(edges):
            if mask[i] == 1:
                continue

            if edge.mutation_rule == "set_on_to_0":
                # Signal edge: set the *_on key to 0
                on_key = self._find_on_key(edge)
                if on_key and on_key in mutated.settings:
                    mutated.settings[on_key] = 0

            elif edge.mutation_rule == "set_bypass_to_1":
                # Modulation edge: find the matching modulation slot and set bypass=1
                self._set_modulation_bypass(
                    mutated, edge.source, edge.destination
                )

        return mutated

    @staticmethod
    def _find_on_key(edge: RouteEdge) -> str | None:
        """Find the *_on settings key for a signal edge."""
        # Source edges: osc_1 -> osc_1_on, sample -> sample_on
        # Filter edges: filter_1 -> filter_1_on, filter_fx -> filter_fx_on
        source = edge.source
        on_key_map = {
            "osc_1": "osc_1_on",
            "osc_2": "osc_2_on",
            "osc_3": "osc_3_on",
            "sample": "sample_on",
            "filter_1": "filter_1_on",
            "filter_2": "filter_2_on",
            "filter_fx": "filter_fx_on",
        }
        return on_key_map.get(source)

    @staticmethod
    def _set_modulation_bypass(
        preset: VitalPreset, source: str, destination: str
    ) -> None:
        """Find the modulation slot matching source+destination and set bypass=1."""
        for mod in preset.modulations:
            if mod.get("source") == source and mod.get("destination") == destination:
                mod["bypass"] = 1
                return

    @staticmethod
    def _is_on(value: float | int | str | list) -> bool:
        """Check if an on/off parameter value represents 'on' (== 1)."""
        if isinstance(value, (int, float)):
            return float(value) == 1.0
        return False

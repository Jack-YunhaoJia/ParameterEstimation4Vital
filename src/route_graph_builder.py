"""
路由图构建模块。

从 VitalPreset 中提取信号路由（Signal_Route）和调制路由（Modulation_Route），
构建路由图供 RouteMaskAugmenter 使用。

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.preset_parser import VitalPreset

logger = logging.getLogger(__name__)


@dataclass
class RouteEdge:
    """路由图中的一条边。

    Attributes:
        edge_type: "signal" | "modulation"
        source: 源节点标识（如 "osc_1", "lfo_1"）
        destination: 目标节点标识（如 "filter_1", "osc_1_level"）
        parameters: 边上的参数（signal: {"on": v, "destination": v}, modulation: {"amount", "bypass", ...}）
        is_active: 当前是否激活
        is_maskable: V1 中是否可被安全掩码
        mutation_rule: 掩码时的变异规则 ("set_on_to_0" | "set_bypass_to_1" | "observed_only")
    """

    edge_type: str
    source: str
    destination: str
    parameters: dict = field(default_factory=dict)
    is_active: bool = False
    is_maskable: bool = True
    mutation_rule: str = "set_on_to_0"


@dataclass
class RouteGraph:
    """预设的路由图。

    Attributes:
        edges: 所有路由边列表
        total_active_edges: 激活边总数
        total_active_maskable_edges: 激活且可掩码的边总数
    """

    edges: list[RouteEdge] = field(default_factory=list)
    total_active_edges: int = 0
    total_active_maskable_edges: int = 0


class RouteGraphBuilder:
    """从 VitalPreset 构建路由图。

    提取两类边：
    - Signal_Route 边：振荡器/采样器/滤波器的开关和路由
    - Modulation_Route 边：调制槽位的 source → destination 连接
    """

    # Oscillator / sample source edges: (source_name, on_key, destination_key)
    _SOURCE_EDGES: list[tuple[str, str, str]] = [
        ("osc_1", "osc_1_on", "osc_1_destination"),
        ("osc_2", "osc_2_on", "osc_2_destination"),
        ("osc_3", "osc_3_on", "osc_3_destination"),
        ("sample", "sample_on", "sample_destination"),
    ]

    # Filter edges: (source_name, on_key, input_key_or_none)
    # filter_1 and filter_2 have a filter_input key; filter_fx does not
    _FILTER_EDGES: list[tuple[str, str, str | None]] = [
        ("filter_1", "filter_1_on", "filter_1_filter_input"),
        ("filter_2", "filter_2_on", "filter_2_filter_input"),
        ("filter_fx", "filter_fx_on", None),
    ]

    # Observed-only edges: (source_name, param_key)
    _OBSERVED_EDGES: list[tuple[str, str]] = [
        ("effect_chain", "effect_chain_order"),
        ("stereo_routing", "stereo_routing"),
    ]

    def build(self, preset: VitalPreset) -> RouteGraph:
        """从 VitalPreset 构建路由图。

        Args:
            preset: 解析后的 Vital 预设

        Returns:
            包含所有信号边和调制边的 RouteGraph
        """
        edges: list[RouteEdge] = []

        # --- Signal_Route edges ---
        edges.extend(self._extract_source_edges(preset))
        edges.extend(self._extract_filter_edges(preset))
        edges.extend(self._extract_observed_edges(preset))

        # --- Modulation_Route edges ---
        edges.extend(self._extract_modulation_edges(preset))

        # Compute counts
        total_active = sum(1 for e in edges if e.is_active)
        total_active_maskable = sum(
            1 for e in edges if e.is_active and e.is_maskable
        )

        return RouteGraph(
            edges=edges,
            total_active_edges=total_active,
            total_active_maskable_edges=total_active_maskable,
        )

    def _extract_source_edges(self, preset: VitalPreset) -> list[RouteEdge]:
        """Extract oscillator and sample source signal edges."""
        edges: list[RouteEdge] = []
        settings = preset.settings

        for source_name, on_key, dest_key in self._SOURCE_EDGES:
            on_value = settings.get(on_key, 0)
            dest_value = settings.get(dest_key, 0)

            # Determine destination label from the numeric destination value
            dest_label = self._resolve_source_destination(source_name, dest_value)

            is_active = self._is_on(on_value)

            edges.append(RouteEdge(
                edge_type="signal",
                source=source_name,
                destination=dest_label,
                parameters={"on": on_value, "destination": dest_value},
                is_active=is_active,
                is_maskable=True,
                mutation_rule="set_on_to_0",
            ))

        return edges

    def _extract_filter_edges(self, preset: VitalPreset) -> list[RouteEdge]:
        """Extract filter signal edges (filter → output)."""
        edges: list[RouteEdge] = []
        settings = preset.settings

        for source_name, on_key, input_key in self._FILTER_EDGES:
            on_value = settings.get(on_key, 0)
            params: dict = {"on": on_value}

            if input_key is not None:
                input_value = settings.get(input_key, 0)
                params["filter_input"] = input_value

            is_active = self._is_on(on_value)

            edges.append(RouteEdge(
                edge_type="signal",
                source=source_name,
                destination="output",
                parameters=params,
                is_active=is_active,
                is_maskable=True,
                mutation_rule="set_on_to_0",
            ))

        return edges

    def _extract_observed_edges(self, preset: VitalPreset) -> list[RouteEdge]:
        """Extract observed-only signal edges (effect_chain_order, stereo_routing).

        These edges are always active but not maskable (is_maskable=False).
        """
        edges: list[RouteEdge] = []
        settings = preset.settings

        for source_name, param_key in self._OBSERVED_EDGES:
            param_value = settings.get(param_key, 0)

            edges.append(RouteEdge(
                edge_type="signal",
                source=source_name,
                destination="global",
                parameters={param_key: param_value},
                is_active=True,
                is_maskable=False,
                mutation_rule="observed_only",
            ))

        return edges

    def _extract_modulation_edges(self, preset: VitalPreset) -> list[RouteEdge]:
        """Extract modulation route edges from active modulation slots.

        A slot is active when both source and destination are non-empty strings.
        """
        edges: list[RouteEdge] = []

        for i, mod in enumerate(preset.modulations):
            source = mod.get("source", "")
            destination = mod.get("destination", "")

            # Active only when both source and destination are non-empty strings
            is_active = (
                isinstance(source, str) and source != ""
                and isinstance(destination, str) and destination != ""
            )

            if not is_active:
                continue

            edges.append(RouteEdge(
                edge_type="modulation",
                source=source,
                destination=destination,
                parameters={
                    "amount": mod.get("amount", 0.0),
                    "bypass": mod.get("bypass", 0),
                    "bipolar": mod.get("bipolar", 0),
                    "power": mod.get("power", 0.0),
                    "stereo": mod.get("stereo", 0.0),
                },
                is_active=True,
                is_maskable=True,
                mutation_rule="set_bypass_to_1",
            ))

        return edges

    @staticmethod
    def _is_on(value: float | int | str | list) -> bool:
        """Check if an on/off parameter value represents 'on' (== 1)."""
        if isinstance(value, (int, float)):
            return float(value) == 1.0
        return False

    @staticmethod
    def _resolve_source_destination(
        source_name: str, dest_value: float | int | str | list
    ) -> str:
        """Resolve the destination label for an oscillator/sample source.

        In Vital, osc_i_destination / sample_destination is a numeric value:
        - 0 → filter_1
        - 1 → filter_2
        - 2 → filter_1 + filter_2
        - 3 → direct (bypasses filters)
        Other values default to "direct".
        """
        if not isinstance(dest_value, (int, float)):
            return "direct"

        dest_int = int(dest_value)
        mapping = {
            0: "filter_1",
            1: "filter_2",
            2: "filter_1+filter_2",
            3: "direct",
        }
        return mapping.get(dest_int, "direct")

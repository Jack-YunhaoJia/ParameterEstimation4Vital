"""
Shared Hypothesis composite strategies for the preset-corpus-pipeline test suite.

Provides reusable strategies for generating VitalPreset objects, modulation slots,
route edges, and route masks across all property-based tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import strategies as st

from src.preset_parser import VitalPreset


# ---------------------------------------------------------------------------
# KNOWN_PARAM_NAMES — representative subset from vital_param_inventory.json
# covering binary, categorical, and continuous parameter types.
# ---------------------------------------------------------------------------

# Binary params: *_on, *_bypass, legato
KNOWN_BINARY_PARAMS: list[str] = [
    "osc_1_on",
    "osc_2_on",
    "osc_3_on",
    "sample_on",
    "filter_1_on",
    "filter_2_on",
    "filter_fx_on",
    "chorus_on",
    "compressor_on",
    "delay_on",
    "distortion_on",
    "eq_on",
    "flanger_on",
    "phaser_on",
    "reverb_on",
    "legato",
]

# Categorical params: *_destination, *_model, *_style, *_type, effect_chain_order, stereo_routing
KNOWN_CATEGORICAL_PARAMS: list[str] = [
    "osc_1_destination",
    "osc_2_destination",
    "osc_3_destination",
    "sample_destination",
    "filter_1_model",
    "filter_1_style",
    "filter_2_model",
    "filter_2_style",
    "filter_fx_model",
    "filter_fx_style",
    "osc_1_distortion_type",
    "osc_1_spectral_morph_type",
    "delay_style",
    "distortion_type",
    "effect_chain_order",
    "stereo_routing",
]

# Continuous params: levels, tunes, pans, cutoffs, resonances, effects, volume, etc.
KNOWN_CONTINUOUS_PARAMS: list[str] = [
    "osc_1_level",
    "osc_1_tune",
    "osc_1_pan",
    "osc_1_transpose",
    "osc_1_unison_detune",
    "osc_1_unison_voices",
    "osc_1_wave_frame",
    "osc_1_frame_spread",
    "osc_2_level",
    "osc_2_tune",
    "osc_2_pan",
    "osc_3_level",
    "osc_3_tune",
    "osc_3_pan",
    "sample_level",
    "sample_tune",
    "sample_pan",
    "filter_1_cutoff",
    "filter_1_resonance",
    "filter_1_drive",
    "filter_1_blend",
    "filter_1_mix",
    "filter_2_cutoff",
    "filter_2_resonance",
    "filter_fx_cutoff",
    "filter_fx_resonance",
    "chorus_frequency",
    "chorus_dry_wet",
    "chorus_feedback",
    "delay_frequency",
    "delay_dry_wet",
    "delay_feedback",
    "reverb_decay_time",
    "reverb_dry_wet",
    "reverb_size",
    "env_1_attack",
    "env_1_decay",
    "env_1_sustain",
    "env_1_release",
    "lfo_1_frequency",
    "lfo_1_phase",
    "polyphony",
    "portamento_time",
    "volume",
]

# Combined list for general-purpose sampling
KNOWN_PARAM_NAMES: list[str] = (
    KNOWN_BINARY_PARAMS + KNOWN_CATEGORICAL_PARAMS + KNOWN_CONTINUOUS_PARAMS
)


# Known modulation sources and destinations (from vital_param_inventory.json)
KNOWN_MOD_SOURCES: list[str] = [
    "env_2",
    "lfo_1",
    "lfo_2",
    "lfo_3",
    "lfo_4",
    "macro_control_1",
    "macro_control_2",
    "mod_wheel",
    "note_in_octave",
    "random_1",
    "random_2",
    "velocity",
]

KNOWN_MOD_DESTINATIONS: list[str] = [
    "osc_1_level",
    "osc_1_tune",
    "osc_1_wave_frame",
    "osc_2_level",
    "osc_2_tune",
    "filter_1_cutoff",
    "filter_1_resonance",
    "filter_2_cutoff",
    "reverb_decay_time",
    "reverb_dry_wet",
    "delay_dry_wet",
    "sample_level",
    "voice_tune",
]


# ---------------------------------------------------------------------------
# Hypothesis Composite Strategies
# ---------------------------------------------------------------------------


@st.composite
def modulation_slots(draw: st.DrawFn) -> dict:
    """Generate a single modulation slot dict.

    Each slot has source, destination, and numeric parameters (amount, bypass,
    bipolar, power, stereo) matching the Vital modulation slot structure.
    """
    # Decide whether this slot is active (has source + destination) or empty
    is_active = draw(st.booleans())

    if is_active:
        source = draw(st.sampled_from(KNOWN_MOD_SOURCES))
        destination = draw(st.sampled_from(KNOWN_MOD_DESTINATIONS))
    else:
        source = ""
        destination = ""

    return {
        "source": source,
        "destination": destination,
        "amount": draw(st.floats(min_value=-1.0, max_value=1.0)),
        "bypass": draw(st.sampled_from([0, 1])),
        "bipolar": draw(st.sampled_from([0, 1])),
        "power": draw(st.floats(min_value=0.0, max_value=10.0)),
        "stereo": draw(st.floats(min_value=-1.0, max_value=1.0)),
    }


@st.composite
def vital_presets(draw: st.DrawFn) -> VitalPreset:
    """Generate a random VitalPreset with mixed binary/categorical/continuous settings.

    Follows the design doc Testing Strategy specification:
    - Settings contain a mix of known param names and occasional random keys
    - Values are a mix of floats, ints, and (for categorical) small ints
    - Modulations list contains 0..64 modulation slot dicts
    - Extra may optionally contain wavetables
    """
    n_settings = draw(st.integers(min_value=0, max_value=50))
    settings: dict[str, float | int | str | list] = {}

    # Keys that collide with PresetParser's internal structure must be excluded
    _RESERVED_KEYS = {"modulations", "wavetables"}

    for _ in range(n_settings):
        # Pick from known params or generate a random key
        key = draw(
            st.one_of(
                st.sampled_from(KNOWN_PARAM_NAMES),
                st.text(
                    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_0123456789"),
                    min_size=1,
                    max_size=30,
                ).filter(lambda k: k not in _RESERVED_KEYS),
            )
        )

        # Generate type-appropriate values based on param category
        if key in KNOWN_BINARY_PARAMS:
            value: float | int | str | list = draw(st.sampled_from([0, 1, 0.0, 1.0]))
        elif key in KNOWN_CATEGORICAL_PARAMS:
            value = draw(st.integers(min_value=0, max_value=10))
        else:
            value = draw(
                st.one_of(
                    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                    st.integers(min_value=0, max_value=127),
                )
            )

        settings[key] = value

    # Modulations: 0..64 slots
    n_mods = draw(st.integers(min_value=0, max_value=64))
    modulations = [draw(modulation_slots()) for _ in range(n_mods)]

    # Extra: optionally include wavetables
    extra: dict = {}
    if draw(st.booleans()):
        n_wt = draw(st.integers(min_value=1, max_value=3))
        extra["wavetables"] = [{"name": f"wt_{i}"} for i in range(n_wt)]

    return VitalPreset(settings=settings, modulations=modulations, extra=extra)


# ---------------------------------------------------------------------------
# Forward-compatible stubs for RouteEdge and RouteMask strategies.
# These will be refined when src/route_graph_builder.py and
# src/route_mask_augmenter.py are implemented.
# ---------------------------------------------------------------------------


@dataclass
class _StubRouteEdge:
    """Lightweight stub for route edge generation until RouteEdge is implemented."""

    edge_type: str  # "signal" or "modulation"
    source: str
    destination: str
    is_active: bool
    is_maskable: bool
    mutation_rule: str  # e.g. "set_on_to_0", "set_bypass_to_1"
    parameters: dict = field(default_factory=dict)


@dataclass
class _StubRouteMask:
    """Lightweight stub for route mask generation until RouteMask is implemented."""

    mask_vector: list[int]  # 1 = keep, 0 = disable
    n_masked: int
    n_total: int


@st.composite
def route_edges(draw: st.DrawFn) -> _StubRouteEdge:
    """Generate a random RouteEdge-like object.

    Stub strategy — will be updated to use the real RouteEdge dataclass
    once src/route_graph_builder.py is implemented.
    """
    edge_type = draw(st.sampled_from(["signal", "modulation"]))

    if edge_type == "signal":
        source = draw(st.sampled_from([
            "osc_1", "osc_2", "osc_3", "sample",
            "filter_1", "filter_2", "filter_fx",
        ]))
        destination = draw(st.sampled_from([
            "filter_1", "filter_2", "direct_out", "effect_chain",
        ]))
        mutation_rule = "set_on_to_0"
    else:
        source = draw(st.sampled_from(KNOWN_MOD_SOURCES))
        destination = draw(st.sampled_from(KNOWN_MOD_DESTINATIONS))
        mutation_rule = "set_bypass_to_1"

    is_active = draw(st.booleans())
    # Observed-only edges (effect_chain, stereo_routing) are not maskable
    is_maskable = draw(st.booleans()) if source not in ("effect_chain", "stereo_routing") else False

    return _StubRouteEdge(
        edge_type=edge_type,
        source=source,
        destination=destination,
        is_active=is_active,
        is_maskable=is_maskable,
        mutation_rule=mutation_rule,
        parameters={"amount": draw(st.floats(min_value=-1.0, max_value=1.0))} if edge_type == "modulation" else {},
    )


@st.composite
def route_masks(draw: st.DrawFn, max_edges: int = 10) -> _StubRouteMask:
    """Generate a random RouteMask-like object.

    Stub strategy — will be updated to use the real RouteMask dataclass
    once src/route_mask_augmenter.py is implemented.

    Args:
        max_edges: Upper bound on the number of maskable edges.
    """
    n_total = draw(st.integers(min_value=1, max_value=max_edges))
    # Each bit is 1 (keep) or 0 (disable), but at least one must be 1
    # to ensure a valid sound path exists
    mask_vector = draw(
        st.lists(
            st.sampled_from([0, 1]),
            min_size=n_total,
            max_size=n_total,
        ).filter(lambda v: sum(v) >= 1)  # at least one edge kept
    )
    n_masked = mask_vector.count(0)

    return _StubRouteMask(
        mask_vector=mask_vector,
        n_masked=n_masked,
        n_total=n_total,
    )

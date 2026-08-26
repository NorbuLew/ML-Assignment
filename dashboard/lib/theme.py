"""Colour and typography constants for the CANE dashboard.

The categorical palette is assigned **by entity, never by rank**: LinUCB is
always blue whether it comes first or last in a filtered view, so a reader who
learns the colours once can carry them between pages. Slots follow a validated
fixed order (see `references/palette.md`); both the light and dark sets were run
through the palette validator and pass the lightness, chroma, colour-vision-
deficiency and normal-vision separation gates.

Three light-mode hues (aqua, yellow, magenta) sit below 3:1 contrast on the light
surface. That triggers the *relief rule*: every chart using them ships visible
direct labels or an accompanying table view, so identity is never carried by
colour alone.
"""

from __future__ import annotations

import altair as alt

# Fixed slot assignment. Learners take slots 1-4, the ensemble slot 5, and the
# two non-learning baselines the last two -- so a glance separates "learned"
# from "did not learn" before any label is read.
AGENT_COLOUR = {
    "LinUCB":      "#2a78d6",   # slot 1  blue
    "DQN":         "#eb6834",   # slot 2  orange
    "DDQN":        "#1baf7a",   # slot 3  aqua
    "PPO":         "#eda100",   # slot 4  yellow
    "Ensemble":    "#e87ba4",   # slot 5  magenta
    "Fixed-18:00": "#008300",   # slot 6  green
    "Random":      "#4a3aa7",   # slot 7  violet
}

AGENT_COLOUR_DARK = {
    "LinUCB":      "#3987e5",
    "DQN":         "#d95926",
    "DDQN":        "#199e70",
    "PPO":         "#c98500",
    "Ensemble":    "#d55181",
    "Fixed-18:00": "#008300",
    "Random":      "#9085e9",
}

# Aliases the result files use for the same entity.
AGENT_ALIASES = {
    "DoubleDQN": "DDQN",
    "Double DQN": "DDQN",
    "ENSEMBLE": "Ensemble",
    "Ensemble-Vote": "Ensemble",
    "Ensemble-Gated": "Ensemble",
}

# Canonical display order: bandit -> value-based -> improved value-based ->
# policy-based -> combined -> baselines. This mirrors the progression the report
# argues for, so charts read left-to-right as the argument does.
AGENT_ORDER = ["LinUCB", "DQN", "DDQN", "PPO", "Ensemble", "Fixed-18:00", "Random"]

# The three actions. Hold is deliberately the recessive neutral: it is the
# default, and a decision chart should draw the eye to the sends.
ACTION_COLOUR = {"Hold": "#9a9a94", "Engage": "#2a78d6", "Incentive": "#eb6834"}
ACTION_ORDER = ["Hold", "Engage", "Incentive"]

ARCHETYPE_ORDER = ["OfficeWorker", "NightOwlStudent", "NightShiftWorker",
                   "NormalStudent", "Housewife"]

# Ink and surface tokens.
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

# Status colours are reserved and never reused as a series hue.
STATUS = {"good": "#008300", "warning": "#eda100",
          "serious": "#eb6834", "critical": "#e34948"}


def canonical(agent: str) -> str:
    """Map a result file's agent label onto the canonical entity name."""
    return AGENT_ALIASES.get(agent, agent)


def colour_for(agent: str) -> str:
    return AGENT_COLOUR.get(canonical(agent), TEXT_MUTED)


def domain_range(agents):
    """(domain, range) pair for an Altair colour scale, in canonical order.

    Passing an explicit domain keeps a filtered chart from repainting the
    survivors: drop DQN from the selection and LinUCB stays blue.
    """
    seen = [a for a in AGENT_ORDER if a in {canonical(x) for x in agents}]
    return seen, [AGENT_COLOUR[a] for a in seen]


# Altair theme. Registered once, on import, via the 5.5+ decorator API;
# alt.themes.register / .enable are deprecated and print a warning wall.
@alt.theme.register("cane", enable=True)
def altair_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig({
        "config": {
            "background": "transparent",
            "font": "Inter, -apple-system, Segoe UI, sans-serif",
            "view": {"stroke": "transparent", "continuousWidth": 560,
                     "continuousHeight": 340},
            "axis": {
                "labelColor": TEXT_SECONDARY, "titleColor": TEXT_SECONDARY,
                "labelFontSize": 11, "titleFontSize": 12, "titleFontWeight": 500,
                "domainColor": GRID, "tickColor": GRID, "gridColor": GRID,
                "gridWidth": 1, "labelPadding": 6, "titlePadding": 10,
            },
            "legend": {
                "labelColor": TEXT_SECONDARY, "titleColor": TEXT_SECONDARY,
                "labelFontSize": 11, "titleFontSize": 11, "symbolType": "square",
                "symbolSize": 90, "titleFontWeight": 500,
            },
            "title": {"color": TEXT_PRIMARY, "fontSize": 14,
                      "fontWeight": 600, "anchor": "start", "offset": 12},
        }
    })

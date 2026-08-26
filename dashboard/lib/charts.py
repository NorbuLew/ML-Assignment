"""Altair chart builders for the CANE dashboard.

Every chart here follows the same rules:

* Colour is assigned by entity through an explicit scale domain, so filtering
  the agent list never repaints the survivors.
* Marks are thin, grids recessive, and the ink stays in text tokens -- a
  coloured mark beside a label carries identity, the label itself does not.
* Charts whose palette slots fall below 3:1 on the light surface ship visible
  direct labels, so identity never rests on colour alone.
* One y-scale per chart, always. Where two measures of different scale need
  comparing (reward against send rate, say) they get two charts or a single
  chart with one measure on each spatial axis -- never twin axes.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from dashboard.lib import theme


def _scale(agents):
    domain, rng = theme.domain_range(agents)
    return alt.Scale(domain=domain, range=rng)


def leaderboard_bars(board: pd.DataFrame, title: str = "Mean reward per episode",
                     width: int = 470):
    """Horizontal bars with 95% CI whiskers, ranked.

    Horizontal because the agent names are words, not codes: vertical bars would
    force rotated labels. Values are printed in a right-hand column, which
    doubles as the relief the low-contrast palette slots require.
    """
    if board.empty:
        return None
    board = board.copy()
    board["agent_c"] = board["agent"].map(theme.canonical)
    order = board.sort_values("reward", ascending=False)["agent"].tolist()

    # Headroom on both ends so the outermost value label has somewhere to go
    # rather than colliding with the axis labels. 18% of the span is enough for
    # a six-character number at 11px.
    lo = min(board["ci_low"].min(), 0.0)
    hi = max(board["ci_high"].max(), 0.0)
    pad = 0.18 * max(hi - lo, 1.0)
    xscale = alt.Scale(domain=[lo - pad, hi + pad], nice=False)

    base = alt.Chart(board).encode(
        y=alt.Y("agent:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=12, labelColor=theme.TEXT_PRIMARY)),
    )
    bars = base.mark_bar(height=16, cornerRadiusEnd=4).encode(
        x=alt.X("reward:Q", title="mean reward per episode", scale=xscale),
        color=alt.Color("agent_c:N", scale=_scale(board["agent_c"]),
                        legend=None),
        tooltip=[alt.Tooltip("agent:N", title="agent"),
                 alt.Tooltip("reward:Q", title="reward", format=".2f"),
                 alt.Tooltip("ci_low:Q", title="95% CI low", format=".2f"),
                 alt.Tooltip("ci_high:Q", title="95% CI high", format=".2f"),
                 alt.Tooltip("ctr:Q", title="CTR", format=".3f"),
                 alt.Tooltip("sends:Q", title="sends/wk", format=".1f"),
                 alt.Tooltip("n:Q", title="seeds")],
    )
    ci = base.mark_rule(strokeWidth=2, color=theme.TEXT_SECONDARY, opacity=0.55).encode(
        x=alt.X("ci_low:Q", scale=xscale), x2="ci_high:Q")
    # Each value sits just past the outer end of its own bar. Positive and
    # negative bars need opposite text anchors, and `align` is a mark property
    # that cannot be bound to a data field, so the split is done in pandas.
    # Without this, a negative bar's label lands on its own fill -- dark ink on a
    # saturated colour, unreadable.
    label_layers = []
    for frame, anchor, dx, xfield in (
            (board[board["reward"] >= 0], "left", 7, "ci_high:Q"),
            (board[board["reward"] < 0], "right", -7, "ci_low:Q")):
        if frame.empty:
            continue
        label_layers.append(
            alt.Chart(frame).mark_text(
                align=anchor, dx=dx, fontSize=11, color=theme.TEXT_SECONDARY
            ).encode(y=alt.Y("agent:N", sort=order, title=None),
                     x=alt.X(xfield, scale=xscale),
                     text=alt.Text("reward:Q", format=".2f")))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeWidth=1, color=theme.TEXT_MUTED, strokeDash=[3, 3]).encode(
        x=alt.X("x:Q", scale=xscale))

    # 44px per row keeps every agent label on its own line: Altair silently
    # drops overlapping axis labels, so an under-tall chart loses names rather
    # than crowding them, which reads as missing data. `width` is explicit
    # because the value column is positioned in pixel space.
    return alt.layer(zero, bars, ci, *label_layers).properties(
        width=width, height=44 * len(board) + 40, title=title)


def efficiency_frontier(board: pd.DataFrame):
    """Sends per week against click-through rate, sized by reward.

    This is the chart that states the assignment's actual objective: engagement
    against fatigue. An agent far right sends a lot; one high up converts well.
    The interesting region is top-left -- high conversion on few sends.

    Every point is directly labelled. Beyond three series the palette cannot
    clear the all-pairs colour-vision gates, and a scatter puts every pair on
    screen at once, so the labels are load-bearing rather than decorative.
    """
    if board.empty:
        return None
    b = board.copy()
    b["agent_c"] = b["agent"].map(theme.canonical)
    b["reward_size"] = b["reward"] - b["reward"].min() + 1.0

    pts = alt.Chart(b).mark_circle(opacity=0.85, stroke="#fcfcfb",
                                   strokeWidth=2).encode(
        x=alt.X("sends:Q", title="notifications sent per week"),
        y=alt.Y("ctr:Q", title="click-through rate"),
        size=alt.Size("reward_size:Q", legend=None, scale=alt.Scale(range=[120, 900])),
        color=alt.Color("agent_c:N", scale=_scale(b["agent_c"]),
                        legend=alt.Legend(title="agent", orient="right")),
        tooltip=[alt.Tooltip("agent:N"), alt.Tooltip("reward:Q", format=".2f"),
                 alt.Tooltip("sends:Q", title="sends/wk", format=".1f"),
                 alt.Tooltip("ctr:Q", format=".3f"),
                 alt.Tooltip("optout:Q", title="opt-out", format=".3f")],
    )
    labels = alt.Chart(b).mark_text(align="left", dx=10, dy=-10, fontSize=11,
                                    color=theme.TEXT_PRIMARY).encode(
        x="sends:Q", y="ctr:Q", text="agent:N")
    return (pts + labels).properties(height=380,
                                     title="Efficiency frontier - engagement against volume")


def reward_by_archetype(df: pd.DataFrame):
    """Grouped bars: each agent's reward on each of the five user archetypes.

    Faceting by archetype rather than stacking is deliberate. The headline
    result -- that a policy which looks degenerate on the mixed population is
    correct on some archetypes and badly wrong on others -- is only visible when
    the archetypes sit side by side.
    """
    if df.empty:
        return None
    d = (df.groupby(["archetype", "agent"])["reward_mean"].mean().reset_index())
    d["agent_c"] = d["agent"].map(theme.canonical)
    order = [a for a in theme.ARCHETYPE_ORDER if a in set(d["archetype"])]

    # Both layers are declared without their own data so that the facet
    # operator can supply it once at the top level -- Altair rejects a faceted
    # layer whose children carry separate data.
    zero = alt.Chart().mark_rule(
        strokeWidth=1, color=theme.TEXT_MUTED, strokeDash=[3, 3]
    ).encode(y=alt.datum(0))
    bars = alt.Chart().mark_bar(cornerRadiusEnd=3, size=14).encode(
        x=alt.X("agent_c:N", title=None, axis=alt.Axis(labels=False, ticks=False),
                sort=theme.AGENT_ORDER),
        y=alt.Y("reward_mean:Q", title="mean reward"),
        color=alt.Color("agent_c:N", scale=_scale(d["agent_c"]),
                        legend=alt.Legend(title="agent", orient="bottom",
                                          columns=4)),
        tooltip=[alt.Tooltip("agent:N"), alt.Tooltip("archetype:N"),
                 alt.Tooltip("reward_mean:Q", title="reward", format=".2f")],
    )
    return alt.layer(zero, bars).properties(width=150, height=260).facet(
        data=d,
        column=alt.Column("archetype:N", sort=order, title=None,
                          header=alt.Header(labelFontSize=12,
                                            labelColor=theme.TEXT_PRIMARY))
    ).properties(title="Reward by agent and user archetype")


def bias_sweep(sweep: pd.DataFrame, archetype: str, members: pd.DataFrame | None = None):
    """Ensemble reward against the learned hold bias, with member reference lines.

    The single most informative chart in the Part B section: it shows the whole
    trade-off rather than just the chosen operating point, so a reader can see
    how sharp the optimum is and how far the ensemble sits above or below its
    own members.
    """
    s = sweep[sweep["archetype"] == archetype]
    if s.empty:
        return None

    line = alt.Chart(s).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(
        size=45, filled=True), color=theme.AGENT_COLOUR["Ensemble"]).encode(
        x=alt.X("hold_bias:Q", title="hold bias  (negative = send more freely)"),
        y=alt.Y("reward_mean:Q", title="mean reward per episode"),
        tooltip=[alt.Tooltip("hold_bias:Q", format="+.2f"),
                 alt.Tooltip("reward_mean:Q", title="reward", format=".2f"),
                 alt.Tooltip("sends_per_episode:Q", title="sends/wk", format=".2f"),
                 alt.Tooltip("ctr:Q", format=".3f")],
    )
    layers = [line]

    best = s.loc[s["reward_mean"].idxmax()]
    layers.append(alt.Chart(pd.DataFrame({"x": [best["hold_bias"]]})).mark_rule(
        strokeWidth=1.5, strokeDash=[4, 3],
        color=theme.AGENT_COLOUR["Ensemble"]).encode(x="x:Q"))

    if members is not None and not members.empty:
        m = members.copy()
        m["agent_c"] = m["member"].map(theme.canonical)
        layers.append(alt.Chart(m).mark_rule(strokeWidth=1.5, opacity=0.8).encode(
            y="reward_mean:Q",
            color=alt.Color("agent_c:N", scale=_scale(m["agent_c"]),
                            legend=alt.Legend(title="member", orient="right")),
            tooltip=[alt.Tooltip("member:N"),
                     alt.Tooltip("reward_mean:Q", title="reward", format=".2f")]))
        layers.append(alt.Chart(m).mark_text(
            align="left", dx=4, dy=-6, fontSize=10,
            color=theme.TEXT_SECONDARY).encode(
            y="reward_mean:Q",
            x=alt.value(4), text="member:N"))

    return alt.layer(*layers).properties(
        height=340, title=f"Ensemble reward against hold bias - {archetype}")


def action_heatmap(grid: pd.DataFrame, title: str):
    """Chosen action over hour-of-day against fatigue.

    A categorical heatmap, so the colour scale is the three-action palette with
    Hold as a recessive neutral: the eye should land on where the policy sends,
    not on the silence that dominates most of the surface.
    """
    if grid.empty:
        return None
    return alt.Chart(grid).mark_rect().encode(
        x=alt.X("hour:O", title="hour of day",
                axis=alt.Axis(values=list(range(0, 24, 2)))),
        y=alt.Y("fatigue:Q", title="fatigue", bin="binned",
                scale=alt.Scale(reverse=True)),
        color=alt.Color("action:N",
                        scale=alt.Scale(domain=theme.ACTION_ORDER,
                                        range=[theme.ACTION_COLOUR[a]
                                               for a in theme.ACTION_ORDER]),
                        legend=alt.Legend(title="action", orient="right")),
        tooltip=[alt.Tooltip("hour:O"), alt.Tooltip("fatigue:Q", format=".2f"),
                 alt.Tooltip("action:N")],
    ).properties(height=300, title=title)


def head_to_head_matrix(matrix: pd.DataFrame):
    """Pairwise reward difference, row agent minus column agent.

    The leaderboard answers "who scored highest"; this answers "who beats whom,
    and is the gap bigger than the seed noise". Those come apart: an agent can
    top the mean while losing to a rival on most archetypes, because one
    archetype it happens to suit carries its average.

    Cells are paired over the (archetype, seed) cells both agents were
    evaluated on, so the comparison never rests on cells only one of them ran.
    A dot marks a difference significant at p < 0.05 on a paired t-test.
    """
    if matrix.empty:
        return None

    order = [a for a in theme.AGENT_ORDER if a in set(matrix["row"])]
    lim = float(matrix["delta"].abs().max()) or 1.0

    base = alt.Chart(matrix).encode(
        x=alt.X("col:N", title=None, sort=order,
                axis=alt.Axis(labelAngle=-35, orient="top")),
        y=alt.Y("row:N", title=None, sort=order),
    )
    cells = base.mark_rect(stroke=theme.SURFACE, strokeWidth=2).encode(
        # Diverging, centred on zero: the sign is the message, so the midpoint
        # has to be pinned rather than inferred from the data range.
        color=alt.Color("delta:Q",
                        scale=alt.Scale(scheme="blueorange",
                                        domain=[-lim, 0, lim]),
                        legend=alt.Legend(title="row - col reward")),
        tooltip=[alt.Tooltip("row:N", title="agent"),
                 alt.Tooltip("col:N", title="versus"),
                 alt.Tooltip("delta:Q", title="reward delta", format="+.2f"),
                 alt.Tooltip("p:Q", title="paired p", format=".4f"),
                 alt.Tooltip("n:Q", title="paired cells")],
    )
    # Label colour flips on the dark ends of the diverging ramp so the number
    # stays readable at both extremes.
    labels = base.mark_text(fontSize=11, fontWeight=500).encode(
        text=alt.Text("label:N"),
        color=alt.condition(f"abs(datum.delta) > {0.55 * lim}",
                            alt.value("#ffffff"), alt.value(theme.TEXT_PRIMARY)),
    )
    n = len(order)
    return (cells + labels).properties(
        width=64 * n, height=48 * n,
        title="Head to head - mean reward difference, row minus column",
    )


def ensemble_bars(table: pd.DataFrame, title: str):
    """Individual members against the ensemble schemes, ranked by reward.

    Colour encodes *scheme*, not identity, which is the opposite of every other
    chart here and deliberate: the question this chart answers is "did combining
    them beat the parts", so the eye needs members to read as one group and the
    ensembles to stand out against them. Agent identity is carried by the axis
    label, which is a word rather than a code and needs no colour key.
    """
    if table.empty:
        return None
    t = table.copy()
    order = t.sort_values("reward_mean", ascending=False)["policy"].tolist()

    lo = min(float(t["reward_mean"].min()), 0.0)
    hi = max(float(t["reward_mean"].max()), 0.0)
    pad = 0.20 * max(hi - lo, 1.0)
    xscale = alt.Scale(domain=[lo - pad, hi + pad], nice=False)

    base = alt.Chart(t).encode(
        y=alt.Y("policy:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=12, labelColor=theme.TEXT_PRIMARY)),
    )
    bars = base.mark_bar(height=16, cornerRadiusEnd=4).encode(
        x=alt.X("reward_mean:Q", title="mean reward per episode", scale=xscale),
        color=alt.Color(
            "scheme:N",
            scale=alt.Scale(domain=["member", "vote", "gated"],
                            range=[theme.TEXT_MUTED, "#8a6bbf",
                                   theme.AGENT_COLOUR["Ensemble"]]),
            legend=alt.Legend(title="policy type", orient="bottom")),
        tooltip=[alt.Tooltip("policy:N"),
                 alt.Tooltip("reward_mean:Q", title="reward", format=".2f"),
                 alt.Tooltip("ctr:Q", format=".3f"),
                 alt.Tooltip("sends_per_episode:Q", title="sends/wk",
                             format=".2f"),
                 alt.Tooltip("optout_rate:Q", title="opt-out", format=".3f")],
    )
    # Same pos/neg anchor split as leaderboard_bars: `align` is a mark property
    # and cannot be bound to a field, so a negative bar's label would otherwise
    # print on top of its own fill.
    label_layers = []
    for frame, anchor, dx in ((t[t["reward_mean"] >= 0], "left", 7),
                              (t[t["reward_mean"] < 0], "right", -7)):
        if frame.empty:
            continue
        label_layers.append(alt.Chart(frame).mark_text(
            align=anchor, dx=dx, fontSize=11, color=theme.TEXT_SECONDARY
        ).encode(y=alt.Y("policy:N", sort=order, title=None),
                 x=alt.X("reward_mean:Q", scale=xscale),
                 text=alt.Text("reward_mean:Q", format=".2f")))

    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeWidth=1, color=theme.TEXT_MUTED, strokeDash=[3, 3]).encode(
        x=alt.X("x:Q", scale=xscale))

    return alt.layer(zero, bars, *label_layers).properties(
        width=470, height=44 * len(t) + 40, title=title)

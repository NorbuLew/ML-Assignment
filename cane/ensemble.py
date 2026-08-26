"""Ensemble agents that combine the trained CANE policies.

Why this is not a routine ensemble
----------------------------------
The usual argument for ensembling -- members make uncorrelated errors, so
averaging cancels them -- does not hold here. Measured on the held-out episodes,
the members are *anti*-correlated in a specific and awkward way:

    LinUCB          39.2 sends/week, reward -22.85   chronic over-sender
    DQN / DDQN / PPO 0.0 sends/week, reward   0.00   unconditional never-sender
    Fixed-18:00      7.0 sends/week, reward  +2.01   beats every learner on
                                                     OfficeWorker

LinUCB optimises a myopic break-even (send whenever immediate expected click
value is positive) and therefore over-sends. The deep agents see the full
discounted cost of a send -- fatigue accumulates at kappa/(1-lam) and churn is
terminal -- correctly conclude that few hours clear the ~0.34 break-even click
rate, and abstain everywhere. Both are locally rational; both are wrong.

That makes plain voting useless: with three never-senders in a four-member pool,
majority vote *is* AlwaysHold, and scores exactly 0.00. `MajorityVoteEnsemble`
below exists to demonstrate that rather than to win, because the failure is the
interesting part and it is the direct evidence that a naive combiner is
inadequate here.

The combiner that can work has to do something a vote cannot: place the decision
threshold *between* the members' biases instead of at whatever point the
majority happens to sit. `GatedEnsemble` does that with a single learned scalar
`hold_bias`, fitted on a validation split that is disjoint from the reported
test episodes.

References: Kuncheva, *Combining Pattern Classifiers* (combiner + decision
threshold); Kittler et al. 1998 (combination rules).
"""

from __future__ import annotations

import numpy as np

from cane.core import (
    HOLD, N_ACTIONS, Agent, CANEEnv, LinUCBAgent, run_episodes,
)

# Validation episodes for fitting ensemble weights, gates and the hold bias.
# Deliberately disjoint from EVAL_SEEDS (900_000..900_199) so that nothing the
# ensemble is tuned on appears in a reported number.
VAL_SEEDS = list(range(920_000, 920_100))

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------
# Each family exposes a per-action preference in its own units: LinUCB a ridge
# mean, the value-based agents a Q-value, PPO a policy logit. These are read
# from the agents directly rather than by adding a method to them, because the
# agent code is generated verbatim from the notebooks by
# tools/extract_package.py and would lose any hand-added method on the next
# extraction.

def member_scores(agent, state) -> np.ndarray:
    """Per-action preference for `agent` in its own native units.

    Returns an array of length N_ACTIONS. Non-learning baselines have no
    meaningful score surface, so they report a one-hot on the action they would
    have taken -- which lets a fixed schedule participate in a vote without
    pretending to have a value function.
    """
    if isinstance(agent, LinUCBAgent):
        x = agent.encoder(state)
        return np.array(
            [float((agent.A_inv[a] @ agent.b[a]) @ x) for a in range(agent.n_actions)]
        )

    # Deep agents. Imported lazily so that a CSV-only consumer of this module
    # (the dashboard's leaderboard, for instance) never has to import torch.
    net = getattr(agent, "online_net", None)
    if net is not None:                                    # DQN / Double DQN
        import torch
        with torch.no_grad():
            x = torch.from_numpy(agent._features(state)).unsqueeze(0)
            return net(x).squeeze(0).numpy().astype(float)

    net = getattr(agent, "net", None)
    if net is not None:                                    # PPO
        import torch
        with torch.no_grad():
            x = torch.from_numpy(agent._features(state)).unsqueeze(0)
            logits, _ = net(x)
            return logits.squeeze(0).numpy().astype(float)

    action, _ = agent.act(state, greedy=True)              # baseline
    onehot = np.zeros(N_ACTIONS)
    onehot[action] = 1.0
    return onehot


def _znorm(scores: np.ndarray) -> np.ndarray:
    """Standardise one member's scores across actions.

    Q-values live in reward units (tens), PPO logits are unitless and of order
    one, and LinUCB's ridge means are one-step reward estimates. Averaging them
    raw would let whichever member has the largest numeric range dictate the
    outcome regardless of how confident it actually is, so each member is put on
    a common scale first.
    """
    sd = scores.std()
    if sd < _EPS:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / sd


def _spread(scores: np.ndarray) -> float:
    """Gap between a member's best and second-best action, in native units."""
    if scores.size < 2:
        return 0.0
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

class MajorityVoteEnsemble(Agent):
    """Hard majority vote over member actions. The negative control.

    Ties resolve to Hold: in a channel where an unwanted send costs more than a
    missed opportunity, disagreement should default to silence.

    This is reported precisely because it is expected to fail. With most members
    being unconditional never-senders, the vote degenerates to AlwaysHold. The
    `disagreement_rate` recorded during evaluation is what makes that visible
    rather than merely asserted.
    """

    def __init__(self, members, labels=None, label=None):
        self.members = list(members)
        self.labels = list(labels) if labels else [m.name for m in self.members]
        self._label = label
        self.n_disagree = 0
        self.n_steps = 0

    @property
    def name(self):
        return self._label or "Ensemble-Vote"

    def act(self, state, greedy=False):
        votes = [int(m.act(state, greedy=True)[0]) for m in self.members]
        counts = np.bincount(votes, minlength=N_ACTIONS)
        self.n_steps += 1
        if len(set(votes)) > 1:
            self.n_disagree += 1
        top = counts.max()
        winners = np.flatnonzero(counts == top)
        action = HOLD if (len(winners) > 1 and HOLD in winners) else int(winners[0])
        return action, {"votes": votes}

    def update(self, state, action, reward, next_state, done, aux):
        return {}

    def reset(self):
        for m in self.members:
            m.reset()

    @property
    def disagreement_rate(self):
        return self.n_disagree / max(self.n_steps, 1)


class GatedEnsemble(Agent):
    """Confidence-gated, z-normalised score fusion with a learned hold bias.

    For each state, member `m` contributes

        w_m * c_m(s) * znorm(scores_m(s))

    where `w_m` weights members by validation reward and the gate

        c_m(s) = min(1, spread_m(s) / sigma_m)

    scales a member down when it is close to indifferent. The gate matters:
    z-normalising alone *amplifies* noise, because two nearly-tied scores get
    stretched to +/-1.41 and an indifferent member ends up shouting. `sigma_m`
    is that member's median spread over the validation split, so a member
    contributes fully only when it is more decisive than it usually is.

    A single scalar `hold_bias` is then added to the Hold score. This is the
    part that addresses the real problem: the members disagree about *rate*
    rather than about *timing*, so what the combiner needs is not a better vote
    but a movable threshold between the over-sender and the never-senders.
    Negative values make the ensemble send more freely.
    """

    def __init__(self, members, weights=None, gates=None, hold_bias=0.0,
                 labels=None, label=None):
        self.members = list(members)
        self.labels = list(labels) if labels else [m.name for m in self.members]
        n = len(self.members)
        self.weights = np.ones(n) / n if weights is None else np.asarray(weights, float)
        self.gates = np.ones(n) if gates is None else np.asarray(gates, float)
        self.hold_bias = float(hold_bias)
        self._label = label

    @property
    def name(self):
        return self._label or "Ensemble-Gated"

    def combine(self, state) -> np.ndarray:
        """Fused per-action score. Exposed so the dashboard can show the
        contribution of each member at a probed state, through exactly the code
        path used to act."""
        total = np.zeros(N_ACTIONS)
        for m, w, sigma in zip(self.members, self.weights, self.gates):
            raw = member_scores(m, state)
            gate = 1.0 if sigma <= _EPS else min(1.0, _spread(raw) / sigma)
            total += w * gate * _znorm(raw)
        total[HOLD] += self.hold_bias
        return total

    def act(self, state, greedy=False):
        fused = self.combine(state)
        return int(np.argmax(fused)), {"fused": fused.tolist()}

    def update(self, state, action, reward, next_state, done, aux):
        return {}

    def reset(self):
        for m in self.members:
            m.reset()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_gates(members, archetype=None, seeds=None, env_seed=8100) -> np.ndarray:
    """Median decision spread per member, measured on the validation split.

    This is the `sigma_m` that normalises the confidence gate. A member whose
    scores are usually near-tied gets a small sigma and so is not penalised for
    being habitually indecisive; one that is usually emphatic must be unusually
    emphatic to contribute fully.
    """
    seeds = VAL_SEEDS if seeds is None else seeds
    spreads = [[] for _ in members]
    for seed in seeds:
        env = CANEEnv(seed=env_seed, archetype=archetype)
        state, _ = env.reset(seed=seed)
        done = False
        while not done:
            for i, m in enumerate(members):
                spreads[i].append(_spread(member_scores(m, state)))
            action, _ = members[0].act(state, greedy=True)
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return np.array([np.median(s) if s else 1.0 for s in spreads])


def fit_weights(val_rewards, temperature=None) -> np.ndarray:
    """Softmax the members' validation rewards into mixture weights.

    A member that collapsed to never-send scores 0.00 while a working member
    scores tens, so this pushes weight towards whichever members actually earned
    something -- without hard-selecting one and discarding the rest.
    """
    r = np.asarray(val_rewards, dtype=float)
    if temperature is None:
        temperature = max(r.std(), 1.0)
    z = (r - r.max()) / temperature
    w = np.exp(z)
    return w / w.sum()


def sweep_hold_bias(members, weights, gates, biases=None, archetype=None,
                    seeds=None, env_seed=8100, labels=None):
    """Line-search the hold bias on the validation split.

    Returns a list of dicts, one per candidate bias, each carrying the reward
    and the send rate. Reporting the whole curve rather than only the winner is
    what shows the trade-off: the ensemble's reward as the threshold slides from
    'never send' to 'always send', with each member's reward as a reference
    line.
    """
    biases = np.linspace(-2.0, 2.0, 21) if biases is None else np.asarray(biases, float)
    seeds = VAL_SEEDS if seeds is None else seeds
    out = []
    for b in biases:
        ens = GatedEnsemble(members, weights=weights, gates=gates,
                            hold_bias=float(b), labels=labels)
        env = CANEEnv(seed=env_seed, archetype=archetype)
        m, _, _, _ = run_episodes(ens, env, seeds=seeds, learn=False, greedy=True)
        out.append({"hold_bias": float(b),
                    "reward_mean": float(m["reward_mean"]),
                    "ctr": float(m["ctr"]),
                    "sends_per_episode": float(m["sends_per_episode"]),
                    "optout_rate": float(m["optout_rate"])})
    return out

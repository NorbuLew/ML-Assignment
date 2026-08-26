"""Minimum-contact agents: coverage guaranteed, timing still learned.

Three of the five archetypes receive nothing from every deep RL agent. That is
not acceptable as a product -- a notification channel that goes silent for a
whole week on 60% of users has stopped being a notification channel -- and it is
not even reward-optimal: an exhaustive search over one-line "send X at hour H
daily" policies beats never-send on *all five* archetypes (+7.53 OfficeWorker,
+2.64 NightShiftWorker, +0.54 NormalStudent).

So silence is removed as an option, and nothing else is.

The wrapper enforces a quota of `min_sends` per episode by masking the Hold
action once the remaining hours are only just enough to fit the remaining
quota. Until that deadline the agent is completely free; the quota changes
*whether* it sends, never *when*. Everything it has to learn -- which hour,
which message type, how the choice depends on fatigue and recency -- is
untouched, and so is the reward function, the environment and the network.

This is the standard budgeted / constrained-MDP formulation: the constraint is
on the policy's feasible set, not on its objective.

    from cane.min_contact import MinContactAgent
    agent = MinContactAgent(DDQNAgent(seed=0), min_sends=1, window=24)
"""

from __future__ import annotations

import numpy as np

from cane.core import ENGAGE, HOLD, N_ACTIONS


class MinContactAgent:
    """Wraps any agent, guaranteeing a per-episode send quota.

    The wrapper is deliberately thin. It forwards `update` untouched, so the
    inner agent learns from exactly the transitions it experienced, including
    the forced ones -- which is what lets it discover that a forced send at a
    bad hour was expensive and a forced send at a good hour paid.
    """

    def __init__(self, inner, min_sends: int = 1, window: int = 24,
                 horizon: int = 168, label: str | None = None):
        self.inner = inner
        self.min_sends = int(min_sends)
        self.window = int(window)
        self.horizon = int(horizon)
        self._label = label
        self.reset()

    # -- bookkeeping --------------------------------------------------------
    def reset(self):
        self.t = 0
        self.sent_in_window = 0
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    @property
    def name(self):
        base = getattr(self.inner, "name", type(self.inner).__name__)
        base = base() if callable(base) else base
        return self._label or f"{base}+min{self.min_sends}/{self.window}h"

    # -- policy -------------------------------------------------------------
    def _quota_forces_send(self) -> bool:
        """True when holding now would miss this window's quota.

        The quota is per rolling `window` (a day by default), not per episode.
        A whole-week deadline looks equivalent but is not: with one deadline at
        the end, a silent agent takes every send in the final hours of the week,
        which lands on whatever clock hour the episode happens to reach then.
        That is an artifact of the deadline, not a choice, and it destroys the
        thing being measured -- which hour the agent picks for this person.

        Per day, the agent has 24 hours to choose from and is forced only in the
        last hour it could still satisfy the quota. Its freedom is unchanged
        within the day; only the option of never sending is removed.
        """
        hours_left_in_window = self.window - (self.t % self.window)
        remaining_quota = self.min_sends - self.sent_in_window
        return remaining_quota > 0 and hours_left_in_window <= remaining_quota

    def act(self, state, greedy=False):
        action, aux = self.inner.act(state, greedy=greedy)

        if self._quota_forces_send() and action == HOLD:
            # Forced to send: let the inner agent choose *which* message, using
            # its own values. Only the hold option is removed.
            q = None
            if isinstance(aux, dict):
                q = aux.get("q_values") or aux.get("logits")
            if q is not None and len(q) == N_ACTIONS:
                action = int(np.argmax(np.asarray(q[1:], dtype=float))) + 1
            else:
                action = ENGAGE
            if isinstance(aux, dict):
                aux = {**aux, "forced": True}

        self.t += 1
        self.sent_in_window += int(action != HOLD)
        if self.t % self.window == 0:
            self.sent_in_window = 0
        return action, aux

    def update(self, state, action, reward, next_state, done, aux):
        out = self.inner.update(state, action, reward, next_state, done, aux)
        if done:
            self.t = 0
            self.sent_in_window = 0
        return out

    # -- passthrough --------------------------------------------------------
    def state_dict_for_save(self):
        return self.inner.state_dict_for_save()

    def __getattr__(self, item):
        # Only reached for attributes this class does not define, so it cannot
        # shadow act/update/reset above.
        return getattr(self.__dict__["inner"], item)

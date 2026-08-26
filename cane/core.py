"""CANE core: environment, agent contract, baselines, harness, LinUCB.

GENERATED FILE -- do not edit by hand.

Produced by `tools/extract_package.py` from the code cells of
`Code/AssignementCode(LinUCB).ipynb` (cells 2, 4, 6, 10, 15, 16, 17, 18), copied verbatim and in
notebook execution order. Editing this file directly would make the package and
the notebook disagree about what was actually run; edit the notebook and re-run
the extractor instead.

Top-level `print(...)` progress lines from the notebook are removed so that
importing the package is silent.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

# ==========================================================================
# notebook cell 2
# ==========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


# --- Action space -----------------------------------------------------------
# Defined once and imported everywhere so the agents, the environment and the
# plotting code can never disagree about what "1" means.
HOLD = 0        # stay silent -> fatigue decays
ENGAGE = 1      # engagement nudge (social / community update)
INCENTIVE = 2   # incentive nudge (high-value promo reward)

N_ACTIONS = 3
ACTION_NAMES = {HOLD: "Hold", ENGAGE: "Engage", INCENTIVE: "Incentive"}


# --- Observation layout -----------------------------------------------------
# The env returns s = [hour, day, active, fatigue, recency] as a flat float32
# array. These constants are the single source of truth for the column order;
# if the env owner reorders the vector, only this block changes.
IDX_HOUR = 0        # 0-23, RAW (not sin/cos) - see the note above
IDX_DAY = 1         # 0-6, Monday = 0
IDX_ACTIVE = 2      # 1 if the user is currently in-app
IDX_FATIGUE = 3     # [0, 1] accumulated notification fatigue
IDX_RECENCY = 4     # [0, 1] normalised time since the last send

N_BLOCKS = 8                 # 3-hour blocks; separates the 19:00 and 23:00 peaks
BLOCK_HOURS = 24 // N_BLOCKS

IDX_ACT_RATE = 5             # 5..12  smoothed in-app activity rate per block
IDX_CLICK_RATE = 13          # 13..20 smoothed click rate per block
IDX_TYPE_CR = 21             # 21,22  click rate for Engage / Incentive sends
IDX_SINCE_CLICK = 23         # normalised hours since the last click

STATE_DIM_BASE = 5
STATE_DIM = 24

BELIEF_PRIOR = (1.0, 4.0)    # Beta(a, b); prior mean 0.2 before any evidence
USE_BELIEF = True            # False reproduces the memoryless 5-feature ablation


def agent_state_dim():
    return STATE_DIM if USE_BELIEF else STATE_DIM_BASE


# ==========================================================================
# notebook cell 4
# ==========================================================================

class Agent(ABC):
    """Shared contract implemented by every CANE agent.

    `act` and `update` are abstract: a subclass that omits either one cannot be
    instantiated, so an incomplete agent fails loudly at construction rather
    than silently producing a policy that never learns.
    """

    @property
    def name(self):
        """Label used in results tables and plot legends.

        Defaults to the class name. LinUCB overrides this, because the feature
        variants (Raw / Harmonic / One-hot) are the same class and would
        otherwise collapse into a single indistinguishable row.
        """
        return self.__class__.__name__

    @abstractmethod
    def act(self, state, greedy=False):
        """Choose an action for the given state.

        Args:
            state: observation array from the env (see the index constants).
            greedy: act without exploration. Used for all reported evaluation
                runs, so the numbers reflect the learned policy rather than
                exploration noise.

        Returns:
            (action, aux). `action` is an int in {HOLD, ENGAGE, INCENTIVE}.
            `aux` carries any per-step quantity the agent will need back at
            update time, and is empty for agents that need nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done, aux):
        """Learn from one transition.

        The full transition is always passed and each agent takes what it
        needs; LinUCB uses only `state`, `action` and `reward`.

        Returns:
            dict of diagnostics for the logger (may be empty).
        """
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode. No-op by default.

        LinUCB must NOT reset here: its A and b matrices accumulate across
        every episode, and that accumulation is the whole of its learning.
        Clearing them per episode would silently reduce it to a random policy.
        On-policy agents such as PPO do override this, to clear their buffer.
        """
        pass


# ==========================================================================
# notebook cell 6
# ==========================================================================

def _harmonics(hour, n_harmonics):
    """sin/cos pairs for the first `n_harmonics` harmonics of the 24h cycle.

    The k-th harmonic completes k full cycles per day, so including k = 1..3
    lets a linear model express up to three peaks per day instead of one.
    """
    out = []
    for k in range(1, n_harmonics + 1):
        theta = 2.0 * np.pi * k * hour / 24.0
        out.extend([np.sin(theta), np.cos(theta)])
    return out


def _shared_tail(state):
    """The non-temporal features, identical across all three variants.

    Keeping these in one place is what makes the ablation clean: the ONLY
    difference between the variants is how time-of-day is encoded.
    """
    feats = [
        1.0 if state[IDX_DAY] >= 5 else 0.0,   # weekend (Sat = 5, Sun = 6)
        float(state[IDX_ACTIVE]),
        float(state[IDX_FATIGUE]),
        float(state[IDX_RECENCY]),
    ]
    if USE_BELIEF:
        feats += list(np.asarray(state[IDX_ACT_RATE:IDX_ACT_RATE + N_BLOCKS], dtype=float))
        feats += list(np.asarray(state[IDX_CLICK_RATE:IDX_CLICK_RATE + N_BLOCKS], dtype=float))
        feats += [float(state[IDX_TYPE_CR]), float(state[IDX_TYPE_CR + 1])]
        feats.append(float(state[IDX_SINCE_CLICK]))
    feats.append(1.0)                           # intercept
    return feats


def encode_raw(state):
    """Variant 1 - the proposal as written. d = 7.

    A linear model over (sin h, cos h) is a single sinusoid, so this can
    represent exactly one receptive period per day.
    """
    feats = _harmonics(state[IDX_HOUR], 1) + _shared_tail(state)
    return np.asarray(feats, dtype=np.float64)


def encode_harmonic(state):
    """Variant 2 - richer Fourier basis. d = 11.

    Harmonics up to k = 3 can express up to three peaks per day, which is what
    the Office Worker archetype (commute / lunch / evening) actually requires.
    """
    feats = _harmonics(state[IDX_HOUR], 3) + _shared_tail(state)
    return np.asarray(feats, dtype=np.float64)


def encode_onehot(state):
    """Variant 3 - unconstrained hourly indicators. d = 29.

    Each hour gets its own free parameter, so ANY hourly pattern is
    representable. This is the variant that decides whether LinUCB's shortfall
    is representational or structural.
    """
    hours = [0.0] * 24
    hours[int(state[IDX_HOUR]) % 24] = 1.0
    feats = hours + _shared_tail(state)
    return np.asarray(feats, dtype=np.float64)


def encode_interaction(state):
    """Variant 4 - hour indicators crossed with the belief block.

    A linear model over additive features cannot express "send at 23:00 IF this
    user's late-evening activity is high": that rule is a product of two
    features. Supplying the products explicitly removes the hypothesis class as
    a confound, so any remaining shortfall against the deep methods is
    attributable to the bandit formulation rather than to the encoding.
    """
    hours = np.zeros(24)
    hours[int(state[IDX_HOUR]) % 24] = 1.0
    feats = list(hours)
    if USE_BELIEF:
        act = np.asarray(state[IDX_ACT_RATE:IDX_ACT_RATE + N_BLOCKS], dtype=float)
        feats += list(np.outer(hours, act).ravel())
    feats += _shared_tail(state)
    return np.asarray(feats, dtype=np.float64)


# --- Registry ---------------------------------------------------------------
# Lets experiments select an encoding by name from config, so the ablation is a
# loop over keys rather than three near-duplicate code paths.
FEATURE_ENCODERS = {
    "raw": encode_raw,
    "harmonic": encode_harmonic,
    "onehot": encode_onehot,
    "interaction": encode_interaction,
}


def feature_dim(encoder):
    """Infer d by probing the encoder with a dummy state.

    Derived rather than hard-coded: if an encoder is later edited, d follows
    automatically instead of silently disagreeing with a stale constant.
    """
    probe = np.zeros(STATE_DIM, dtype=np.float32)
    return len(encoder(probe))


# ==========================================================================
# notebook cell 10
# ==========================================================================

class LinUCBAgent(Agent):
    """Disjoint LinUCB contextual bandit (Li et al., 2010).

    Serves as the deliberately stateless baseline in the CANE comparison. It
    can see fatigue in its context vector, but has no transition model, so it
    cannot represent the fact that its own send is what *causes* fatigue later.

    Efficiency note: A^-1 is maintained incrementally via the Sherman-Morrison
    identity rather than re-solving a d x d system at every step. Because each
    update is a rank-1 modification A <- A + x x^T, the inverse can be updated
    directly:

        (A + x x^T)^-1 = A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)

    This reduces per-step cost from O(d^3) to O(d^2), which matters most for the
    one-hot encoding (d = 29) and makes the alpha sweep tractable. `self.A` is
    still maintained so the cached inverse can be checked against an exact solve.
    """

    def __init__(self, encoder_name="raw", alpha=1.0, lam=1.0,
                 n_actions=N_ACTIONS, seed=0, label=None):
        """
        Args:
            encoder_name: key into FEATURE_ENCODERS ("raw"/"harmonic"/"onehot").
            alpha: exploration coefficient. 0 = greedy ridge regression, which
                is a useful ablation for separating the value of exploration
                from the value of the linear model itself.
            lam: ridge parameter. Scales the initial A = lam * I, keeping it
                invertible before any data and setting the initial confidence
                width.
            seed: controls tie-breaking only; the agent is otherwise
                deterministic given its data.
            label: overrides the display name in results tables.
        """
        self.encoder_name = encoder_name
        self.encoder = FEATURE_ENCODERS[encoder_name]
        self.d = feature_dim(self.encoder)

        self.alpha = float(alpha)
        self.lam = float(lam)
        self.n_actions = int(n_actions)
        self._label = label
        self.rng = np.random.default_rng(seed)

        self._init_stats(self.d)

    def _init_stats(self, d):
        """Allocate per-arm sufficient statistics.

        A starts at lam*I so it is invertible with zero observations; its
        inverse is therefore I/lam.
        """
        eye = np.eye(d)
        self.A = np.stack([eye * self.lam for _ in range(self.n_actions)])
        self.A_inv = np.stack([eye / self.lam for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, d))
        self.n_pulls = np.zeros(self.n_actions, dtype=int)

    @property
    def name(self):
        """Distinguishes the feature variants, which are all this same class."""
        return self._label or f"LinUCB-{self.encoder_name}"

    def act(self, state, greedy=False):
        """Score every arm and play the highest.

        Score = predicted reward + alpha * (confidence width).
        Under `greedy` the confidence term is dropped, leaving pure exploitation.
        """
        x = self.encoder(state)
        scores = np.empty(self.n_actions)

        for a in range(self.n_actions):
            A_inv = self.A_inv[a]
            theta = A_inv @ self.b[a]           # ridge estimate
            mean = float(theta @ x)
            if greedy:
                scores[a] = mean
            else:
                # max(..., 0) guards against tiny negative values from
                # floating-point error; the quadratic form is >= 0 in exact
                # arithmetic because A is positive definite.
                width = np.sqrt(max(float(x @ (A_inv @ x)), 0.0))
                scores[a] = mean + self.alpha * width

        # Random tie-breaking. At t=0 all arms score identically, and a plain
        # argmax would always return HOLD, biasing the opening decisions.
        best = np.flatnonzero(scores == scores.max())
        action = int(self.rng.choice(best)) if best.size > 1 else int(best[0])

        return action, {}

    def update(self, state, action, reward, next_state=None, done=False, aux=None):
        """Rank-1 ridge update on the PLAYED arm only.

        `next_state`, `done` and `aux` are accepted to satisfy the shared Agent
        interface and are deliberately unused: LinUCB has no transition model.
        That is precisely the limitation the RQ2 comparison is designed to expose.
        """
        x = self.encoder(state)

        # Only the played arm observed a reward. Updating any other arm would
        # invent evidence that was never collected.
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

        # Sherman-Morrison rank-1 inverse update.
        A_inv = self.A_inv[action]
        Ax = A_inv @ x
        self.A_inv[action] = A_inv - np.outer(Ax, Ax) / (1.0 + float(x @ Ax))

        self.n_pulls[action] += 1
        return {"arm": action, "reward": reward}

    def theta(self, action):
        """Current coefficient estimate for one arm (for the analysis section)."""
        return self.A_inv[action] @ self.b[action]

    def inverse_drift(self):
        """Largest discrepancy between the cached inverse and an exact solve.

        Sherman-Morrison accumulates floating-point error over many updates, so
        this is checked rather than assumed. Values around 1e-10 are expected.
        """
        return max(float(np.abs(self.A_inv[a] - np.linalg.inv(self.A[a])).max())
                   for a in range(self.n_actions))

    def __repr__(self):
        return (f"LinUCBAgent(encoder={self.encoder_name!r}, d={self.d}, "
                f"alpha={self.alpha}, lam={self.lam})")


# ==========================================================================
# notebook cell 15
# ==========================================================================

# --- Configuration ----------------------------------------------------------
# Every tunable lives here so experiments can sweep them without touching the
# environment code, and so the reward-weight sensitivity study has one target.

# Relevance weight of each message type, per archetype. An Incentive Nudge is a
# discount, so it lands harder with price-sensitive users (students, deal-seekers)
# and adds little for time-poor ones who value relevance over savings.
ARCHETYPE_W = {
    "OfficeWorker":     {ENGAGE: 1.15, INCENTIVE: 1.05},
    "NightOwlStudent":  {ENGAGE: 0.90, INCENTIVE: 1.60},
    "NightShiftWorker": {ENGAGE: 1.00, INCENTIVE: 1.25},
    "NormalStudent":    {ENGAGE: 0.90, INCENTIVE: 1.55},
    "Housewife":        {ENGAGE: 1.05, INCENTIVE: 1.45},
}

CANE_CONFIG = dict(
    # Episode structure
    steps_per_episode=168,      # 1 step = 1 hour, 1 episode = 1 week

    # Fatigue dynamics:  F' = clip(lam*F + kappa*1{a != 0}, 0, 1)
    lam=0.90,                   # recovery factor (proposal §4b)
    kappa={HOLD: 0.0, ENGAGE: 0.12, INCENTIVE: 0.18},   # Incentive is pushier                 # per-send increment (proposal §4b)

    # Click model:  P = clip(p0(arch,h,d) * (1 - mu*F) * w(a), 0, 1)
    mu=0.80,                    # fatigue attenuation strength
    w_arch=ARCHETYPE_W,

    # Churn model:  P(opt-out | F) = sigmoid(g0 + g1*F), only above threshold
    churn_threshold=0.70,
    gamma_0=-7.0,
    gamma_1=4.0,

    # Reward weights (proposal §4e). Ordering: R_churn >> R_click > W_fat > W_send
    R_click=10.0,
    W_send={HOLD: 0.0, ENGAGE: 1.0, INCENTIVE: 2.5},    # a promo has real margin cost
    W_fat=2.0,
    R_churn=30.0,

    # Retention-streak shaping
    beta=1.15,
    streak_mode="marginal",     # "marginal" (beta**n) or "cumulative" (sum)
    streak_cap=5.0,             # ceiling; None disables. Protects the reward ordering.
    streak_lapse_hours=36,      # no click for this long -> streak resets

    # In-app activity: probability the user is already in the app, which scales
    # with their current receptiveness (an engaged user is more likely present).
    active_scale=0.6,
)


# --- Archetype responsiveness curves ----------------------------------------
# Each archetype is a sum of Gaussian "receptive windows" over the 24h clock.
# Amplitudes are set so peak baseline click propensity sits around 0.35-0.50,
# which after fatigue attenuation and the message weight yields open rates in a
# plausible range rather than an implausibly generous one.

ARCHETYPES = ["OfficeWorker", "NightOwlStudent", "NightShiftWorker",
              "NormalStudent", "Housewife"]


def _bump(hours, centre, width, amp):
    """A Gaussian receptive window, wrapped correctly around midnight."""
    delta = (hours - centre + 12.0) % 24.0 - 12.0
    return amp * np.exp(-0.5 * (delta / width) ** 2)


def archetype_curve(archetype, hours=None):
    """Baseline click propensity p0 by hour for one archetype."""
    h = np.arange(24.0) if hours is None else np.asarray(hours, dtype=float)

    if archetype == "OfficeWorker":
        # Commute, lunch, evening. Suppressed through working hours.
        c = _bump(h, 7.5, 1.0, 0.42) + _bump(h, 12.5, 1.1, 0.30) + _bump(h, 19.5, 2.0, 0.50)
    elif archetype == "NightOwlStudent":
        # Peaks 22:00-02:00, quiet all morning.
        c = _bump(h, 0.0, 2.2, 0.32) + _bump(h, 22.0, 1.6, 0.26) + _bump(h, 16.0, 2.0, 0.10)
    elif archetype == "NightShiftWorker":
        # Phase-inverted: before the shift (late afternoon), after it (early am).
        c = _bump(h, 16.5, 1.6, 0.45) + _bump(h, 6.5, 1.5, 0.40) + _bump(h, 2.0, 1.8, 0.18)
    elif archetype == "NormalStudent":
        # After school, then an evening study break. Asleep by midnight.
        c = _bump(h, 16.0, 1.6, 0.45) + _bump(h, 21.0, 1.5, 0.42) + _bump(h, 7.0, 1.0, 0.20)
    elif archetype == "Housewife":
        # The most time-flexible: broad daytime availability, family-hours dip.
        c = _bump(h, 10.0, 2.4, 0.42) + _bump(h, 14.5, 2.4, 0.40) + _bump(h, 20.5, 1.6, 0.22)
    else:
        raise ValueError(f"unknown archetype: {archetype}")

    return np.clip(c, 0.0, 1.0)


# Precomputed lookup: (archetype, hour) -> p0. Avoids recomputing exponentials
# 100k+ times during training.
ARCHETYPE_TABLE = {a: archetype_curve(a) for a in ARCHETYPES}

# Weekend modulation. Work- and school-bound routines loosen at the weekend;
# the Housewife and Night-Owl routines are largely unaffected.
WEEKEND_FLATTEN = {"OfficeWorker": 0.55, "NormalStudent": 0.50,
                   "NightShiftWorker": 0.30, "NightOwlStudent": 0.15,
                   "Housewife": 0.05}


# ==========================================================================
# notebook cell 16
# ==========================================================================

class CANEEnv:
    """Context-Aware Notification Engine simulator.

    Gymnasium-style API (`reset` -> (obs, info); `step` -> (obs, reward,
    terminated, truncated, info)) but without a gymnasium dependency, so the
    LinUCB notebook runs standalone.

    NOTE for the DQN / PPO implementations: to use this with stable-baselines3
    it must subclass `gymnasium.Env` and declare
        observation_space = spaces.Box(low, high, shape=(STATE_DIM,), float32)
        action_space      = spaces.Discrete(N_ACTIONS)
    The dynamics below are unchanged by that; only the declarations are added.
    """

    def __init__(self, config=None, archetype=None, seed=0):
        """
        Args:
            config: overrides for CANE_CONFIG.
            archetype: pin a single archetype (for per-archetype evaluation).
                If None, one is drawn uniformly at each reset.
            seed: RNG seed. Evaluation uses reserved seeds so that every agent
                is scored on the identical sequence of users and random draws.
        """
        self.cfg = {**CANE_CONFIG, **(config or {})}
        self.fixed_archetype = archetype
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- helpers -------------------------------------------------------------

    def _p0(self):
        """Baseline click propensity for the current archetype, hour and day."""
        base = ARCHETYPE_TABLE[self.archetype][self.hour]
        if self.day >= 5:
            # At the weekend a work-bound routine flattens toward its own mean:
            # the peaks soften and the troughs lift.
            f = WEEKEND_FLATTEN[self.archetype]
            base = (1.0 - f) * base + f * ARCHETYPE_TABLE[self.archetype].mean()
        return float(base)

    def _observe(self):
        """Assemble the observation vector. Hour is RAW; encoding is the agent's job."""
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[IDX_HOUR] = self.hour
        s[IDX_DAY] = self.day
        s[IDX_ACTIVE] = self.active
        s[IDX_FATIGUE] = self.fatigue
        s[IDX_RECENCY] = self.recency

        a0, b0 = BELIEF_PRIOR
        s[IDX_ACT_RATE:IDX_ACT_RATE + N_BLOCKS] = (
            (self.blk_act_hits + a0) / (self.blk_act_obs + a0 + b0))
        s[IDX_CLICK_RATE:IDX_CLICK_RATE + N_BLOCKS] = (
            (self.blk_clicks + a0) / (self.blk_sends + a0 + b0))
        s[IDX_TYPE_CR] = (self.type_clicks[ENGAGE] + a0) / (self.type_sends[ENGAGE] + a0 + b0)
        s[IDX_TYPE_CR + 1] = (self.type_clicks[INCENTIVE] + a0) / (self.type_sends[INCENTIVE] + a0 + b0)
        s[IDX_SINCE_CLICK] = min(self.hours_since_click / 48.0, 1.0)
        return s

    def _roll_activity(self):
        """Whether the user is already in-app; more likely when receptive.

        Matters because the CTR convention counts a click only when the user was
        inactive: sending to someone already using the app is treated as wasted.
        """
        p = min(1.0, self.cfg["active_scale"] * self._p0())
        self.active = int(self.rng.random() < p)

        b = self.hour // BLOCK_HOURS
        self.blk_act_obs[b] += 1
        self.blk_act_hits[b] += self.active

    # -- API -----------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.archetype = (self.fixed_archetype
                          or ARCHETYPES[self.rng.integers(len(ARCHETYPES))])

        self.t = 0
        self.hour = int(self.rng.integers(24))       # random start time of week
        self.day = int(self.rng.integers(7))
        self.fatigue = 0.0
        self.recency = 1.0                           # no send yet -> maximally stale
        self.hours_since_send = 24
        self.streak = 0
        self.hours_since_click = 0
        self.opted_out = False

        self.blk_act_obs = np.zeros(N_BLOCKS)
        self.blk_act_hits = np.zeros(N_BLOCKS)
        self.blk_sends = np.zeros(N_BLOCKS)
        self.blk_clicks = np.zeros(N_BLOCKS)
        self.type_sends = np.zeros(N_ACTIONS)
        self.type_clicks = np.zeros(N_ACTIONS)

        self._roll_activity()
        return self._observe(), {"archetype": self.archetype}

    def step(self, action):
        cfg = self.cfg
        action = int(action)
        sent = action != HOLD

        # Fatigue is read BEFORE this action's increment, for both the reward
        # penalty and the click probability (matches the proposal's worked example).
        f_now = self.fatigue

        # --- 1. Click / response model -------------------------------------
        # A click requires a send AND an inactive user (CTR convention: the
        # metric is winning back an absent user, not pinging a present one).
        clicked = False
        if sent and not self.active:
            w_a = cfg["w_arch"][self.archetype][action]
            p_click = np.clip(self._p0() * (1.0 - cfg["mu"] * f_now) * w_a, 0.0, 1.0)
            clicked = bool(self.rng.random() < p_click)

        blk = self.hour // BLOCK_HOURS
        if sent:
            self.blk_sends[blk] += 1
            self.blk_clicks[blk] += clicked
            self.type_sends[action] += 1
            self.type_clicks[action] += clicked

        # --- 2. Retention-streak shaping -----------------------------------
        if clicked:
            self.streak += 1
            self.hours_since_click = 0
        else:
            self.hours_since_click += 1
            if self.hours_since_click > cfg["streak_lapse_hours"]:
                self.streak = 0        # lapsed: the bonus collapses to zero

        streak_bonus = 0.0
        if clicked and self.streak > 0:
            if cfg["streak_mode"] == "cumulative":
                b = cfg["beta"]
                streak_bonus = sum(b ** k for k in range(1, self.streak + 1))
            else:                                   # "marginal"
                streak_bonus = cfg["beta"] ** self.streak
            if cfg["streak_cap"] is not None:
                streak_bonus = min(streak_bonus, cfg["streak_cap"])

        # --- 3. Fatigue dynamics -------------------------------------------
        self.fatigue = float(np.clip(cfg["lam"] * f_now + cfg["kappa"][action], 0.0, 1.0))

        # --- 4. Churn model (terminal) -------------------------------------
        if self.fatigue > cfg["churn_threshold"]:
            hazard = 1.0 / (1.0 + np.exp(-(cfg["gamma_0"] + cfg["gamma_1"] * self.fatigue)))
            self.opted_out = bool(self.rng.random() < hazard)

        # --- 5. Reward ------------------------------------------------------
        reward = (cfg["R_click"] * clicked
                  - cfg["W_send"][action]
                  - cfg["W_fat"] * f_now
                  - cfg["R_churn"] * self.opted_out
                  + streak_bonus)

        # --- 6. Advance time ------------------------------------------------
        self.t += 1
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day + 1) % 7

        self.hours_since_send = 0 if sent else self.hours_since_send + 1
        self.recency = float(min(self.hours_since_send / 24.0, 1.0))
        self._roll_activity()

        terminated = self.opted_out
        truncated = self.t >= cfg["steps_per_episode"]

        info = {
            "archetype": self.archetype,
            "clicked": clicked, "sent": sent,
            "fatigue": self.fatigue, "fatigue_pre": f_now,
            "opted_out": self.opted_out, "streak": self.streak,
            "streak_bonus": streak_bonus,
            "hour": int((self.hour - 1) % 24), "active_pre": self.active,
        }
        return self._observe(), float(reward), terminated, truncated, info


# ==========================================================================
# notebook cell 17
# ==========================================================================

# --- Non-learning baselines -------------------------------------------------
# These answer RQ1: does learning beat not learning? Both implement the same
# Agent interface so the harness treats them identically to a trained agent.

class FixedScheduleAgent(Agent):
    """Sends one Engagement Nudge per day at a fixed hour (default 18:00).

    Represents the conventional context-blind notification system: the same
    push, to every user, at the same time, regardless of state.
    """

    def __init__(self, hour=18, action=ENGAGE):
        self.hour, self.action = hour, action

    @property
    def name(self):
        return f"Fixed-{self.hour:02d}:00"

    def act(self, state, greedy=False):
        return (self.action if int(state[IDX_HOUR]) == self.hour else HOLD), {}

    def update(self, *args, **kwargs):
        return {}


class RandomAgent(Agent):
    """Uniformly random action. The floor: what happens with no policy at all."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return "Random"

    def act(self, state, greedy=False):
        return int(self.rng.integers(N_ACTIONS)), {}

    def update(self, *args, **kwargs):
        return {}


# --- Harness ----------------------------------------------------------------

def run_episodes(agent, env, n_episodes=None, seeds=None, learn=True,
                 greedy=False, collect_log=False):
    """Run `agent` in `env` and return aggregate metrics.

    Args:
        n_episodes: number of episodes (ignored if `seeds` is given).
        seeds: explicit per-episode reset seeds. This is the evaluation
            protocol: passing the same seed list to every agent guarantees they
            all face the identical sequence of archetypes and starting
            conditions, so differences in score reflect the policy rather than
            luck of the draw. (Within an episode the RNG streams necessarily
            diverge, because the agents take different actions.)
        learn: call agent.update(). False for evaluation and for baselines.
        greedy: act without exploration. Used for all reported results.
        collect_log: also return a per-step record, for the decision-distribution
            plots and the animation.

    Returns:
        (metrics, log, per-episode rewards, hour x action decision counts)
    """
    if seeds is None:
        seeds = [None] * int(n_episodes)

    ep_rewards, ep_sends, ep_clicks, ep_optouts, ep_lengths = [], [], [], [], []
    hour_actions = np.zeros((24, N_ACTIONS), dtype=int)
    log = []

    for ep, sd in enumerate(seeds):
        state, info = env.reset(seed=sd)
        agent.reset()
        total_r = sends = clicks = 0.0

        while True:
            action, aux = agent.act(state, greedy=greedy)
            next_state, reward, terminated, truncated, info = env.step(action)

            if learn:
                agent.update(state, action, reward, next_state, terminated, aux)

            total_r += reward
            sends += info["sent"]
            clicks += info["clicked"]
            hour_actions[info["hour"], action] += 1

            if collect_log:
                log.append({"episode": ep, "step": env.t, "hour": info["hour"],
                            "archetype": info["archetype"], "action": action,
                            "clicked": info["clicked"], "sent": info["sent"],
                            "fatigue": info["fatigue"], "reward": reward,
                            "streak": info["streak"], "opted_out": info["opted_out"]})

            state = next_state
            if terminated or truncated:
                ep_optouts.append(float(terminated))
                break

        ep_rewards.append(total_r)
        ep_sends.append(sends)
        ep_clicks.append(clicks)
        ep_lengths.append(env.t)

    total_sends = float(np.sum(ep_sends))
    metrics = {
        "agent": agent.name,
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        # CTR = clicks / sends (clicks counted only when the user was inactive).
        "ctr": float(np.sum(ep_clicks) / total_sends) if total_sends > 0 else 0.0,
        "sends_per_episode": float(np.mean(ep_sends)),
        "clicks_per_episode": float(np.mean(ep_clicks)),
        "optout_rate": float(np.mean(ep_optouts)),
        "episode_length": float(np.mean(ep_lengths)),
    }
    return metrics, (log if collect_log else None), np.array(ep_rewards), hour_actions


# --- Evaluation protocol ----------------------------------------------------
# Reserved seeds, disjoint from anything used in training. This is the RL
# analogue of a held-out test set: the same episodes score every method.
# QUICK trades statistical strength for speed while iterating; every reported
# number should come from a QUICK = False run.

QUICK = False

EVAL_SEEDS = list(range(900_000, 900_050 if QUICK else 900_200))
TRAIN_EPISODES = 150 if QUICK else 600
N_SEEDS = 2 if QUICK else 5


# ==========================================================================
# notebook cell 18
# ==========================================================================

# --- Training / evaluation harness ---------------------------------------

# Probe episodes for mid-training snapshots. Disjoint from the training seeds
# (1000+) and from EVAL_SEEDS, so watching a policy learn never contaminates a
# reported number.
PROBE_SEEDS = list(range(950_000, 950_020))
SNAPSHOT_ARCHETYPES = ("OfficeWorker", "NightOwlStudent")


def policy_snapshot(agent, archetypes=SNAPSHOT_ARCHETYPES, seeds=PROBE_SEEDS):
    """Freeze the current policy and record what it does, per archetype.

    Greedy with learning disabled, so the agent is measured rather than
    advanced. `run_episodes` calls `agent.reset()` on each probe episode, so an
    agent must not hold un-consumed learning state across `reset()`.
    """
    out = {}
    for arch in archetypes:
        env = CANEEnv(seed=8000, archetype=arch)
        m, _, _, hours = run_episodes(agent, env, seeds=seeds,
                                      learn=False, greedy=True)
        out[arch] = {"hour_actions": hours, "reward": m["reward_mean"],
                     "ctr": m["ctr"], "sends": m["sends_per_episode"],
                     "optout": m["optout_rate"]}
    return out


def train_and_evaluate(make_agent, n_seeds=None, train_episodes=TRAIN_EPISODES,
                       archetype=None, collect_hours=False,
                       snapshot_every=None, snapshot_seed=0,
                       snapshot_archetypes=SNAPSHOT_ARCHETYPES):
    """Train and greedily evaluate one configuration across several seeds.

    Seed averaging is this study's analogue of cross-validation: it separates a
    genuine effect from a lucky initialisation. Training and evaluation use
    disjoint seed ranges so nothing is scored on episodes it was fitted to.

    snapshot_every: pause every N training episodes and record the policy. Only
        `snapshot_seed` is instrumented -- each snapshot costs
        len(PROBE_SEEDS) x len(snapshot_archetypes) episodes, so instrumenting
        every seed would cost more than the training itself.
    """
    n_seeds = N_SEEDS if n_seeds is None else n_seeds
    per_seed, hour_stack, snapshots = [], [], []

    for seed in range(n_seeds):
        agent = make_agent(seed)
        train_env = CANEEnv(seed=1000 + seed, archetype=archetype)

        if snapshot_every and seed == snapshot_seed:
            snapshots.append((0, policy_snapshot(agent, snapshot_archetypes)))
            done = 0
            while done < train_episodes:
                chunk = min(snapshot_every, train_episodes - done)
                run_episodes(agent, train_env, n_episodes=chunk,
                             learn=True, greedy=False)
                done += chunk
                snapshots.append((done, policy_snapshot(agent, snapshot_archetypes)))
        else:
            run_episodes(agent, train_env, n_episodes=train_episodes,
                         learn=True, greedy=False)

        # Evaluate (greedy, frozen) on the reserved held-out episodes.
        eval_env = CANEEnv(seed=7000 + seed, archetype=archetype)
        m, _, rewards, hours = run_episodes(agent, eval_env, seeds=EVAL_SEEDS,
                                            learn=False, greedy=True)
        per_seed.append(m)
        hour_stack.append(hours)

    rewards = np.array([m["reward_mean"] for m in per_seed])
    summary = {
        "agent": per_seed[0]["agent"],
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(ddof=0),
        "reward_sem_std": rewards.std(ddof=1) if n_seeds > 1 else 0.0,
        "ctr": np.mean([m["ctr"] for m in per_seed]),
        "sends_per_episode": np.mean([m["sends_per_episode"] for m in per_seed]),
        "optout_rate": np.mean([m["optout_rate"] for m in per_seed]),
        "per_seed_rewards": rewards,
    }
    if collect_hours:
        summary["hour_actions"] = np.sum(hour_stack, axis=0)
    if snapshots:
        summary["snapshots"] = snapshots
    return summary


def evaluate_baseline(make_agent, archetype=None):
    """Baselines do not learn, so they are only evaluated."""
    agent = make_agent(0)
    env = CANEEnv(seed=7000, archetype=archetype)
    m, _, rewards, hours = run_episodes(agent, env, seeds=EVAL_SEEDS,
                                        learn=False, greedy=True)
    return {"agent": m["agent"], "reward_mean": m["reward_mean"],
            "reward_std": m["reward_std"], "reward_sem_std": 0.0,
            "ctr": m["ctr"], "sends_per_episode": m["sends_per_episode"],
            "optout_rate": m["optout_rate"],
            "per_seed_rewards": np.array([m["reward_mean"]]),
            "hour_actions": hours}

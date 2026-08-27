"""CANE deep agents: DQN, Double DQN and PPO.

GENERATED FILE -- do not edit by hand.

Produced by `tools/extract_package.py`, copied verbatim from the notebooks that
own each algorithm:

  * DQN/AssignmentCode(DQN).ipynb
      cells [19, 20, 21, 22] -- DQN_CONFIG, ReplayBuffer, QNetwork, DQNAgent
  * Code/doubleDQN.ipynb
      cells [18, 21] -- DDQN_CONFIG, DDQNAgent
  * Code/AssignementCode(PPO).ipynb
      cells [16, 17, 18] -- PPO_CONFIG, ActorCritic, PPOAgent

Only the agent code is taken. Each of those notebooks also carries its own copy
of the environment and harness; those duplicates are exactly what `cane.core`
replaces, and `tools/parity_check.py` proves the copies agree.

This module is deliberately NOT imported by `cane/__init__.py`, so `import cane`
stays free of torch. Import it explicitly when you need the deep agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cane.core import (
    HOLD, ENGAGE, INCENTIVE, N_ACTIONS, ACTION_NAMES,
    IDX_HOUR, IDX_DAY, IDX_ACTIVE, IDX_FATIGUE, IDX_RECENCY,
    STATE_DIM, agent_state_dim, Agent,
)

# The project trains and evaluates on CPU throughout: the networks are two
# 64-unit hidden layers, far too small for a GPU transfer to pay for itself, and
# pinning this keeps runs reproducible across machines.
DEVICE = "cpu"
torch.set_num_threads(1)

# ==========================================================================
# DQN/AssignmentCode(DQN).ipynb  cell 19
# ==========================================================================

# --- Configuration ----------------------------------------------------------
# Kept in step with the DQN notebook, which had moved on without this copy. The
# symptom was subtle: studies driven through this package showed DQN sending
# exactly 7 notifications a week on most archetypes, which is precisely the
# minimum-contact quota of one per day and therefore means the agent chose to
# send nothing at all -- the wrapper was doing every send. At gamma=0.99 over a
# 168-step episode that is the correct policy, because a send's discounted
# fatigue tail outweighs a click that pays once.
#
# `tools/config_parity.py` now checks this automatically. `parity_check.py`
# could not: it compares the deterministic baseline rows, which do not depend
# on any learner's hyperparameters.
DQN_CONFIG = dict(
    lr=3e-4,                  # Adam learning rate            (was 1e-4)
    batch_size=64,            # Minibatch size
    max_grad_norm=0.5,        # Gradient clipping norm
    gamma=0.90,               # Discount factor               (was 0.99)
    epsilon_start=1.0,        # Initial exploration rate
    epsilon_end=0.08,         # Final exploration rate        (was 0.05)
    epsilon_decay_steps=60000,# Linear decay duration in steps(was 50000)
    buffer_size=20000,        # Replay buffer capacity        (was 10000)
    min_replay_size=1000,     # Warmup steps before learning
    target_update_freq=500,   # Target network hard update frequency
    hidden_sizes=(64, 64),    # Q-network hidden layer dimensions
)


# ==========================================================================
# DQN/AssignmentCode(DQN).ipynb  cell 20
# ==========================================================================

class ReplayBuffer:
    """Experience replay buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity, state_dim, seed=0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# ==========================================================================
# DQN/AssignmentCode(DQN).ipynb  cell 21
# ==========================================================================

class QNetwork(nn.Module):
    """Multi-Layer Perceptron Q-network."""

    def __init__(self, d_in, n_actions, hidden_sizes):
        super().__init__()
        self.net = self._mlp(d_in, hidden_sizes, n_actions, out_gain=0.01)

    @staticmethod
    def _mlp(d_in, hidden, d_out, out_gain):
        layers, prev = [], d_in
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            prev = h
        head = nn.Linear(prev, d_out)
        nn.init.orthogonal_(head.weight, gain=out_gain)
        nn.init.zeros_(head.bias)
        layers.append(head)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================================================
# DQN/AssignmentCode(DQN).ipynb  cell 22
# ==========================================================================

class DQNAgent(Agent):
    """Deep Q-Network Agent (Mnih et al., 2015)."""

    def __init__(self, seed=0, label=None, **overrides):
        unknown = set(overrides) - set(DQN_CONFIG)
        if unknown:
            raise ValueError(f"unknown DQN hyperparameter(s): {sorted(unknown)}")
        self.cfg = {**DQN_CONFIG, **overrides}
        self._label = label

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.seed = seed

        self.d_in = agent_state_dim()
        self.online_net = QNetwork(self.d_in, N_ACTIONS, tuple(self.cfg["hidden_sizes"])).to(DEVICE)
        self.target_net = QNetwork(self.d_in, N_ACTIONS, tuple(self.cfg["hidden_sizes"])).to(DEVICE)
        self._sync_target()
        self.target_net.eval()

        self.opt = torch.optim.Adam(self.online_net.parameters(), lr=self.cfg["lr"])
        self.buffer = ReplayBuffer(self.cfg["buffer_size"], self.d_in, seed=seed)

        self.history = []
        self.n_updates = 0
        self.total_steps = 0

    def _features(self, state):
        x = np.asarray(state, dtype=np.float32)[:self.d_in].copy()
        x[IDX_HOUR] = x[IDX_HOUR] / 23.0
        x[IDX_DAY] = x[IDX_DAY] / 6.0
        return x

    def _epsilon(self):
        cfg = self.cfg
        frac = min(1.0, self.total_steps / max(1, cfg["epsilon_decay_steps"]))
        return cfg["epsilon_start"] + frac * (cfg["epsilon_end"] - cfg["epsilon_start"])

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @property
    def name(self):
        return self._label or "DQN"

    def act(self, state, greedy=False):
        x = torch.from_numpy(self._features(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(x).squeeze(0)

        if greedy:
            action = int(torch.argmax(q_values).item())
        else:
            eps = self._epsilon()
            if self.rng.random() < eps:
                action = int(self.rng.integers(N_ACTIONS))
            else:
                action = int(torch.argmax(q_values).item())

        return action, {"q_values": q_values.numpy().tolist()}

    def update(self, state, action, reward, next_state, done, aux):
        feat_s = self._features(state)
        feat_ns = self._features(next_state)

        self.buffer.push(feat_s, action, reward, feat_ns, done)
        self.total_steps += 1

        if len(self.buffer) < self.cfg["min_replay_size"]:
            return {}

        diag = self._optimise()

        if self.total_steps % self.cfg["target_update_freq"] == 0:
            self._sync_target()

        return diag

    def reset(self):
        pass

    def _optimise(self):
        cfg = self.cfg
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg["batch_size"])

        q_all = self.online_net(states)
        q_values = q_all.gather(1, actions.unsqueeze(1).long()).squeeze(1)

        # Standard DQN Bellman target (Mnih et al., 2015)
        with torch.no_grad():
            next_q_target = self.target_net(next_states)
            max_next_q = next_q_target.max(dim=1).values
            target = rewards + cfg["gamma"] * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), cfg["max_grad_norm"])
        self.opt.step()

        self.n_updates += 1
        diag = {
            "loss": float(loss.detach()),
            "mean_q": float(q_values.mean().detach()),
            "max_q": float(q_values.max().detach()),
            "epsilon": self._epsilon(),
            "update": self.n_updates,
            "step": self.total_steps,
        }
        self.history.append(diag)
        return diag

    def state_dict_for_save(self):
        return {"online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "cfg": dict(self.cfg),
                "d_in": self.d_in, "seed": self.seed,
                "n_updates": self.n_updates, "total_steps": self.total_steps}


# ==========================================================================
# Code/doubleDQN.ipynb  cell 18
# ==========================================================================

# --- Configuration ----------------------------------------------------------
# Kept in step with section 1 of the Double DQN notebook. The two had diverged:
# the notebook was retuned after its first full run collapsed to never-send and
# this copy was not, so every study driven through this package went on
# reproducing the collapse while the notebook's own results showed the fix.
#
# gamma is the lever. Over a 168-step episode at gamma=0.99 the discounted tail
# of a send's fatigue cost is worth on the order of a hundred steps of future
# debt against a click that pays once, and silence is then the correct answer.
# At 0.90 that tail is worth roughly ten steps and contact becomes affordable.
# The other four values come with it, and match the DQN notebook, so the two
# value-based agents stay tuned alike.
DDQN_CONFIG = dict(
    lr=3e-4,                  # Adam learning rate            (was 1e-4)
    batch_size=64,            # Minibatch size
    max_grad_norm=0.5,        # Gradient clipping norm
    gamma=0.90,               # Discount factor               (was 0.99)
    epsilon_start=1.0,        # Initial exploration rate
    epsilon_end=0.08,         # Final exploration rate        (was 0.05)
    epsilon_decay_steps=60000,# Linear decay duration in steps(was 50000)
    buffer_size=20000,        # Replay buffer capacity        (was 10000)
    min_replay_size=1000,     # Warmup steps before learning
    target_update_freq=500,   # Target network hard update frequency
    hidden_sizes=(64, 64),    # Q-network hidden layer dimensions
)


# ==========================================================================
# Code/doubleDQN.ipynb  cell 21
# ==========================================================================

class DDQNAgent(Agent):
    """Double Deep Q-Network Agent."""

    def __init__(self, seed=0, label=None, **overrides):
        unknown = set(overrides) - set(DDQN_CONFIG)
        if unknown:
            raise ValueError(f"unknown DDQN hyperparameter(s): {sorted(unknown)}")
        self.cfg = {**DDQN_CONFIG, **overrides}
        self._label = label

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.seed = seed

        self.d_in = agent_state_dim()
        self.online_net = QNetwork(self.d_in, N_ACTIONS, tuple(self.cfg["hidden_sizes"])).to(DEVICE)
        self.target_net = QNetwork(self.d_in, N_ACTIONS, tuple(self.cfg["hidden_sizes"])).to(DEVICE)
        self._sync_target()
        self.target_net.eval()

        self.opt = torch.optim.Adam(self.online_net.parameters(), lr=self.cfg["lr"])
        self.buffer = ReplayBuffer(self.cfg["buffer_size"], self.d_in, seed=seed)

        self.history = []
        self.n_updates = 0
        self.total_steps = 0

    def _features(self, state):
        x = np.asarray(state, dtype=np.float32)[:self.d_in].copy()
        x[IDX_HOUR] = x[IDX_HOUR] / 23.0
        x[IDX_DAY] = x[IDX_DAY] / 6.0
        return x

    def _epsilon(self):
        cfg = self.cfg
        frac = min(1.0, self.total_steps / max(1, cfg["epsilon_decay_steps"]))
        return cfg["epsilon_start"] + frac * (cfg["epsilon_end"] - cfg["epsilon_start"])

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @property
    def name(self):
        return self._label or "DDQN"

    def act(self, state, greedy=False):
        x = torch.from_numpy(self._features(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(x).squeeze(0)

        if greedy:
            action = int(torch.argmax(q_values).item())
        else:
            eps = self._epsilon()
            if self.rng.random() < eps:
                action = int(self.rng.integers(N_ACTIONS))
            else:
                action = int(torch.argmax(q_values).item())

        return action, {"q_values": q_values.numpy().tolist()}

    def update(self, state, action, reward, next_state, done, aux):
        feat_s = self._features(state)
        feat_ns = self._features(next_state)

        self.buffer.push(feat_s, action, reward, feat_ns, done)
        self.total_steps += 1

        if len(self.buffer) < self.cfg["min_replay_size"]:
            return {}

        diag = self._optimise()

        if self.total_steps % self.cfg["target_update_freq"] == 0:
            self._sync_target()

        return diag

    def reset(self):
        pass

    def _optimise(self):
        cfg = self.cfg
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg["batch_size"])

        q_all = self.online_net(states)
        q_values = q_all.gather(1, actions.unsqueeze(1).long()).squeeze(1)

        # Double DQN target decoupling
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            target = rewards + cfg["gamma"] * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), cfg["max_grad_norm"])
        self.opt.step()

        self.n_updates += 1
        diag = {
            "loss": float(loss.detach()),
            "mean_q": float(q_values.mean().detach()),
            "max_q": float(q_values.max().detach()),
            "epsilon": self._epsilon(),
            "update": self.n_updates,
            "step": self.total_steps,
        }
        self.history.append(diag)
        return diag

    def state_dict_for_save(self):
        return {"online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "cfg": dict(self.cfg),
                "d_in": self.d_in, "seed": self.seed,
                "n_updates": self.n_updates, "total_steps": self.total_steps}


# ==========================================================================
# Code/AssignementCode(PPO).ipynb  cell 16
# ==========================================================================

# --- 6.1 Hyperparameters ----------------------------------------------------
# Every PPO tunable lives here, mirroring CANE_CONFIG, so the random search in
# section 11 can sweep them without reaching into the class body.

PPO_CONFIG = dict(
    # Optimisation
    lr=3e-4,                # Adam step size; the PPO default from Schulman et al.
    n_epochs=10,            # passes over each rollout before it is discarded
    batch_size=64,          # minibatch within an epoch
    n_steps=168,            # rollout length; one CANE episode = 168 hourly steps
    max_grad_norm=0.5,      # gradient clipping, guards against a single bad batch

    # Objective
    gamma=0.99,             # discount; a send's fatigue cost persists for days
    gae_lambda=0.95,        # GAE bias/variance trade-off
    clip_eps=0.2,           # trust region on the probability ratio
    vf_coef=0.5,            # weight of the value loss
    ent_coef=0.01,          # entropy bonus; keeps the policy from collapsing early

    # Network
    hidden_sizes=(64, 64),  # actor and critic MLPs; in = agent_state_dim(), out = N_ACTIONS
)


# ==========================================================================
# Code/AssignementCode(PPO).ipynb  cell 17
# ==========================================================================

# --- 6.2 Actor-critic network -----------------------------------------------

class ActorCritic(nn.Module):
    """Separate policy and value MLPs sharing no parameters.

    Separate trunks rather than a shared one: the value function has to track a
    reward scale spanning +10 (a click) to -30 (an opt-out), and letting those
    gradients into the policy trunk is a known source of instability at this
    size. Two 64-unit MLPs are cheap enough that sharing buys nothing.
    """

    def __init__(self, d_in, n_actions, hidden_sizes):
        super().__init__()
        self.actor = self._mlp(d_in, hidden_sizes, n_actions, out_gain=0.01)
        self.critic = self._mlp(d_in, hidden_sizes, 1, out_gain=1.0)

    @staticmethod
    def _mlp(d_in, hidden, d_out, out_gain):
        """Orthogonal init with sqrt(2) on hidden layers, `out_gain` on the head.

        The small head gain matters for the policy: it starts the logits near
        zero, so the initial policy is almost uniform. Starting confident would
        mean the first rollout explores one action and the clip then makes that
        commitment expensive to undo.
        """
        layers, prev = [], d_in
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            prev = h
        head = nn.Linear(prev, d_out)
        nn.init.orthogonal_(head.weight, gain=out_gain)
        nn.init.zeros_(head.bias)
        layers.append(head)
        return nn.Sequential(*layers)

    def forward(self, x):
        """Returns (logits, value). Value is squeezed to shape (batch,)."""
        return self.actor(x), self.critic(x).squeeze(-1)


# ==========================================================================
# Code/AssignementCode(PPO).ipynb  cell 18
# ==========================================================================

# --- 6.3 PPO agent ----------------------------------------------------------

class PPOAgent(Agent):
    """Proximal policy optimisation over the shared Agent interface.

    Implements exactly the four members the harness calls -- `name`, `act`,
    `update`, `reset` -- so it drops into the registry beside LinUCB and the
    baselines with no change to the harness.

    The rollout buffer is filled by `update()` and drained when it reaches
    `n_steps`. `reset()` deliberately does NOT drain it: the harness calls
    `reset()` on evaluation and snapshot episodes as well, where `update()` is
    never called, so optimising there would mean that measuring the policy also
    changed it.
    """

    def __init__(self, seed=0, label=None, **overrides):
        """
        Args:
            seed: seeds numpy (action sampling, minibatch shuffling) and torch
                (network initialisation). Everything stochastic in this agent
                derives from this one argument.
            label: overrides the display name in results tables, mirroring
                LinUCBAgent's `label`.
            **overrides: any key of PPO_CONFIG. Unknown keys raise rather than
                being silently ignored, so a typo in the random search fails
                loudly instead of quietly scoring the default configuration.
        """
        unknown = set(overrides) - set(PPO_CONFIG)
        if unknown:
            raise ValueError(f"unknown PPO hyperparameter(s): {sorted(unknown)}")
        self.cfg = {**PPO_CONFIG, **overrides}
        self._label = label

        # One seed drives everything: numpy for sampling and shuffling, torch
        # for the initial weights.
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.seed = seed

        # Derived, never hard-coded -- 24 with the belief state, 5 without.
        self.d_in = agent_state_dim()
        self.net = ActorCritic(self.d_in, N_ACTIONS,
                               tuple(self.cfg["hidden_sizes"])).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])

        self._clear_buffer()
        self.history = []        # one dict per optimisation step, for the P2 plot
        self.n_updates = 0
        self.total_steps = 0

    # -- helpers -------------------------------------------------------------

    def _clear_buffer(self):
        self._buf = {k: [] for k in ("states", "next_states", "actions", "logp",
                                     "values", "rewards", "terminal", "boundary")}

    def _features(self, state):
        """Rescale the raw observation onto a common range.

        Hour (0-23) and day (0-6) arrive as raw counts while every other entry
        is already in [0, 1]. Dividing by their maxima puts the whole vector on
        one scale without changing its dimensionality: this is a rescaling, not
        an encoding, so the input width stays agent_state_dim().
        """
        x = np.asarray(state, dtype=np.float32)[:self.d_in].copy()
        x[IDX_HOUR] = x[IDX_HOUR] / 23.0
        x[IDX_DAY] = x[IDX_DAY] / 6.0
        return x

    @property
    def name(self):
        return self._label or "PPO"

    # -- interface -----------------------------------------------------------

    def act(self, state, greedy=False):
        """Sample an action, returning the quantities the update will need.

        `aux` carries the log-probability and the value estimate AS OF THIS
        STATE AND THIS POLICY. Both must be captured now: PPO's ratio is
        pi_new/pi_old, and recomputing the denominator after the policy has
        moved would drive the ratio to 1 and silently disable the clipping.

        greedy=True takes the argmax of the logits, which is deterministic and
        is what every reported evaluation number uses.
        """
        x = torch.from_numpy(self._features(state)).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits.squeeze(0)
        logp_all = torch.log_softmax(logits, dim=-1)

        if greedy:
            action = int(torch.argmax(logits).item())
        else:
            # Sampled with numpy rather than torch so that action selection
            # depends only on self.rng, which makes a run reproducible from the
            # seed alone regardless of any other torch RNG consumer.
            probs = torch.exp(logp_all).double().numpy()
            probs = probs / probs.sum()          # guard against fp drift
            action = int(self.rng.choice(N_ACTIONS, p=probs))

        return action, {"log_prob": float(logp_all[action].item()),
                        "value": float(value.item())}

    def update(self, state, action, reward, next_state, done, aux):
        """Store one transition; optimise when the rollout is full.

        `done` is the harness's `terminated` flag -- a true opt-out. A week that
        merely ran out of hours arrives here with done=False and is closed by
        `reset()` instead, which is what keeps a truncation from being mistaken
        for an absorbing state.

        Returns the optimisation diagnostics on the step that triggers an
        update, and {} on every other step.
        """
        self._buf["states"].append(self._features(state))
        self._buf["next_states"].append(self._features(next_state))
        self._buf["actions"].append(int(action))
        self._buf["logp"].append(float(aux["log_prob"]))
        self._buf["values"].append(float(aux["value"]))
        self._buf["rewards"].append(float(reward))
        self._buf["terminal"].append(bool(done))
        self._buf["boundary"].append(bool(done))   # reset() may promote this
        self.total_steps += 1

        if len(self._buf["states"]) >= self.cfg["n_steps"]:
            return self._optimise()
        return {}

    def reset(self):
        """Close the current trajectory segment. Never takes a gradient step.

        The harness calls this at the start of every episode, including greedy
        evaluation and `policy_snapshot` probes where `update()` is never
        called. So this does exactly one thing: mark the last stored transition
        as an episode boundary, so GAE stops carrying credit across the join.
        Marking is idempotent, which is what makes it safe to call from an
        evaluation episode that stores nothing.
        """
        if self._buf["boundary"]:
            self._buf["boundary"][-1] = True

    # -- optimisation --------------------------------------------------------

    @staticmethod
    def compute_gae(rewards, values, next_values, terminal, boundary,
                    gamma, gae_lambda):
        """Generalised advantage estimation with truncation handled correctly.

        Two flags, doing two different jobs:

        `terminal` -- a real opt-out. The episode is over and there is no future
            reward, so V(s') is dropped from the TD residual.
        `boundary` -- the last transition of any episode, terminal or not. GAE
            must not carry credit past it into an unrelated episode, but a
            merely-truncated episode still has a future, so V(s') is bootstrapped
            in the residual even though lambda-credit stops.

        A truncated week therefore keeps its bootstrap and a churned one does
        not, which is exactly the distinction the harness cannot express.
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in range(T - 1, -1, -1):
            v_next = 0.0 if terminal[t] else next_values[t]
            delta = rewards[t] + gamma * v_next - values[t]
            carry = 0.0 if boundary[t] else 1.0
            running = delta + gamma * gae_lambda * carry * running
            adv[t] = running
        return adv

    def _optimise(self):
        """One PPO update over the filled rollout, then discard it."""
        cfg = self.cfg
        buf = self._buf

        states = torch.from_numpy(np.asarray(buf["states"], dtype=np.float32))
        next_states = torch.from_numpy(np.asarray(buf["next_states"], dtype=np.float32))
        actions = torch.from_numpy(np.asarray(buf["actions"], dtype=np.int64))
        old_logp = torch.from_numpy(np.asarray(buf["logp"], dtype=np.float32))

        # Values are recomputed here rather than reusing the ones cached in aux:
        # within a rollout the policy has not moved, so they agree, and a single
        # batched forward pass is far cheaper than 168 individual ones.
        with torch.no_grad():
            _, values = self.net(states)
            _, next_values = self.net(next_states)

        adv_np = self.compute_gae(
            np.asarray(buf["rewards"], dtype=np.float64),
            values.numpy().astype(np.float64),
            next_values.numpy().astype(np.float64),
            np.asarray(buf["terminal"], dtype=bool),
            np.asarray(buf["boundary"], dtype=bool),
            cfg["gamma"], cfg["gae_lambda"])
        returns = torch.from_numpy(
            (adv_np + values.numpy().astype(np.float64)).astype(np.float32))

        adv = torch.from_numpy(adv_np.astype(np.float32))
        # Normalised once per rollout: makes the gradient scale independent of
        # the reward magnitude, which here spans +10 (click) to -30 (opt-out).
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        T = states.shape[0]
        logs = {"policy_loss": [], "value_loss": [], "entropy": [],
                "approx_kl": [], "clip_frac": []}

        for _ in range(cfg["n_epochs"]):
            order = self.rng.permutation(T)
            for start in range(0, T, cfg["batch_size"]):
                mb = order[start:start + cfg["batch_size"]]
                if len(mb) < 2:
                    continue                     # a 1-sample minibatch has no signal
                idx = torch.from_numpy(mb)

                logits, value = self.net(states[idx])
                logp_all = torch.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                entropy = -(logp_all.exp() * logp_all).sum(dim=-1).mean()

                ratio = torch.exp(logp - old_logp[idx])
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1.0 - cfg["clip_eps"],
                                      1.0 + cfg["clip_eps"]) * adv[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(value, returns[idx])
                loss = (policy_loss
                        + cfg["vf_coef"] * value_loss
                        - cfg["ent_coef"] * entropy)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator; non-negative by
                    # construction, unlike the naive (old - new) mean.
                    log_ratio = logp - old_logp[idx]
                    approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()

                logs["policy_loss"].append(float(policy_loss.detach()))
                logs["value_loss"].append(float(value_loss.detach()))
                logs["entropy"].append(float(entropy.detach()))
                logs["approx_kl"].append(float(approx_kl))
                logs["clip_frac"].append(float(clip_frac))

        self._clear_buffer()
        self.n_updates += 1
        diag = {k: float(np.mean(v)) if v else 0.0 for k, v in logs.items()}
        diag["update"] = self.n_updates
        diag["step"] = self.total_steps
        self.history.append(diag)
        return diag

    # -- persistence ---------------------------------------------------------

    def state_dict_for_save(self):
        """Everything needed to reconstruct this agent's policy.

        The project has no model-persistence convention yet (LinUCB is never
        saved), so this records the config and input width alongside the
        weights rather than assuming a loader will know them.
        """
        return {"net": self.net.state_dict(), "cfg": dict(self.cfg),
                "d_in": self.d_in, "seed": self.seed,
                "n_updates": self.n_updates, "total_steps": self.total_steps}

    def __repr__(self):
        return (f"PPOAgent(d_in={self.d_in}, hidden={tuple(self.cfg['hidden_sizes'])}, "
                f"lr={self.cfg['lr']}, clip_eps={self.cfg['clip_eps']}, "
                f"ent_coef={self.cfg['ent_coef']})")

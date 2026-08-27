"""CANE -- Context-Aware Notification Engine.

A reinforcement-learning agent that decides, each hour, whether to stay silent
or send one of two notification types to a simulated app user, maximising
long-term engagement while paying for the fatigue each send creates.

This package exists so that all four algorithms (LinUCB, DQN, Double DQN, PPO)
can be evaluated through one harness on identical held-out episodes. The code is
extracted verbatim from the LinUCB notebook -- see `cane/core.py` and
`tools/extract_package.py`.
"""

from cane.core import (  # noqa: F401
    # action space
    HOLD, ENGAGE, INCENTIVE, N_ACTIONS, ACTION_NAMES,
    # observation layout
    IDX_HOUR, IDX_DAY, IDX_ACTIVE, IDX_FATIGUE, IDX_RECENCY,
    IDX_ACT_RATE, IDX_CLICK_RATE, IDX_TYPE_CR, IDX_SINCE_CLICK,
    N_BLOCKS, BLOCK_HOURS, STATE_DIM, STATE_DIM_BASE,
    BELIEF_PRIOR, USE_BELIEF, agent_state_dim,
    # configuration
    CANE_CONFIG, ARCHETYPE_W, ARCHETYPES, ARCHETYPE_TABLE,
    WEEKEND_FLATTEN, archetype_curve,
    # environment
    CANEEnv,
    # agents
    Agent, FixedScheduleAgent, RandomAgent, LinUCBAgent,
    FEATURE_ENCODERS, feature_dim,
    # evaluation protocol
    QUICK, EVAL_SEEDS, TRAIN_EPISODES, N_SEEDS,
    # harness
    run_episodes, evaluate_baseline, train_and_evaluate, policy_snapshot,
)

__version__ = "1.0.0"

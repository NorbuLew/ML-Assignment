# G1 — Final coding files

**BMDS2114 Machine Learning**

*Context-Aware Notification Engine (CANE): A Reinforcement Learning Approach to
Fatigue-Aware Push-Notification Pacing for User-Engagement Optimization*

---

## What the system does

CANE is a platform-side decision agent for an app's push-notification channel.
Every hour it observes the user's context and chooses one of three actions —
send an **Engagement Nudge**, send an **Incentive Nudge**, or **Hold** — to
sustain long-term engagement rather than maximise the response to any single
message. Every send buys a chance of a click and pays a fatigue cost, and a user
whose fatigue crosses the churn threshold opts out permanently.

Four algorithms are compared on one shared environment against two non-learning
baselines:

| | Algorithm | Family |
|---|---|---|
| 1 | LinUCB | contextual bandit |
| 2 | DQN | value-based |
| 3 | Double DQN | value-based, decoupled selection/evaluation |
| 4 | PPO | policy-gradient |
| — | Fixed-18:00, Random | non-learning baselines |

**There is no external dataset.** The environment is a simulated five-archetype
population defined in code; the reference for this is the environment
specification in the LinUCB notebook, not a downloaded file.

---

## The four main coding files

Each is supplied as both `.ipynb` and `.pdf`. Every notebook is self-contained
and carries its own copy of the environment, harness and evaluation protocol, so
each can be run and explained independently.

| File | Contents |
|---|---|
| `G1 final coding 1 - LinUCB and shared environment` | The environment, the evaluation harness, the figure conventions, and LinUCB. **Read this one first** — the other three build on the definitions it establishes. |
| `G1 final coding 2 - DQN` | Deep Q-Network |
| `G1 final coding 3 - DDQN` | Double DQN, the minimum-contact wrapper of section 8.1, and the random hyperparameter search |
| `G1 final coding 4 - PPO` | Proximal Policy Optimisation |

### Shared evaluation protocol

Identical across all four notebooks. This is what makes the comparison
like-for-like, and it is verified rather than asserted — see experimental coding
file 4.

| Setting | Value |
|---|---|
| Held-out evaluation episodes | seeds 900,000–900,199 (200 episodes) |
| Training budget | 600 episodes (~100,800 steps) |
| Seeds per learner | 5 |
| Episode length | 168 steps (one week, hourly) |
| State dimension | 24 (belief-state representation) |
| Archetypes | OfficeWorker, NightOwlStudent, NightShiftWorker, NormalStudent, Housewife |
| Break-even click rate | 0.340 |

---

## Deployment — the interactive dashboard

`dashboard and shared package/` contains the Streamlit application and the code
it imports.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Six pages: **Overview** (leaderboard and KPIs), **Head to Head** (paired
per-cell comparison with significance — the RQ2 answer), **Live Simulation**
(all five users animated through one week), **Personalisation** (chosen hour
against each person's true best hour, plus the RQ3 result), **Diagnosis** (why
some users received nothing, and the seven interventions tested), and
**Ensemble**.

The app runs fully offline and renders whatever result files exist, naming
anything missing rather than failing.

### Supporting code

| Path | Contents |
|---|---|
| `cane/` | the environment, harness, all four agents, ensembling and the study drivers, extracted once so the dashboard and studies share a single definition rather than a fifth copy |
| `dashboard/` | the Streamlit application |
| `tools/` | study drivers, the headless notebook runner, and the verification scripts |
| `run_all.py` | runs the whole pipeline in dependency order, one log per stage |

---

## Reproducing everything

```bash
python run_all.py            # the whole pipeline, in dependency order
python run_all.py --list     # what it will do, without doing it
python run_all.py --quick    # smoke pass; numbers NOT reportable
```

Verification, which can be run on its own:

```bash
.venv/Scripts/python.exe tools/check_results.py   # all four CSVs describe one experiment
.venv/Scripts/python.exe tools/parity_check.py    # cane/ reproduces the notebooks
.venv/Scripts/python.exe tools/render_check.py dashboard/pages/1_Head_to_Head.py
```

`check_results.py` compares the deterministic baseline rows across all four
result files. If they agree, the learners were scored on identical held-out
episodes. `parity_check.py` re-evaluates those baselines through `cane/` and
compares them to the notebooks' own rows — currently 10/10 agreeing to within
8.3e-17, which is floating-point noise rather than a behavioural difference.

---

## Notes for the reader

- **`DQN/AssignmentCode(DQN).ipynb` cell 2 is a raw cell** holding the original
  CUDA install command as a provisioning record. It is deliberately raw: as a
  code cell it would trigger a multi-gigabyte download. All reported results are
  CPU-only (`DEVICE = "cpu"`, `torch.set_num_threads(1)`).
- **A reward of `0.00` is not a missing value.** It is exactly what an agent
  earns by never sending — no clicks collected, no costs paid. Experimental
  coding file 1 is the investigation of why that happens and what fixes it.
- **The Double DQN notebook is deliberately not on the shipped hyperparameters.**
  It runs `gamma=0.90` rather than `0.99`, plus the minimum-contact wrapper of
  its section 8.1. Both changes exist for one reason: at `gamma=0.99` over a
  168-step episode, a send's discounted fatigue tail is worth about a hundred
  steps of future debt while the click pays once, so silence is genuinely the
  optimal policy and no amount of further training escapes it. With the two
  changes, Double DQN goes from contacting **2 of the 5** archetypes to **all
  5**. LinUCB, DQN and PPO are unchanged, so where a comparison turns on
  hyperparameters rather than on algorithm, this is the difference to name.
  The environment, reward function, evaluation protocol and seeds are identical
  across all four — the constraint acts on the policy's feasible set, never on
  its objective, which is the standard budgeted-MDP formulation.
- **PPO is silent on 3 of the 5 archetypes in the submitted run.** That is the
  same never-send optimum, reached by a different route, and it is reported
  rather than patched — the Diagnosis page reads which agents collapsed from the
  result files rather than from hardcoded text.
- **The four notebooks each carry their own copy of the environment.** That is
  deliberate: each team member has to be able to run and explain their own
  notebook without the others present.

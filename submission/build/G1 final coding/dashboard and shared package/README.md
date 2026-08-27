# CANE — Context-Aware Notification Engine

BMDS2114 Machine Learning assignment. A reinforcement-learning agent that decides,
each hour, whether to stay silent or send one of two notification types to a simulated
app user — maximising long-term engagement while paying for the fatigue each send
creates.

Four algorithms are compared on one shared environment: **LinUCB** (contextual
bandit), **DQN**, **Double DQN**, and **PPO**, against two non-learning baselines
(a fixed 18:00 daily schedule and a random sender).

## Setup

One shared virtual environment at the repo root. Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas matplotlib scipy streamlit altair ipykernel
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Torch comes from the **CPU index deliberately**. The code pins `DEVICE = "cpu"` and
`torch.set_num_threads(1)`; a CUDA wheel is a ~2.5 GB download for no benefit.

## Running everything

```bash
python run_all.py            # the whole pipeline
python run_all.py --list     # what it will do, in order
python run_all.py --quick    # smoke pass in minutes; numbers NOT reportable
```

Seven stages, in dependency order: train the two missing algorithms, verify all
four CSVs describe one experiment, verify `cane/` reproduces the notebooks, run
the hyperparameter search, run the exploration and ensemble studies, then render
every dashboard page. Each writes `logs/<stage>.log`; a failing stage does not
stop the later ones, and the exit code is non-zero if any failed.

Individual stages when iterating:

```bash
python run_all.py --only parity dashboard
python run_all.py --skip notebooks      # reuse CSVs that already exist
```

The two notebook runs go in parallel, which roughly halves the wall clock; every
later stage is sequential because each depends on the one before. The Double DQN
hyperparameter search is cell 41 of its notebook and takes about as long as the
training does, so the `notebooks` stage skips it and the `search` stage runs it
separately -- otherwise every downstream stage waits on a CSV that was ready at
the start. Cell indices are whole-notebook indices, markdown cells included, so
they match what Jupyter shows.

## Layout

| Path | Contents |
|---|---|
| `Code/AssignementCode(LinUCB).ipynb` | LinUCB + the shared environment, harness and figure conventions |
| `Code/AssignementCode(PPO).ipynb` | PPO |
| `Code/doubleDQN.ipynb` | Double DQN |
| `DQN/AssignmentCode(DQN).ipynb` | DQN |
| `Code/results/`, `DQN/results/` | per-algorithm result CSVs |
| `Code/models/`, `DQN/models/` | trained checkpoints (`.pt`, and `.npz` for LinUCB) |
| `Code/figures/`, `DQN/figures/` | generated PNGs |
| `cane/` | the shared package: environment, harness, agents, ensemble (below) |
| `dashboard/` | Streamlit app - the interactive prototype (below) |
| `artifacts/` | outputs of the ensemble and exploration studies |
| `tools/` | helper scripts (below) |

## Tools

```bash
# Execute a notebook headlessly, no Jupyter install needed.
.venv/Scripts/python.exe tools/run_notebook.py "Code/doubleDQN.ipynb" --quick
.venv/Scripts/python.exe tools/run_notebook.py "Code/doubleDQN.ipynb"

# Verify all four result CSVs describe the same experiment.
.venv/Scripts/python.exe tools/check_results.py
```

`check_results.py` compares the `Fixed-18:00` and `Random` rows across all four
CSVs. Those baselines are deterministic given the shared evaluation protocol, so
if they agree, the learners were evaluated on identical held-out episodes and the
four-way comparison is like-for-like. It is the fastest way to catch a notebook
that drifted from the shared config.

## The `cane/` package

The four notebooks each carry their own inline copy of the environment, which is
how the assignment was developed and is deliberately left alone -- each teammate
has to be able to run and explain their own notebook. `cane/` is that same code
extracted once, so the dashboard and the study scripts have a single definition
to import rather than a fifth copy.

| Module | Contents |
|---|---|
| `cane/core.py` | `CANEEnv`, the evaluation harness, the `Agent` protocol, LinUCB |
| `cane/agents_deep.py` | DQN, Double DQN, PPO (imports torch; not re-exported by `cane/__init__.py`, so `import cane` stays torch-free) |
| `cane/persistence.py` | loads the saved `.pt` / `.npz` checkpoints back into agents -- the notebooks save but never reload |
| `cane/ensemble.py` | majority vote, confidence-gated fusion, the hold-bias sweep |
| `cane/ensemble_study.py` | fits and evaluates the ensembles, per archetype |
| `cane/exploration_study.py` | the five-variant diagnosis of the never-send collapse |

The extraction is verbatim, and that is checked rather than asserted:

```bash
.venv/Scripts/python.exe tools/parity_check.py
```

It re-evaluates the deterministic baselines through `cane/` and compares them to
the notebooks' own CSV rows. Current status: **10/10 rows agree, max absolute
difference 8.3e-17** -- floating-point noise, not a behavioural difference. That
agreement is what makes the four-way comparison defensible.

## Dashboard

```bash
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Runs entirely offline. Pages render whatever result files exist and say plainly
which are still missing, so it is usable before every run has finished.

| Page | What it shows |
|---|---|
| Overview | KPI cards, headline leaderboard, per-file readiness |
| Head to Head | paired per-cell comparison of the algorithms, with significance -- this is the RQ2 answer |
| Live Simulation | all five archetypes animated side by side through one week -- the presentation page |
| Personalisation | the hour each agent chose for each person against that person's true best hour, and the RQ3 single-policy result |
| Diagnosis | which users received nothing and from which agents (read from the result files, not hardcoded), the seven interventions tested, and the one that recovered both coverage and profit |
| Ensemble | members against both ensemble schemes, the hold-bias search, and the caveats that come with them |

### The configuration switch

Overview and Head to Head carry a sidebar **Configuration** radio, and both read
the same `session_state` key, so the choice follows you between them.

| Setting | Source | Seeds | Episodes |
|---|---|---|---|
| **Final run (5 seeds)** — the default | the four `*_results.csv` the notebooks wrote | 5 | 600 |
| Tuning study (1 seed) | final-episode slice of `artifacts/learning_curves_tuned.csv` | **1** | 1500 |

The final run is what the report quotes and what the dashboard opens on. The
tuning study is kept reachable because it answers a different question: it is
the evidence that the never-send collapse was a local optimum rather than a
ceiling.

Note the four notebooks are no longer uniform. Double DQN runs a retuned
`gamma=0.90` plus the minimum-contact wrapper of section 8.1, which is what
takes it from contacting 2 of the 5 archetypes to all 5; LinUCB, DQN and PPO
run their original hyperparameters. The setting is therefore labelled by *run*
rather than by "shipped defaults", which stopped being true of DDQN.

The tuned run was trained on a single seed, so its leaderboard ships **without
confidence intervals** -- `charts.leaderboard_bars(..., show_ci=False)` drops the
whiskers rather than drawing a zero-width one, which would read as a precise
measurement. It also has no `optout_rate`: the curve files record reward, CTR
and send rate only, and that column is left blank rather than invented. To put
the tuned run on equal footing, run:

```bash
.venv/Scripts/python.exe tools/learning_curves.py --tuned --out-suffix _tuned \
    --episodes 1500 --every 50 --seeds 5
```

The Personalisation page has its own **Run** selector for the same reason: the
tuned 1500-episode run (`tune_study.csv`, variant `both_long`) is the reported
result, and the earlier 600-episode minimum-contact run is kept because it is
what the quota discussion on that page describes.

Pages are verified headlessly, without a browser:

```bash
.venv/Scripts/python.exe tools/render_check.py dashboard/pages/1_Head_to_Head.py
```

`render_check.py` executes a page against a stub `streamlit` and calls
`.to_dict()` on every chart it builds. Altair only validates a spec at
serialisation time, so a chart that constructs fine can still fail in the
browser; this catches that on the command line.

## Studies

```bash
# Why three of the four learners converge to sending nothing.
.venv/Scripts/python.exe -m cane.exploration_study

# Ensemble: fit on a validation split, report on the held-out test split.
.venv/Scripts/python.exe -m cane.ensemble_study
```

Both run in parallel across cores by default -- the exploration study's 45 cells
and the ensemble study's 5 archetypes are fully independent, and each cell is
single-threaded because `cane` pins torch to one thread. Measured on 16 logical
cores: the exploration study drops from roughly 90 minutes to about 10, the
ensemble study from about 50 to under 10. Pass `--jobs 1` to force the serial
path when debugging, or `--jobs N` to cap it.

Both accept `--quick` for a fast smoke pass. **Do not report `--quick` numbers.**
The quick ensemble pass in particular fits its weights on 20 validation episodes,
which is small enough that a member that abstains outscores one that takes a
reasonable risk -- the page says so, but the fix is to run the full pass.

## Evaluation protocol

Shared by all four notebooks; do not change one without changing all four.

| Setting | Value |
|---|---|
| Held-out eval episodes | seeds `900_000 … 900_199` (200) |
| Training budget | 600 episodes (~100,800 steps) |
| Seeds per learner | 5 |
| Episode length | 168 steps (one week, hourly) |
| Archetypes | OfficeWorker, NightOwlStudent, NightShiftWorker, NormalStudent, Housewife |

## Working rules

- **Do not move or rename the notebooks.** `Code/AssignementCode(PPO).ipynb` cell 24
  opens `AssignementCode(LinUCB).ipynb` by relative path and diffs its source text to
  prove both share one environment. Renaming either breaks that check.
- **Do not delete the LinUCB notebook's env/config/harness cells** — the same check
  reads them.
- `Notebooks Save/AssignementCode(LinUCB).ipynb` is a stale pre-belief-state
  duplicate. Leave it where it is; moving it into `Code/` may make the parity check
  resolve to the wrong file.
- `Code/run3.log` predates the 24-dimensional belief state (it prints `State dim: 5`).
  **Do not quote numbers from it.**
- `DQN/AssignmentCode(DQN).ipynb` cell 2 is a **raw** cell holding the original CUDA
  install command as a provisioning record. Keep it raw — as a code cell it would
  trigger a multi-gigabyte download mid-demo.

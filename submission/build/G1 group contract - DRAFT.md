# G1 — Group contract

**BMDS2114 Machine Learning**

*Context-Aware Notification Engine (CANE): A Reinforcement Learning Approach to
Fatigue-Aware Push-Notification Pacing for User-Engagement Optimization*

> **DRAFT — fill in the names and IDs before submitting.** The task allocation
> below is reconstructed from the repository's commit history
> (`github.com/NorbuLew/ML-Assignment`), so it reflects what was actually
> committed rather than what was planned. Correct anything that misrepresents a
> member's contribution — particularly work done offline or on the report, which
> leaves no trace in git.

---

## Members

| # | Name | Student ID | Git identity | Algorithm owned |
|---|---|---|---|---|
| 1 | Lew Xin Yi | `________` | `NorbuLew` / `Lew Xin Yi` | LinUCB + platform |
| 2 | `________` | `________` | `Kaider12` | DQN |
| 3 | `________` | `________` | `DrillerCat` | Double DQN |
| 4 | `________` | `________` | `Chiew Reyes` | PPO |

---

## Working agreement

**One algorithm per person, end to end.** Each member owns their algorithm
through the whole chain — implementation, Part B tuning, the corresponding
report section, and the demo Q&A for it. This was agreed at the start so that
every member can explain, justify and modify their own work during the
presentation, as the assignment requires.

**The environment is shared and frozen.** All four algorithms are evaluated on
one `CANEEnv` with one evaluation protocol. After the interface freeze, any
change to the environment invalidates all four members' results simultaneously,
so environment changes required agreement from the whole group.

**The comparison is verified, not assumed.** Because the four algorithms were
developed in separate notebooks, `tools/check_results.py` checks that the
deterministic baseline rows agree across all four result files before any
four-way comparison is reported.

---

## Task allocation

### Member 1 — Lew Xin Yi (platform + LinUCB)

- Designed and implemented the shared environment (`CANEEnv`), the five user
  archetypes, the reward model and the fatigue/churn dynamics.
- Designed the evaluation harness and the shared evaluation protocol (200
  held-out episodes, 5 seeds, 600-episode training budget).
- Diagnosed the original state representation as unable to support
  personalisation, and designed the 24-dimensional belief state that replaced it.
- Implemented and tuned **LinUCB**, and wrote its report section.
- Built the `cane/` package and the parity check proving it reproduces the
  notebooks.
- Built the **Streamlit dashboard** (six pages) — the deployment deliverable.
- Ran the experimental studies: the never-send collapse investigation, the
  personalisation and RQ3 tests, the ensembling study, and the convergence
  analysis.
- Integration: merging the four members' results and verifying the comparison
  is like-for-like.

### Member 2 — `________` (DQN)

- Implemented the Deep Q-Network agent in `DQN/AssignmentCode(DQN).ipynb`.
- Ran the DQN training and evaluation across five seeds and all five archetypes,
  and produced the Q-series figures.
- Ran the DQN random hyperparameter search.
- Writes the DQN section of the report and answers for DQN in the demo.

### Member 3 — `________` (Double DQN)

- Implemented the Double DQN agent in `Code/doubleDQN.ipynb`, including the
  decoupled action-selection / action-evaluation target.
- Writes the Double DQN section of the report and answers for DDQN in the demo.

### Member 4 — `________` (PPO)

- Implemented the PPO agent in `Code/AssignementCode(PPO).ipynb`, including the
  clipped surrogate objective and the ablation analysis.
- Writes the PPO section of the report and answers for PPO in the demo.

---

## Shared deliverables

| Deliverable | Owner | Status |
|---|---|---|
| Final coding files (4 notebooks + dashboard) | all, integrated by member 1 | |
| Experimental coding files (4 notebooks) | member 1 | |
| Report — introduction, problem statement, methodology | | |
| Report — per-algorithm sections | each member for their own | |
| Report — results, discussion, conclusion | | |
| Slides | | |
| Video | | |
| AI usage disclosure form | all members declare individually | |
| Assessment rubric (names + IDs) | | |

---

## Presentation

All members present, in formal attire, on the tasks listed above. Each member
takes the questions on their own algorithm.

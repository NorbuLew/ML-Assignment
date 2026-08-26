# Dashboard demo script

Running order for the recorded walkthrough. Roughly 6 minutes at a normal
speaking pace. Every number quoted below is read off the screen, not from these
notes -- if a number here disagrees with the dashboard, the dashboard is right.

```bash
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Before recording: press **Play week** once on the Live Simulation page so the
traces are warm. The first play of a given policy and seed simulates five
episodes; every play after that is instant.

---

## 1. Overview  (~40s)

Open on the KPI cards and the leaderboard.

Say what CANE is: an agent that decides, each hour, whether to stay silent or
send one of two notification types, maximising engagement while paying for the
fatigue every send creates. Four algorithms, one shared environment, 200
held-out episodes.

Point at the readiness dots in the sidebar -- they show which result files
exist. This is a live app reading real artifacts, not a slide.

## 2. Live Simulation  (~90s)  -- the centrepiece

Select the policy, then press **Play week**.

Talk over the animation:

- Five simulated users, five different daily rhythms. Same policy driving all
  of them.
- The meter under each name is fatigue. It climbs on every send and decays
  about 10% an hour. Green, then amber past 0.33, then red past 0.66.
- Watch the lanes diverge. The same agent behaves completely differently
  depending on who it is talking to.

If a lane stays silent all week, stop on it. **This is the headline finding**,
and it is the thing a bar chart gets wrong: a `0.00` in a results table looks
like a broken run, and watching it happen shows it is a decision -- the agent
is choosing to hold.

Scrub the slider back to an interesting hour to freeze the frame while you talk.

## 3. Personalisation  (~70s)

This is the assignment's actual claim, so give it the time.

The chart plots, for each person, the hour the agent chose against the hour
exhaustive search proved was best for them. Grey tick is their true best hour;
the coloured marks are what the agent picked. Distance along the axis is the
error, in hours.

Read the numbers off the screen. Under the tuned configuration three of five
users are contacted within an hour of their own best time and the agent picks
four distinct hours across five people. Random timing averages a six-hour error.

Say plainly what is not being claimed: these are per-archetype agents, so this
is personalisation by training, not one policy inferring who it is talking to.
That stricter question is RQ3 and the answer there is no.

Also read the honesty panel out loud -- the cells where the agent sent exactly
the quota are cells where the deadline chose, not the policy. Naming that is
worth more than quoting the better-looking average.

## 4. Diagnosis  (~80s)

This page answers *why* three users received nothing, and it is the strongest
Part B material in the project.

Three beats, in order:

1. **Silence was not optimal.** The best fixed schedule beats never-send on all
   five users, so `0.00` is a local optimum. Quote the forfeited reward.
2. **The cause.** Both charts, left then right. Every Q-value collapses --
   including Hold, which earns exactly zero and should sit near zero but ends
   near -10. The right chart says why: epsilon-greedy sends on 68% of
   exploratory steps, fatigue reaches a steady state of 1.00 against a churn
   threshold of 0.70, and the user quits before the agent can learn anything.
3. **Seven interventions, five failures.** Walk the refuted cards, not just the
   supported ones. Two of them were predictions that came from this diagnosis
   and turned out wrong.

Land it on the result table: the largest single gain came from the plainest
control in the study -- training 600 to 1500 episodes. The over-sending was
undertraining.

Closing line for this page: the surviving explanation is worth more shown next
to the five that failed than it would be shown alone.

## 5. Head to Head  (~70s)

This is the RQ2 answer.

Explain the matrix in one sentence: each cell is the row agent's mean reward
minus the column agent's, paired over the same (archetype, seed) cells, with a
dot where the difference clears p < 0.05.

Stress the pairing. The leaderboard says who scored highest; this says who
actually beats whom on the same episodes. Those come apart -- an agent can top
the mean while losing to a rival on most archetypes, because one archetype it
happens to suit carries its average.

Then the efficiency frontier: top-left is the objective, high click-through on
few sends. Bubble size is reward.

## 6. Ensemble  (~60s)

Two ensembles, reported side by side, because reporting only the one that works
would hide the finding.

- **Majority vote** is the control. With a silent member in a pool of three it
  stops being a vote and becomes an AND-gate -- a notification goes out only
  when both remaining members want one. Few sends, unusually high precision.
- **Confidence-gated fusion** is the proposal: each member's scores are
  temperature-matched so raw Q-value scale cannot dominate the mixture, weights
  come from a validation split disjoint from the test episodes, and the hold
  bias is line-searched on validation only.

Show the bias-sweep chart. It shows the whole trade-off rather than just the
chosen operating point, so you can see how sharp the optimum is and where the
ensemble sits relative to its own members.

Read the two caveats at the bottom out loud. Being explicit that the weighting
rule over-rewards abstention, and naming validation sample size as the cause,
is worth more than a clean number.

---

## If something goes wrong on camera

- **A page says a result file is missing.** Say so and move on -- the app is
  built to render whatever exists. Do not stop to fix it.
- **A chart looks empty.** Check the sidebar filter first; the agent multiselect
  persists across pages.
- **Play does nothing.** The traces are still simulating. Wait for the spinner.
- **Numbers differ from these notes.** Read the screen. These notes were written
  before the final run finished.

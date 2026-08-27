# Dashboard demo script

Running order for the recorded walkthrough. Roughly 6 minutes at a normal
speaking pace. Every number quoted below is read off the screen, not from these
notes -- if a number here disagrees with the dashboard, the dashboard is right.

Refreshed against the final run: Double DQN retuned and no longer silent, and
the DQN notebook re-executed so its cells agree with its results file.

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

Check the sidebar **Run** selector reads **Final run (5 seeds)**. That is the
default and the result the report quotes; the Tuning study option below it is a
longer 1500-episode sweep on a single seed, kept as supporting evidence only.

Read the leaderboard top to bottom: **DQN +6.55**, **Double DQN +2.73**,
**PPO -3.28**, **Fixed-18:00 -7.06**, **LinUCB -17.53**, **Random -125.69**.
Two things are worth saying out loud. DQN is the only method with a positive
mean reward. And LinUCB, a genuine learning algorithm, finishes *below* a
hard-coded 18:00 daily alarm -- that is the project's case for reinforcement
learning over bandits, visible in a single row.

The users-reached card reads **5 of 5**. Point at the readiness dots in the
sidebar too -- they show which result files exist. This is a live app reading
real artifacts, not a slide.

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

To get silent lanes reliably, pick **Double DQN (default settings)** or **PPO
(default settings)** from the Policy dropdown -- both stay quiet on all five.
**DQN (default settings)** sends on one lane of five. **LinUCB** sends on all
five, and **Tuned Double DQN** on four -- use that one to show the fix.

Have this ready, because it looks like a contradiction and someone will ask.
This page replays per-archetype demo checkpoints from
`tools/train_demo_agents.py`; the Overview leaderboard reads the four
notebooks' own result files. They are separate training runs, so an agent can be
quiet here and active there. Say it plainly rather than talking around it: the
demo checkpoints are older, and the leaderboard is the reported result.

Scrub the slider back to an interesting hour to freeze the frame while you talk.

## 3. Personalisation  (~70s)

This is the assignment's actual claim, so give it the time.

The chart plots, for each person, the hour the agent chose against the hour
exhaustive search proved was best for them. Grey tick is their true best hour;
the coloured marks are what the agent picked. Distance along the axis is the
error, in hours.

The **Run** selector at the top defaults to **Final run (5 seeds)**. Leave it
there -- that is the result the project reports, and it is the five-seed run the
leaderboard and the report quote. The Tuning study below it is a longer
1500-episode sweep on a single seed, kept as supporting evidence only.

Read the numbers off the screen rather than reciting them; the figures below are
what the current files hold. Double DQN picks **four distinct hours across the
five people**, hits **NightOwlStudent's exact best hour**, and averages a
**2.6-hour** timing error. Per person the error is 0h on NightOwlStudent, 1h on
NormalStudent, 2h on OfficeWorker, 3h on Housewife and 7h on NightShiftWorker.

Say the NightShiftWorker number out loud rather than hiding it. That user is
contacted only because the minimum-contact quota forces it -- exactly 7 sends a
week, the daily deadline and nothing more -- and the agent has no good hour to
pick, because their click propensity never clears the 0.340 break-even at any
hour. It is counted in every average above rather than dropped from it. An
average computed only over the users an agent chose to talk to is not a
personalisation result.

**Only Double DQN appears on this page.** The DQN teammate's notebook reports
reward, CTR and send rate but not the chosen hour, so there is nothing to plot
for it here; its numbers appear on every other page.

Say plainly what is not being claimed: these are per-archetype agents, so this
is personalisation by training, not one policy inferring who it is talking to.
That stricter question is RQ3 and the answer there is no.

Then the honesty panel, which exists to separate the agent's choices from the
quota's. **Four of the five users got far more than the seven-per-week minimum**
-- 43.3 sends for Housewife, 35.7 for OfficeWorker, 20.6 for NormalStudent, 19.3
for NightOwlStudent -- so on those four the timing shown is the policy's own
choice. NightShiftWorker sits at exactly 7.00, which is the deadline choosing and
the agent only picking which message to send. That distinction is the point of
the panel, and volunteering it is stronger than being asked.

## 4. Diagnosis  (~80s)

This page answers *why* some users received nothing, and it is the strongest
Part B material in the project.

**Read the opening line off the screen, do not recite it.** It is computed from
the result files rather than written into the page, so it changes whenever a
notebook is re-run. On the current files it names **PPO only**: NightShiftWorker,
NormalStudent and OfficeWorker received nothing from it -- three of twenty
agent-user pairs. LinUCB, DQN and Double DQN all contacted everyone.

Double DQN used to be on that list, and its absence is the story worth telling.
It contacted two of five users and scored exactly 0.00 on the rest. Two changes
fixed it: the discount factor from 0.99 to 0.90, and a minimum-contact rule of
one send per rolling day. Both are in section 8.1 of its notebook. That is the
strongest evidence for the conclusion this page lands on -- silence was an
economic result, not a broken optimiser -- so offer it before anyone digs.

If someone asks why the Live Simulation page still shows quieter agents, the
honest answer is that those demo checkpoints were trained earlier and separately
from the leaderboard runs. Say so rather than talking around it.

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

Housewife is exactly that case: **+45.41 for DQN and +43.96 for Double DQN**,
far above anything else in the table and doing a lot of work in both means.
Point at it rather than waiting to be asked.

The run selector is shared with the Overview, so confirm it still reads **Final
run (5 seeds)** -- whatever you picked there follows you here.

Then the efficiency frontier: top-left is the objective, high click-through on
few sends. Bubble size is reward.

## 6. Ensemble  (~60s)

Two ensembles, reported side by side, because reporting only the one that works
would hide the finding.

- **Majority vote** is the control. With a quiet member in a pool of three it
  stops being a vote and becomes an AND-gate -- a notification goes out only
  when both remaining members want one. Few sends, unusually high precision.
  On OfficeWorker the gated ensemble reaches **+5.71 at a CTR of 0.404**, the
  only result on that user that beats the Fixed-18:00 baseline's +2.02.
- **Confidence-gated fusion** is the proposal: each member's scores are
  temperature-matched so raw Q-value scale cannot dominate the mixture, weights
  come from a validation split disjoint from the test episodes, and the hold
  bias is line-searched on validation only.

Show the bias-sweep chart. It shows the whole trade-off rather than just the
chosen operating point, so you can see how sharp the optimum is and where the
ensemble sits relative to its own members.

Read the two caveats at the bottom out loud. Being explicit that the weighting
rule over-rewards abstention, and naming validation sample size as the cause, is
worth more than a clean number.

One more, and volunteer it before anyone asks: the member rewards on this page
do not match the Overview leaderboard, and that is a difference of agent rather
than of measurement. This page loads the saved checkpoints, trained on the
**mixed population** with the archetype hidden and resampled each episode; the
leaderboard reports **per-archetype specialists**. The gap between the two is
itself the RQ3 result -- one policy for everybody is worse than one per person.

---

## If something goes wrong on camera

- **A page says a result file is missing.** Say so and move on -- the app is
  built to render whatever exists. Do not stop to fix it.
- **A chart looks empty.** Check the sidebar filter first; the agent multiselect
  persists across pages.
- **Play does nothing.** The traces are still simulating. Wait for the spinner.
- **Numbers differ from these notes.** Read the screen, always. These notes
  track the final run, but any re-run moves them again.

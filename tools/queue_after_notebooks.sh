#!/usr/bin/env bash
# Wait for the two notebook runs to finish, then run the remaining experiments
# back-to-back. Each stage logs separately so a failure in one does not hide the
# others' output.
set -u
cd "D:/Uni Assignments/ML"
PY=".venv/Scripts/python.exe"

echo "[queue] waiting for notebook runs to finish..."
until grep -qE "Notebook completed|FAILED|Traceback" logs/linucb_full.log 2>/dev/null \
   && grep -qE "Notebook completed|FAILED|Traceback" logs/ddqn_full.log 2>/dev/null; do
  sleep 30
done
echo "[queue] notebooks done at $(date)"

echo "[queue] --- stage 1/3: protocol check ---"
"$PY" tools/check_results.py > logs/check_results.log 2>&1
echo "[queue] check_results exit=$?"

echo "[queue] --- stage 2/3: hyperparameter search (DDQN cell 41) ---"
"$PY" -u tools/run_notebook.py "Code/doubleDQN.ipynb" --until 41 \
      --skip 12 13 14 15 16 31 32 34 35 36 38 39 \
      > logs/ddqn_search.log 2>&1
echo "[queue] search exit=$?"

echo "[queue] --- stage 3/3: exploration study ---"
"$PY" -u -m cane.exploration_study > logs/exploration_study.log 2>&1
echo "[queue] exploration exit=$?"

echo "[queue] ALL STAGES COMPLETE at $(date)"
tail -20 logs/exploration_study.log

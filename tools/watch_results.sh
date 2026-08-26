#!/usr/bin/env bash
# Emit one line per result CSV as it lands, plus one line per notebook failure,
# then exit once all three notebooks have reported.
#
# DQN's CSV already exists from an earlier run, so it is watched by modification
# time rather than existence -- waiting for it to appear would wait forever.
set -u
cd "D:/Uni Assignments/ML"

LIN="Code/results/linucb_results.csv"
DDQN="Code/results/ddqn_results.csv"
DQN="DQN/results/dqn_results.csv"

mtime() { stat -c %Y "$1" 2>/dev/null || echo 0; }
rows()  { echo $(( $(wc -l < "$1" 2>/dev/null || echo 1) - 1 )); }

DQN0=$(mtime "$DQN")

lin_done=0; ddqn_done=0; dqn_done=0
lin_fail=0; ddqn_fail=0; dqn_fail=0

# A notebook that dies leaves no CSV, so silence would look identical to "still
# training". Watch the logs for the failure signatures too.
check_fail() {
  local log="$1" flag_name="$2" current="$3"
  [ "$current" -eq 1 ] && return 0
  if grep -qE "FAILED after|Traceback \(most recent" "$log" 2>/dev/null; then
    echo "FAILED: $log -- $(grep -m1 -E 'FAILED after|Traceback' "$log" | cut -c1-90)"
    return 1
  fi
  return 0
}

while true; do
  if [ "$lin_done" -eq 0 ] && [ -f "$LIN" ]; then
    lin_done=1
    echo "CSV LANDED  linucb_results.csv  ($(rows "$LIN") rows)"
  fi
  if [ "$ddqn_done" -eq 0 ] && [ -f "$DDQN" ]; then
    ddqn_done=1
    echo "CSV LANDED  ddqn_results.csv  ($(rows "$DDQN") rows)"
  fi
  if [ "$dqn_done" -eq 0 ] && [ "$(mtime "$DQN")" -gt "$DQN0" ]; then
    dqn_done=1
    echo "CSV LANDED  dqn_results.csv  (rewritten, $(rows "$DQN") rows)"
  fi

  if [ "$lin_fail" -eq 0 ] && [ "$lin_done" -eq 0 ]; then
    check_fail logs/linucb_full.log linucb "$lin_done" || lin_fail=1
  fi
  if [ "$ddqn_fail" -eq 0 ] && [ "$ddqn_done" -eq 0 ]; then
    check_fail logs/ddqn_full.log ddqn "$ddqn_done" || ddqn_fail=1
  fi
  if [ "$dqn_fail" -eq 0 ] && [ "$dqn_done" -eq 0 ]; then
    check_fail logs/dqn_full.log dqn "$dqn_done" || dqn_fail=1
  fi

  lin_settled=$(( lin_done + lin_fail ))
  ddqn_settled=$(( ddqn_done + ddqn_fail ))
  dqn_settled=$(( dqn_done + dqn_fail ))
  if [ $(( lin_settled > 0 ? 1 : 0 )) -eq 1 ] \
  && [ $(( ddqn_settled > 0 ? 1 : 0 )) -eq 1 ] \
  && [ $(( dqn_settled > 0 ? 1 : 0 )) -eq 1 ]; then
    echo "ALL THREE NOTEBOOKS SETTLED"
    exit 0
  fi

  sleep 20
done

#!/usr/bin/env bash
# Complete experiment, start to finish. ~8-10 minutes on a laptop with the defaults.
set -euo pipefail
OUT=${OUT:-results/loop}; SEED=${SEED:-7}
pip install -q cryptography scikit-learn scipy numpy
python3 loop.py --seed $SEED --trials 300 --dict 500 --iters 1000 --corpus 3000 --out $OUT   # attack -> analyze -> improve -> attack again
python3 bench.py $OUT                                                                       # timing micro-benchmarks
python3 build_models_js.py $OUT                                                             # export HE-v1/HE-v2 + attacker tables for the browser
python3 parity.py $OUT && node browser_sim/test_parity.mjs $(pwd)/$OUT                      # Python vs browser consistency (needs Node 18+)
python3 build_simulator.py $OUT simulator.html
python3 build_dashboard.py $OUT dashboard.html
echo "done: open dashboard.html and simulator.html; raw data in $OUT/trials.csv and $OUT/features_*.csv"

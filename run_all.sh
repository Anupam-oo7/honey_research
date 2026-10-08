#!/usr/bin/env bash
# Complete experiment, start to finish. ~8-10 minutes on a laptop with the defaults.
set -euo pipefail
OUT=${OUT:-results/loop}; SEED=${SEED:-7}
pip install -q cryptography scikit-learn scipy numpy
python3 -m research.loop --seed $SEED --trials 300 --dict 500 --iters 1000 --corpus 3000 --out $OUT   # attack -> analyze -> improve -> attack again
python3 -m research.bench $OUT                                                                        # timing micro-benchmarks
python3 -m research.build_models_js $OUT                                                              # export HE-v1/HE-v2 + attacker tables for the browser
python3 -m research.parity $OUT && node website/browser_sim/test_parity.mjs $(pwd)/$OUT              # Python vs browser consistency (needs Node 18+)
python3 website/build_simulator.py $OUT website/simulator.html
python3 -m website.build_dashboard $OUT website/dashboard.html
echo "done: open pages under website/; raw data in $OUT/trials.csv and $OUT/features_*.csv"

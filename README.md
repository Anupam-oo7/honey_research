# Honey Encryption vs. Password-Based Encryption

An experimental comparison of honey encryption (HE) and password-based encryption (PBE). The research pipeline generates controlled synthetic data, analyzes distinguishability, improves decoy models, and evaluates offline attacks.

All plaintexts and ground truth are synthetic; this project does not use real payment-card or account credentials.

## Project layout

- `models/`: reusable encryption implementations and synthetic ground-truth, feature, and Honey Encryption model code.
- `research/`: analysis, attack experiments, model-improvement loop, benchmarks, parity checks, and research pipeline entry points.
- `website/`: a plain-language home dashboard, detailed results and findings, a parameter-editable simulator, and interactive Honey security levels 1–3; includes their templates, browser simulator source, and HTML builders.
- `results/`: checked-in pilot data and experiment outputs. The larger CSV files are generated synthetic data.
- `run_all.sh`: run the complete experiment and rebuild its outputs.

## Run the experiment

Requires Python 3 and Node.js 18 or later for the browser parity check. Install the Python dependencies with:

```sh
python -m pip install -r requirements.txt
```

Then, from the project root, run the full pipeline in a Bash-compatible shell (for example, Git Bash on Windows):

```sh
bash run_all.sh
```

The script uses default settings of **1,500 paired trials** per dataset, a dictionary size of 500, 1,000 attack iterations, a corpus size of 5,000 and 20% of users outside the dictionary. On one CPU core the full run takes roughly 15-20 minutes. It writes experiment data under `results/loop/` and rebuilds the home dashboard, results and findings page, Honey security page, and interactive simulator under `website/`. Set `OUT` or `SEED` to override the output directory or random seed. Useful flags for `python -m research.loop`: `--trials`, `--ood` (share of users whose password is not in the attacker dictionary), `--shift` (how wrong the mis-informed attacker's population model is; 0 = correct), `--corpus`, `--dict`.

To open the checked-in pages without rerunning the experiment, open `website/dashboard.html`, `website/simulator.html`, `website/comparison.html`, or `website/honey_security.html` in a browser (Chart.js loads from a CDN, so an internet connection is needed for charts). Every page has a light/dark toggle.

## Limitations: what was fixed and what remains

| Earlier limitation | Now |
| --- | --- |
| Only 300 trials | 1,500 paired trials per dataset; Wilson and bootstrap confidence intervals throughout |
| The user's password was always in the attacker dictionary | 20% of users pick a password outside it. Authenticated PBE then lets the attacker see that every guess failed (100% of misses); with Honey Encryption the attacker still sees plausible results (0% see nothing) |
| The attacker was trained on the exact generator it attacked | A second attacker is trained on a shifted population and evaluated beside the matched one. Revised decoys stay near chance (held-out AUC 0.506 card, 0.505 account); the initial decoys do not (0.965, 0.834) |
| One improvement round | After the revision, the same analysis is re-run to check whether a second round would change anything. It would not (both datasets converged) |
| PBKDF2 iterations reduced | Unchanged by design, because success rates do not depend on it. Attacker cost does, so the results page converts the benchmark into the time for one dictionary sweep at 100k and 600k iterations |

Headline results (1,500 trials, 20% outside the dictionary): held-out AUC of the revised decoys is 0.494 (card) and 0.510 (account), down from 0.969 and 0.852. Real password found within 10 candidates, all users: PBE 79% / 81%, initial HE 61% / 50%, revised HE 33% / 36% (card / account). Among users whose password is in the dictionary PBE is always found, versus 42% / 44% for revised HE.

Still not addressed: all data is synthetic; the defender fits the revised decoy model on a corpus from the true generator (a real defender would have an imperfect sample); the attacker is a gradient-boosted classifier on hand-built features, not a proven optimal attacker; the dictionary has 500 entries; the mismatched attacker uses one specific shift.

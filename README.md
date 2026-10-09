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

The script uses default settings of 300 trials, a dictionary size of 500, 1,000 attack iterations, and a corpus size of 3,000. It writes experiment data under `results/loop/` and rebuilds the home dashboard, results and findings page, Honey security page, and interactive simulator under `website/`. Set `OUT` or `SEED` to override the output directory or random seed.

To open the checked-in pages without rerunning the experiment, open `website/dashboard.html`, `website/simulator.html`, `website/comparison.html`, or `website/honey_security.html` in a browser.

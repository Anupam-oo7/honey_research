# Honey Encryption vs. Password-Based Encryption

An experimental comparison of honey encryption (HE) and password-based encryption (PBE). The research pipeline generates controlled synthetic data, analyzes distinguishability, improves decoy models, and evaluates offline attacks.

All plaintexts and ground truth are synthetic; this project does not use real payment-card or account credentials.

## Project layout

- `common.py`, `pbe.py`, `honey.py`, `reveng.py`, `attack.py`: encryption implementations and the original pilot attack.
- `groundtruth.py`, `features.py`, `analysis.py`, `hemodels.py`, `improve.py`, `loop.py`: synthetic corpus generation and the research pipeline.
- `bench.py`, `parity.py`: timing and Python/browser consistency checks.
- `browser_sim/`: browser simulator source and parity test.
- `build_*.py`: build the browser simulator and dashboard from experiment results.
- `dashboard.html`, `simulator.html`: generated, ready-to-open browser demos.
- `results/`: checked-in pilot and experiment outputs. The larger CSV files are generated synthetic data.
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

The script uses default settings of 300 trials, a dictionary size of 500, 1,000 attack iterations, and a corpus size of 3,000. It writes experiment data under `results/loop/` and rebuilds `dashboard.html` and `simulator.html`. Set `OUT` or `SEED` to override the output directory or random seed.

To open the checked-in demos without rerunning the experiment, open `dashboard.html` or `simulator.html` in a browser.

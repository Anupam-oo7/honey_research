"""Embed the experiment outputs into the dashboard. Reads only files produced by loop.py / bench.py."""
import csv, json, subprocess, sys
d = sys.argv[1] if len(sys.argv) > 1 else "results/loop"; ld = lambda n: json.load(open(f"{d}/{n}.json"))
cols = ["workload", "trial", "scheme", "rank", "plaintext", "password", "dict_size", "wrong_total", "cand_l1", "tp_l1", "fp_l1", "pos_floor", "pos_l1", "cand_l3", "tp_l3", "fp_l3", "pos_l3", "pos_l3p"]
num = set(cols) - {"workload", "scheme", "plaintext", "password"}; rows = []
for r in csv.DictReader(open(f"{d}/trials.csv")): rows.append([(float(r[c]) if "." in r[c] else int(r[c])) if (c in num and r[c] != "") else (None if c in num else r[c]) for c in cols])
import os
par = subprocess.run(["node", "browser_sim/test_parity.mjs", os.path.abspath(d)], capture_output=True, text=True).stdout.splitlines()
parity = next((l for l in par if l.startswith("parity checks")), "not run")
repro = open(f"{d}/repro_check.txt").read().strip() if __import__("os").path.exists(f"{d}/repro_check.txt") else "not run"
data = {"report": ld("analysis_report"), "summary": ld("attack_summary"), "bench": ld("bench"), "manifest": ld("manifest"), "trials": {"cols": cols, "rows": rows}, "parity": parity, "repro": repro}
t = open("dashboard_loop_template.html").read().replace("/*DATA*/", json.dumps(data, separators=(",", ":")))
open(sys.argv[2], "w").write(t); print("dashboard bytes", len(t))

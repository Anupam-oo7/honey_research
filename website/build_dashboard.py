"""Build the static results, dashboard, and Honey security pages."""
import csv, json, sys
from pathlib import Path
from models import groundtruth as G

site = Path(__file__).resolve().parent
d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/loop")
ld = lambda n: json.loads((d / f"{n}.json").read_text(encoding="utf-8"))
report = ld("analysis_report")
summary = ld("attack_summary")
manifest = ld("manifest")

home = {
    workload: {
        "label": "card-number" if workload == "card" else "account",
        "v1": report[workload]["v1"]["l3"]["auc"],
        "v2": report[workload]["v2"]["l3"]["auc"],
        "trials": summary[workload]["schemes"]["HE-v1"]["n"],
        "note": "A score near 0.5 means the test had difficulty telling genuine items from decoys."
    }
    for workload in ("card", "cred")
}
template = (site / "dashboard_template.html").read_text(encoding="utf-8")
t = template.replace("/*DATA*/", json.dumps(home, separators=(",", ":")))
output = Path(sys.argv[2]) if len(sys.argv) > 2 else site / "dashboard.html"
output.write_text(t, encoding="utf-8")
print("dashboard bytes", len(t))

comparison = {"workloads": {}, "manifest": manifest, "bench": ld("bench")}
for workload in ("card", "cred"):
    schemes = {}
    for name, result in summary[workload]["schemes"].items():
        tier = "l1" if name.startswith("PBE") else "l3p"
        schemes[name] = {
            "cand_l1": result["cand_l1"],
            "dist_l1": result["dist_l1"],
            "success": result["success"][tier],
        }
    comparison["workloads"][workload] = {
        "report": report[workload],
        "paired": summary[workload]["paired"],
        "schemes": schemes,
    }
comparison_template = (site / "comparison_template.html").read_text(encoding="utf-8")
(site / "comparison.html").write_text(comparison_template.replace("/*DATA*/", json.dumps(comparison, separators=(",", ":"))), encoding="utf-8")
print("comparison bytes", (site / "comparison.html").stat().st_size)

security = {
    "users": G.USERS,
    "words": G.WORDS,
    "suffixes": G.SUF2,
    "bins": G.BINS,
    "datasets": {},
}
for workload in ("card", "cred"):
    security["datasets"][workload] = {}
    for version in ("v1", "v2"):
        with (d / f"features_{workload}_{version}.csv").open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row["split"] == "T"]
        feature = "first_digit" if workload == "card" else "pw_len"
        options = (
            {"first_digit": {"label": "First digit", "kind": "categories", "values": [str(i) for i in range(10)]},
             "digit_entropy": {"label": "Digit variety", "kind": "entropy", "values": ["<2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5+"]},
             "distinct_digits": {"label": "Different digits", "kind": "categories", "values": [str(i) for i in range(11)]}}
            if workload == "card" else
            {"pw_len": {"label": "Password length", "kind": "categories", "values": [str(i) for i in range(17)]},
             "digit_frac": {"label": "Digit share", "kind": "fraction", "values": [f"{i*10}–{(i+1)*10}%" for i in range(10)]},
             "symbol_frac": {"label": "Symbol share", "kind": "fraction", "values": [f"{i*10}–{(i+1)*10}%" for i in range(10)]},
             "has_upper": {"label": "Capitalization", "kind": "categories", "values": ["Lowercase", "Capitalized", "Unparsed"]},
             "word_rank": {"label": "Word popularity group", "kind": "rank", "values": ["Top 3", "4–10", "11–25", "26+", "Unparsed"]}}
        )

        def make_hist(name, config):
            values = config["values"]
            genuine = [0] * len(values)
            decoy = [0] * len(values)
            for row in rows:
                raw = float(row[name])
                if config["kind"] == "entropy":
                    index = 0 if raw < 2 else min(4, int((raw - 2) / .5) + 1)
                elif config["kind"] == "fraction":
                    index = min(9, max(0, int(raw * 10)))
                elif config["kind"] == "rank":
                    index = 4 if raw < 0 else 0 if raw < 3 else 1 if raw < 10 else 2 if raw < 25 else 3
                elif name == "has_upper":
                    index = 2 if raw < 0 else 1 if raw == 1 else 0
                else:
                    index = int(raw)
                    if index < 0 or index >= len(values):
                        continue
                bucket = genuine if row["label"] == "1" else decoy
                bucket[index] += 1
            total_genuine, total_decoy = sum(genuine), sum(decoy)
            return {
                "labels": values,
                "genuine": [round(n * 100 / total_genuine, 3) for n in genuine],
                "decoy": [round(n * 100 / total_decoy, 3) for n in decoy],
                "similarity": round((1 - sum(abs(a / total_genuine - b / total_decoy) for a, b in zip(genuine, decoy)) / 2) * 100, 1),
            }

        distributions = {name: {"label": config["label"], **make_hist(name, config)} for name, config in options.items()}
        samples = {
            "genuine": [r["output"] for r in rows if r["label"] == "1"][:4],
            "decoy": [r["output"] for r in rows if r["label"] == "0"][:4],
        }
        security["datasets"][workload][version] = {
            "l1": report[workload][version]["l1"],
            "distributions": distributions,
            "samples": samples,
            "count_per_group": sum(row["label"] == "1" for row in rows),
        }
security_template = (site / "honey_security_template.html").read_text(encoding="utf-8")
(site / "honey_security.html").write_text(security_template.replace("/*DATA*/", json.dumps(security, separators=(",", ":"))), encoding="utf-8")
print("honey security bytes", (site / "honey_security.html").stat().st_size)

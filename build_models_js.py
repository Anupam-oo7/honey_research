"""Assemble the browser simulator's model bundle from the experiment outputs (never edited by hand)."""
import json, sys
import groundtruth as G
out = sys.argv[1] if len(sys.argv) > 1 else "results/loop"
m = json.load(open(f"{out}/models.json")); rep = json.load(open(f"{out}/analysis_report.json"))
for wl in rep:
    m[wl]["changes"] = rep[wl]["improvement"]["changes"]
    m[wl]["auc"] = {"v1": rep[wl]["v1"]["l3"]["auc"], "v2": rep[wl]["v2"]["l3"]["auc"]}
m["gen"] = {"bin_w": G.BIN_W, "p_cap": G.P_CAP, "suf_w": {str(k): v for k, v in G.SUF_W.items()}}   # researcher-side generator only
json.dump(m, open(f"{out}/models_bundle.json", "w"), separators=(",", ":"))
print("bundle bytes", len(json.dumps(m, separators=(',', ':'))))

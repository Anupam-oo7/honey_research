"""IMPROVE stage: turn analysis findings into targeted changes of the decoy model.
Inputs: the v1 analysis report (from the attacker's corpus A) and a SEPARATE defender corpus F used only for fitting.
The held-out test corpus is never touched here."""
import copy
from collections import Counter
import groundtruth as G

CARD_MAP = {"luhn_valid": ["luhn"], "bin_id": ["bin_w"], "first_digit": ["bin_w"]}
CRED_MAP = {"has_upper": ["cap"], "word_bucket": ["word"], "word_rank": ["word"], "suffix_id": ["suffix"], "sym_suffix": ["suffix"],
            "cap_x_sym": ["interaction"], "pw_len": ["word", "suffix"], "digit_frac": ["suffix"], "symbol_frac": ["suffix"],
            "entropy": ["word", "suffix"]}


def triggers(rep):
    """feature -> {sources, evidence} for every feature flagged by L1, L2 or confirmed by L3 permutation importance."""
    t = {}
    def add(f, src, ev): t.setdefault(f, {"sources": [], "evidence": []}); t[f]["sources"].append(src); t[f]["evidence"].append(ev)
    for r in rep["l1"]:
        if r["flag"]: add(r["feature"], "L1", f"genuine pass {r['real_pass']:.2f} vs decoy pass {r['decoy_pass']:.2f}")
    for r in rep["l2"]:
        if r["flag"]: add(r["feature"], "L2", f"effect {r['effect']:.2f}, adj. p {r['p_adj']:.1e}")
    for r in rep["l3"]["importance"]:
        if r["perm_auc_drop"] >= 0.01: add(r["feature"], "L3", f"permutation AUC drop {r['perm_auc_drop']:.3f}")
    return t


def improve(spec, rep, F, alpha=0.5, only=None):
    on = lambda k: only is None or k == only   # `only` applies a single component (used for the ablation study)
    wl, new, t = spec["workload"], copy.deepcopy(spec), triggers(rep)
    mp = CARD_MAP if wl == "card" else CRED_MAP
    comps, why, unaddressed = {}, {}, []
    for f, info in t.items():
        if f not in mp: unaddressed.append({"feature": f, **info}); continue
        for c in mp[f]: comps.setdefault(c, set()).add(f)
    changes = []
    new["version"] = "v2"
    if wl == "card":
        if "luhn" in comps and on("luhn"):
            new["luhn"] = True
            changes.append({"key": "luhn", "component": "check digit", "action": "generate decoys with a valid Luhn check digit", "triggered_by": sorted(comps["luhn"])})
        if "bin_w" in comps and on("bin_w"):
            cnt = Counter(m[:6] for m in F); tot = len(F) + alpha * len(G.BINS)
            new["bin_w"] = [(cnt.get(b, 0) + alpha) / tot for b in G.BINS]
            changes.append({"key": "bin_w", "component": "issuer (BIN) weights", "action": f"refit weights from {len(F)} defender-corpus cards (add-{alpha} smoothing)",
                            "triggered_by": sorted(comps["bin_w"]), "fitted": [round(w, 4) for w in new["bin_w"]]})
    else:
        P = [G.parse_pw(m.split(":", 1)[1]) for m in F]
        n = len(P); U = lambda k: [1.0 / k] * k
        need_cap, need_word = "cap" in comps, "word" in comps
        need_suf, need_int = "suffix" in comps, "interaction" in comps
        Pc = U(2); Pw = U(len(G.WORDS)); Ps = {0: U(len(G.SUF2)), 1: U(len(G.SUF2))}
        if need_cap and on("cap"):
            c = Counter(p[0] for p in P); Pc = [(c.get(k, 0) + alpha) / (n + 2 * alpha) for k in (0, 1)]
            changes.append({"key": "cap", "component": "capitalisation rate", "action": "fit P(capitalised)", "triggered_by": sorted(comps["cap"]), "fitted": [round(x, 4) for x in Pc]})
        if need_word and on("word"):
            c = Counter(p[1] for p in P); Pw = [(c.get(k, 0) + alpha) / (n + len(G.WORDS) * alpha) for k in range(len(G.WORDS))]
            changes.append({"key": "word", "component": "base-word popularity", "action": "fit P(word) from frequencies", "triggered_by": sorted(comps["word"]),
                            "fitted_top5": [round(x, 4) for x in Pw[:5]]})
        if (need_suf or need_int) and on("suffix"):
            if need_int:   # interaction flagged: suffix distribution conditioned on capitalisation
                for cp in (0, 1):
                    c = Counter(p[2] for p in P if p[0] == cp); m = sum(c.values())
                    Ps[cp] = [(c.get(k, 0) + alpha) / (m + len(G.SUF2) * alpha) for k in range(len(G.SUF2))]
                changes.append({"key": "suffix", "component": "suffix | capitalisation", "action": "fit P(suffix | capitalised) separately for each case",
                                "triggered_by": sorted(comps.get("interaction", set()) | comps.get("suffix", set()))})
            else:
                c = Counter(p[2] for p in P); Ps[0] = Ps[1] = [(c.get(k, 0) + alpha) / (n + len(G.SUF2) * alpha) for k in range(len(G.SUF2))]
                changes.append({"key": "suffix", "component": "suffix popularity", "action": "fit pooled P(suffix)", "triggered_by": sorted(comps["suffix"])})
        w = []
        for cp in (0, 1):
            for wi in range(len(G.WORDS)):
                for si in range(len(G.SUF2)): w.append(Pc[cp] * Pw[wi] * Ps[cp][si])
        new["pw_w"] = w
    return new, changes, unaddressed

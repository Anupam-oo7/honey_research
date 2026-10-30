"""ATTACK -> ANALYZE -> IDENTIFY WEAKNESS -> IMPROVE DECOYS -> ATTACK AGAIN.
Run:  python -m research.loop --seed 7 --trials 1500 --dict 500 --iters 1000 --corpus 5000 --out results/loop
Everything below is synthetic/controlled (see groundtruth.py). Nothing is hard-coded: every reported number is computed here."""
import argparse, csv, hashlib, json, math, os, platform, random, sys, time
from pathlib import Path
import numpy as np, scipy, sklearn
from sklearn.metrics import roc_auc_score
from models import common, features as FT, groundtruth as G, hemodels as HM, pbe
from research import analysis as AN, improve as IM

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--trials", type=int, default=1500)
ap.add_argument("--dict", type=int, default=500); ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--corpus", type=int, default=5000); ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--workloads", nargs="+", default=["card", "cred"]); ap.add_argument("--ood", type=float, default=0.2, help="share of users whose password is NOT in the attacker dictionary")
ap.add_argument("--shift", type=float, default=1.0, help="how wrong the mismatched attacker's population model is (0 = correct)"); ap.add_argument("--out", default="results/loop")
args = ap.parse_args()
common.ITERS = args.iters; os.makedirs(args.out, exist_ok=True)
KS = [1, 2, 5, 10, 20, 50, 100, 200]
mat = lambda wl, xs: np.array([FT.extract(wl, s) for s in xs], float)


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def decoys(sch, msgs, dic, zw, rng):
    """Real pipeline: encrypt each genuine message under a Zipf-chosen password with a fresh salt, then decrypt with a
    uniformly chosen WRONG dictionary password. The result is what the attacker sees for a wrong guess."""
    D, out = len(dic), []
    for m in msgs:
        salt = rng.randbytes(16); r = rng.choices(range(D), zw)[0]; blob = sch.enc(dic[r], m.encode(), salt)
        w = rng.randrange(D - 1); w += w >= r
        out.append(sch.dec(dic[w], blob).decode("latin1"))
        if len(out) % 400 == 0: common._C.clear()
    common._C.clear(); return out


def analyze(wl, spec, ctx, light=False):
    """ANALYZE: generate decoys for this HE version, run L1 / L2 / L3, and train the attacker's classifier."""
    sch, rng, nm = HM.make(spec), ctx["rng"], FT.names(wl); random.seed(rng.getrandbits(64))
    dA, dT = decoys(sch, ctx["A"], ctx["dic"], ctx["zw"], rng), decoys(sch, ctx["T"], ctx["dic"], ctx["zw"], rng)
    FDA, FDT, FRA, FRT = mat(wl, dA), mat(wl, dT), ctx["FRA"], ctx["FRT"]
    tr, va = ctx["perm"][:int(.7 * len(ctx["perm"]))], ctx["perm"][int(.7 * len(ctx["perm"])):]
    clf, tau = AN.l3_train(FRA[tr], FDA[tr], FRA[va], args.seed)
    l3, sr, sd = AN.l3_eval(clf, tau, nm, FRA[tr], FDA[tr], FRT, FDT, args.seed)
    if light: return {"l3": l3}, None
    FRAm = ctx["FRAm"]   # attacker who learned "genuine" from a WRONG population (shifted generator)
    clf_m, tau_m = AN.l3_train(FRAm[tr], FDA[tr], FRAm[va], args.seed)
    l3m, _, _ = AN.l3_eval(clf_m, tau_m, nm, FRAm[tr], FDA[tr], FRT, FDT, args.seed, B=100)
    nb = AN.nb_fit(wl, FRA, FDA); nr, nd = AN.nb_score(wl, nb, FRT), AN.nb_score(wl, nb, FDT)
    rep = {"version": spec["version"], "l1": AN.l1(wl, FRA, FDA), "l2": AN.l2(wl, FRA, FDA), "l3": l3,
           "nb": {"auc": float(roc_auc_score(np.r_[np.ones(len(nr)), np.zeros(len(nd))], np.r_[nr, nd])),
                  "tpr": float((nr >= nb["tau"]).mean()), "fpr": float((nd >= nb["tau"]).mean())}}
    rep["l3_mismatch"] = {k: l3m[k] for k in ("auc", "auc_ci", "tpr", "fpr", "advantage")}
    path = f"{args.out}/features_{wl}_{spec['version']}.csv"
    with open(path, "w", newline="") as f:   # raw material for ROC / importance / re-analysis
        w = csv.writer(f); w.writerow(["split", "label", "output", "gb_score"] + nm)
        for sp, FR_, FD_, R, Dd in (("A", FRA, FDA, ctx["A"], dA), ("T", FRT, FDT, ctx["T"], dT)):
            for lab, X, S in ((1, FR_, R), (0, FD_, Dd)):
                sc = AN.score(clf, X)
                for s, x, c in zip(S, X, sc): w.writerow([sp, lab, s, round(float(c), 4)] + [round(float(v), 4) for v in x])
    return rep, {"clf": clf, "tau": tau, "clf_m": clf_m, "tau_m": tau_m, "nb": nb, "sr": sr, "sd": sd}


BIG = 10 ** 9   # "never found" position (password outside the dictionary)


def attack(wl, schemes, models, ctx):
    """ATTACK: offline dictionary attack, identical budget and inputs for every scheme in a trial (paired design).
    A share `--ood` of users picks a password OUTSIDE the dictionary: the attacker can never succeed on them and must judge from outputs alone."""
    dic, D, zw, rng, N = ctx["dic"], len(ctx["dic"]), ctx["zw"], ctx["rng"], args.trials
    rows, scores, ranks = [], {s.name: np.zeros((N, D), np.float32) for s in schemes if s.fam == "HE"}, []
    lp, t0 = np.log(np.arange(1, D + 1)), time.time()
    for t in range(N):
        salt = rng.randbytes(16); pt = G.GEN[wl](rng); r = rng.choices(range(D), zw)[0]; ood = rng.random() < args.ood
        pw = f"zz{rng.getrandbits(48):012x}" if ood else dic[r]; ranks.append(-1 if ood else r)
        common._C.clear(); random.seed(rng.getrandbits(64))
        for sch in schemes:
            blob = sch.enc(pw, pt.encode(), salt); outs = [sch.dec(g, blob) for g in dic]
            strs = [o.decode("latin1") if o is not None else None for o in outs]
            a1 = np.array([s is not None and FT.l1_ok(wl, s) for s in strs])
            if not ood:
                assert outs[r] == pt.encode(), "correct password must recover the plaintext"; assert a1[r]
            row = {"workload": wl, "trial": t, "scheme": sch.name, "in_dict": int(not ood), "rank": -1 if ood else r, "plaintext": pt, "password": pw,
                   "dict_size": D, "wrong_total": D if ood else D - 1, "cand_l1": int(a1.sum()), "tp_l1": 0 if ood else 1,
                   "fp_l1": int(a1.sum()) - (0 if ood else 1), "pos_floor": BIG if ood else r, "pos_l1": BIG if ood else int(a1[:r].sum())}
            if sch.fam == "HE":
                m = models[sch.name]; X = np.array([FT.extract(wl, s) for s in strs])
                sc, scm = AN.score(m["clf"], X), AN.score(m["clf_m"], X); scores[sch.name][t] = sc
                acc, accm = sc >= m["tau"], scm >= m["tau_m"]
                pos = lambda v: BIG if ood else float((v > v[r]).sum() + .5 * ((v == v[r]).sum() - 1))
                tp3 = 0 if ood else int(acc[r])
                row.update(cand_l3=int(acc.sum()), tp_l3=tp3, fp_l3=int(acc.sum()) - tp3, cand_l3m=int(accm.sum()),
                           pos_l3=pos(sc), pos_l3p=pos(sc - lp), pos_l3m=pos(scm), pos_l3pm=pos(scm - lp))
            rows.append(row)
        if t % 100 == 99: print(f"   {wl}: trial {t + 1}/{N}  [{time.time() - t0:.0f}s]", flush=True)
    return rows, scores, ranks


def summarize(wl, rows, scores, ranks):
    D, by, out = rows[0]["dict_size"], {}, {}
    for r in rows: by.setdefault(r["scheme"], []).append(r)
    ks = [k for k in KS if k <= D]
    for name, rs in by.items():
        ins, oo = [r for r in rs if r["in_dict"]], [r for r in rs if not r["in_dict"]]; n, ni = len(rs), len(ins); he = "tp_l3" in rs[0]
        S = {"n": n, "n_in": ni, "n_ood": len(oo), "cand_l1": float(np.mean([r["cand_l1"] for r in ins]))}
        wt = sum(r["wrong_total"] for r in ins)
        for tier in ("l1",) + (("l3",) if he else ()):
            tp, fp = sum(r["tp_" + tier] for r in ins), sum(r["fp_" + tier] for r in ins)
            S[f"tpr_{tier}"] = {"v": tp / ni, "ci": wilson(tp, ni)}; S[f"fpr_{tier}"] = {"v": fp / wt, "ci": wilson(fp, wt)}
            S[f"adv_{tier}"] = tp / ni - fp / wt; S[f"dist_{tier}"] = 1 - fp / wt
            if tier == "l3": S["cand_l3"] = float(np.mean([r["cand_l3"] for r in ins]))
        tiers = [("floor", "pos_floor"), ("l1", "pos_l1")] + ([("l3", "pos_l3"), ("l3p", "pos_l3p"), ("l3m", "pos_l3m"), ("l3pm", "pos_l3pm")] if he else [])
        for key, pool in (("success", rs), ("success_in", ins)):   # success = all users (incl. out-of-dictionary); success_in = dictionary users only
            S[key] = {tier: [{"k": k, "p": sum(r[col] < k for r in pool) / len(pool), "ci": wilson(sum(r[col] < k for r in pool), len(pool))} for k in ks] for tier, col in tiers}
        if oo:   # what does the attacker see when NO guess is right?
            z = lambda col: float(np.mean([r[col] for r in oo])); zr = lambda col: float(np.mean([r[col] == 0 for r in oo]))
            S["ood"] = {"cand_l1": z("cand_l1"), "learns_miss_l1": zr("cand_l1")}
            if he: S["ood"].update(cand_l3=z("cand_l3"), learns_miss_l3=zr("cand_l3"), cand_l3m=z("cand_l3m"), learns_miss_l3m=zr("cand_l3m"))
        if name in scores:
            idx_all = np.array(ranks); keep = idx_all >= 0; sc = scores[name][keep]; idx = idx_all[keep]
            g = sc[np.arange(len(idx)), idx]; mask = np.ones_like(sc, bool); mask[np.arange(len(idx)), idx] = False
            S["attack_auc"] = float(roc_auc_score(np.r_[np.ones(len(g)), np.zeros(mask.sum())], np.r_[g, sc[mask]]))
        out[name] = S
    paired, rs_ = {}, np.random.RandomState(args.seed)
    if "HE-v1" in by and "HE-v2" in by:
        for tier, col in (("l3", "pos_l3"), ("l3p", "pos_l3p")):
            for k in (1, 10, 50):
                if k > D: continue
                a = np.array([r[col] < k for r in by["HE-v1"]], float); b = np.array([r[col] < k for r in by["HE-v2"]], float); d = b - a
                bs = [d[rs_.randint(0, len(d), len(d))].mean() for _ in range(2000)]
                paired[f"{tier}@{k}"] = {"v1": a.mean(), "v2": b.mean(), "delta": d.mean(), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out, paired


def verdict(rep1, rep2, sr1, sd1, sr2, sd2, rs):
    d, ci = AN.auc_delta(sr1, sd1, sr2, sd2, 300, rs); a1, a2 = rep1["l3"]["auc"], rep2["l3"]["auc"]
    t = ("held-out AUC fell from %.3f to %.3f (change %+.3f, 95%% CI [%+.3f, %+.3f] excludes 0): v2 decoys are harder to separate." if ci[1] < 0 else
         "held-out AUC rose from %.3f to %.3f (change %+.3f, 95%% CI [%+.3f, %+.3f] excludes 0): v2 decoys are EASIER to separate." if ci[0] > 0 else
         "held-out AUC went from %.3f to %.3f (change %+.3f, 95%% CI [%+.3f, %+.3f] includes 0): no demonstrated change.") % (a1, a2, d, ci[0], ci[1])
    lo = rep2["l3"]["auc_ci"][0]
    t += (" Residual leakage remains: v2 AUC CI lower bound %.3f is above 0.5." % lo) if lo > 0.5 else (" v2 AUC is not distinguishable from 0.5 at this sample size (CI lower bound %.3f)." % lo)
    return {"auc_v1": a1, "auc_v2": a2, "delta": d, "delta_ci": ci, "text": t}


def run(wl, rng):
    print(f"== workload: {wl} ==", flush=True)
    dic = common.dictionary(args.dict); ctx = {"rng": rng, "dic": dic, "zw": common.zipf(len(dic))}
    ctx["A"], F, ctx["T"] = (G.corpus(wl, args.corpus, rng) for _ in range(3))   # attacker corpus / defender fit corpus / held-out test corpus
    ctx["FRA"], ctx["FRT"] = mat(wl, ctx["A"]), mat(wl, ctx["T"]); ctx["perm"] = np.random.RandomState(args.seed).permutation(args.corpus)
    with G.shifted(args.shift): ctx["Am"] = G.corpus(wl, args.corpus, rng)   # what a mis-informed attacker believes "genuine" looks like
    ctx["FRAm"] = mat(wl, ctx["Am"])
    spec1 = HM.spec_v1(wl)
    print(" [1] attack-surface analysis of HE-v1 (L1/L2/L3)", flush=True); rep1, m1 = analyze(wl, spec1, ctx)
    print(" [2] identify weaknesses + [3] modify decoy model", flush=True); spec2, changes, unaddr = IM.improve(spec1, rep1, F, args.alpha)
    print("     changes:", [c["component"] for c in changes], flush=True)
    print(" [4] re-run the SAME analysis on HE-v2", flush=True); rep2, m2 = analyze(wl, spec2, ctx)
    _, ch3, un3 = IM.improve(spec2, rep2, F, args.alpha)   # would a 2nd improvement round change anything?
    conv = {"round2_changes": [c["component"] for c in ch3], "round2_unaddressed": [u["feature"] for u in un3], "converged": not ch3}
    print("     round-2 check:", "converged (no findings left to fix)" if not ch3 else f"would still change {conv['round2_changes']}", flush=True)
    abl = []   # ablation: apply ONE finding-driven change at a time, re-measure held-out AUC (which fix removes which leak?)
    for c in changes:
        sp, _, _ = IM.improve(spec1, rep1, F, args.alpha, only=c["key"]); r_, _ = analyze(wl, sp, ctx, light=True)
        abl.append({"key": c["key"], "component": c["component"], "auc": r_["l3"]["auc"], "auc_ci": r_["l3"]["auc_ci"]})
        print(f"     ablation [{c['key']} only] held-out AUC {r_['l3']['auc']:.3f}", flush=True)
    v = verdict(rep1, rep2, m1["sr"], m1["sd"], m2["sr"], m2["sd"], np.random.RandomState(args.seed + 1))
    schemes = HM.pbe_schemes() + [HM.make(spec1), HM.make(spec2)]
    print(" [5] offline dictionary attack on PBE, HE-v1, HE-v2 (paired trials)", flush=True)
    rows, scores, ranks = attack(wl, schemes, {"HE-v1": m1, "HE-v2": m2}, ctx); summ, paired = summarize(wl, rows, scores, ranks)
    report = {"workload": wl, "v1": rep1, "improvement": {"changes": changes, "unaddressed": unaddr}, "ablation": abl, "v2": rep2, "verdict": v, "convergence": conv}
    models = {"v1": spec1, "v2": spec2, "nb": {"v1": m1["nb"], "v2": m2["nb"]}}
    return report, {"schemes": summ, "paired": paired}, models, rows


def main():
    rng = random.Random(args.seed); np.random.seed(args.seed); pbe.set_rng(lambda n: rng.randbytes(n))
    rep, summ, mod, rows = {}, {}, {}, []
    for wl in args.workloads:
        rep[wl], summ[wl], mod[wl], r = run(wl, rng); rows += r
    cols = ["workload", "trial", "scheme", "in_dict", "rank", "plaintext", "password", "dict_size", "wrong_total", "cand_l1", "tp_l1", "fp_l1", "pos_floor",
            "pos_l1", "cand_l3", "tp_l3", "fp_l3", "pos_l3", "pos_l3p", "cand_l3m", "pos_l3m", "pos_l3pm"]
    with open(f"{args.out}/trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, restval=""); w.writeheader(); w.writerows(rows)
    public = {"bins": G.BINS, "users": G.USERS, "words": G.WORDS, "suf": G.SUF2, "sym": sorted(G.SYM), "vocab": G.CRED_VOCAB}
    for name, obj in (("analysis_report", rep), ("attack_summary", summ), ("models", {"public": public, **mod})):
        json.dump(obj, open(f"{args.out}/{name}.json", "w"), indent=1)
    root = Path(__file__).resolve().parents[1]
    source_files = ("models/common.py", "models/pbe.py", "models/honey.py", "models/reveng.py", "models/groundtruth.py",
                    "models/features.py", "models/hemodels.py", "research/analysis.py", "research/attack.py",
                    "research/improve.py", "research/loop.py")
    src = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()[:16] for p in source_files}
    outs = {p: hashlib.sha256(open(f"{args.out}/{p}", "rb").read()).hexdigest()[:16] for p in sorted(os.listdir(args.out)) if p != "manifest.json" and p != "bench.json"}
    json.dump({"args": vars(args), "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "source_sha256_16": src, "output_sha256_16": outs, "synthetic": True,
               "assumptions": ["plaintexts come from synthetic generators (groundtruth.py), not real users",
                               "a share --ood of users pick a password outside the attacker dictionary (success impossible for them)",
                               "attacker password-popularity prior is Zipf(1); a mismatched attacker is also evaluated (--shift)",
                               "main attacker trains on the true generator (upper bound); a mismatched attacker trained on a shifted generator is reported next to it",
                               "PBKDF2 iterations are reduced for simulation speed; success rates do not depend on it, attacker cost does (see bench kdf_ms)",
                               "improvement rounds repeat until no finding is left; round-2 convergence is checked and reported"]},
              open(f"{args.out}/manifest.json", "w"), indent=1)
    for wl in args.workloads: print(wl, "->", rep[wl]["verdict"]["text"])


if __name__ == "__main__":
    main()

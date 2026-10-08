"""L1 structural, L2 statistical and L3 learned analysis of real-vs-decoy outputs."""
import math
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from models import features as FT

FLAG_P, FLAG_EFFECT, FLAG_L1_REAL, FLAG_L1_DECOY, FLAG_PERM = 0.01, 0.10, 0.99, 0.95, 0.01


def l1(wl, FR, FD):
    """Per structural predicate: pass rate on genuine vs decoy outputs. A rule that every genuine output passes but
    many decoys fail is a free filter for the attacker."""
    nm, out = FT.names(wl), []
    for k in FT.SPEC[wl]["L1"]:
        i = nm.index(k); rp, dp = float((FR[:, i] == 1).mean()), float((FD[:, i] == 1).mean())
        out.append({"feature": k, "real_pass": round(rp, 4), "decoy_pass": round(dp, 4),
                    "decoys_rejected": round(1 - dp, 4), "flag": bool(rp >= FLAG_L1_REAL and dp <= FLAG_L1_DECOY)})
    return out


def l2(wl, FR, FD):
    """Two-sample tests per distributional feature: chi-square (Cramer's V) for categorical, KS (D) for numeric.
    Bonferroni-corrected over the number of tests."""
    nm, spec, rows = FT.names(wl), FT.SPEC[wl]["L2"], []
    for k, typ in spec.items():
        i = nm.index(k); a, b = FR[:, i], FD[:, i]
        if typ == "cat":
            vals = sorted(set(a) | set(b)); tab = np.array([[(a == v).sum() for v in vals], [(b == v).sum() for v in vals]])
            tab = tab[:, tab.sum(0) > 0]
            if tab.shape[1] < 2: chi, p, eff = 0.0, 1.0, 0.0
            else:
                chi, p, _, _ = stats.chi2_contingency(tab); eff = math.sqrt(chi / tab.sum())
        else:
            res = stats.ks_2samp(a, b); p, eff = float(res.pvalue), float(res.statistic)
        rows.append({"feature": k, "type": typ, "effect": round(eff, 4), "p": float(p), "real_mean": round(float(a.mean()), 4),
                     "decoy_mean": round(float(b.mean()), 4)})
    m = len(rows)
    for r in rows:
        r["p_adj"] = min(1.0, r["p"] * m); r["flag"] = bool(r["p_adj"] < FLAG_P and r["effect"] >= FLAG_EFFECT)
    return sorted(rows, key=lambda r: -r["effect"])


def score(clf, X):
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def auc_ci(sr, sd, B, rs):
    y = np.r_[np.ones(len(sr)), np.zeros(len(sd))]; s = np.r_[sr, sd]; est = roc_auc_score(y, s); bs = []
    for _ in range(B):
        i, j = rs.randint(0, len(sr), len(sr)), rs.randint(0, len(sd), len(sd))
        bs.append(roc_auc_score(y, np.r_[sr[i], sd[j]]))
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def auc_delta(sr1, sd1, sr2, sd2, B, rs):
    """Bootstrap CI of AUC(v2) - AUC(v1) (independent held-out samples)."""
    y1 = np.r_[np.ones(len(sr1)), np.zeros(len(sd1))]; y2 = np.r_[np.ones(len(sr2)), np.zeros(len(sd2))]; ds = []
    for _ in range(B):
        a = roc_auc_score(y1, np.r_[sr1[rs.randint(0, len(sr1), len(sr1))], sd1[rs.randint(0, len(sd1), len(sd1))]])
        b = roc_auc_score(y2, np.r_[sr2[rs.randint(0, len(sr2), len(sr2))], sd2[rs.randint(0, len(sd2), len(sd2))]])
        ds.append(b - a)
    return float(np.mean(ds)), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def l3_train(FRtr, FDtr, FRval, seed):
    """Train on the attacker's own corpus; pick the accept threshold on a separate validation slice so that 95% of
    genuine validation outputs are accepted."""
    X = np.vstack([FRtr, FDtr]); y = np.r_[np.ones(len(FRtr)), np.zeros(len(FDtr))]
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, subsample=0.8, random_state=seed).fit(X, y)
    return clf, float(np.percentile(score(clf, FRval), 5))


def l3_eval(clf, tau, names, FRtr, FDtr, FRte, FDte, seed, B=300):
    """Evaluate on HELD-OUT data (fresh real messages + fresh decoys) the classifier never saw."""
    rs = np.random.RandomState(seed); sr, sd = score(clf, FRte), score(clf, FDte)
    auc, ci = auc_ci(sr, sd, B, rs); tpr, fpr = float((sr >= tau).mean()), float((sd >= tau).mean())
    tr_auc = float(roc_auc_score(np.r_[np.ones(len(FRtr)), np.zeros(len(FDtr))], np.r_[score(clf, FRtr), score(clf, FDtr)]))
    X = np.vstack([FRte, FDte]); y = np.r_[np.ones(len(FRte)), np.zeros(len(FDte))]
    pi = permutation_importance(clf, X, y, scoring="roc_auc", n_repeats=5, random_state=seed)
    fpr_c, tpr_c, _ = roc_curve(y, np.r_[sr, sd]); pick = np.linspace(0, len(fpr_c) - 1, min(60, len(fpr_c))).astype(int)
    imp = sorted(({"feature": n, "perm_auc_drop": round(float(m), 4), "gb_importance": round(float(g), 4)}
                  for n, m, g in zip(names, pi.importances_mean, clf.feature_importances_)), key=lambda r: -r["perm_auc_drop"])
    return {"auc": auc, "auc_ci": ci, "train_auc": tr_auc, "tau": tau, "tpr": tpr, "fpr": fpr, "advantage": tpr - fpr,
            "importance": imp, "roc": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr_c[pick], tpr_c[pick])]}, sr, sd


def nb_fit(wl, FR, FD, alpha=1.0):
    """Tiny naive-Bayes attacker over a few categorical features (exportable to the browser simulator)."""
    nm, feats, llr = FT.names(wl), FT.NB_FEATS[wl], {}
    for f in feats:
        i = nm.index(f); vals = sorted(set(FR[:, i]) | set(FD[:, i])); K = len(vals); t = {}
        for v in vals:
            pr = ((FR[:, i] == v).sum() + alpha) / (len(FR) + alpha * K); pd = ((FD[:, i] == v).sum() + alpha) / (len(FD) + alpha * K)
            t[str(int(v))] = round(float(math.log(pr / pd)), 6)
        llr[f] = t
    model = {"feats": feats, "llr": llr}
    model["tau"] = float(np.percentile(nb_score(wl, model, FR), 5))
    return model


def nb_score(wl, model, X):
    nm = FT.names(wl); s = np.zeros(len(X))
    for f in model["feats"]:
        i = nm.index(f); t = model["llr"][f]
        s += np.array([t.get(str(int(v)), 0.0) for v in X[:, i]])
    return s

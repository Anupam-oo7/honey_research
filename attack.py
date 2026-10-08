"""Offline-guessing simulator + reverse-engineering analysis + dataset generation.
Usage: python attack.py [--trials 300] [--dict 1000]"""
import argparse, csv, json, math, os, random, re, time
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import common, pbe, honey, reveng
from common import luhn_ok, luhn_check

ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=300); ap.add_argument("--dict", type=int, default=1000)
A = ap.parse_args()
random.seed(1337); np.random.seed(1337)
OUT = "results"; os.makedirs(OUT, exist_ok=True)

DICT = common.dictionary(A.dict); D = len(DICT); ZW = common.zipf(D)
SALTS = [os.urandom(16) for _ in range(8)]
BINS = ["411111", "424242", "453201", "510510", "555555", "601100"]
REAL_W = [40, 25, 12, 10, 8, 5]
USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
         "karan", "leela", "mohan", "nisha", "omar", "priya", "quinn", "rahul", "sana", "tarun"]
VOCAB = [w for w in common.dictionary(700) if len(w) <= 8][:150]
VRANK = {w: i for i, w in enumerate(VOCAB)}; VW = common.zipf(len(VOCAB))
CLEN = lambda s: s


# ---------- workloads ----------
def real_card():
    s = random.choices(BINS, REAL_W)[0] + f"{random.randrange(10**9):09d}"
    return (s + luhn_check(s)).encode()


def real_cred():
    return f"{random.choice(USERS)}:{random.choices(VOCAB, VW)[0]}".encode()


CRED_RE = re.compile(rb"^[a-z]{3,10}:[\x21-\x7e]{1,12}$")
valid = {"card": lambda b: len(b) == 16 and b.isdigit() and luhn_ok(b.decode()),
         "cred": lambda b: bool(CRED_RE.match(b))}
real = {"card": real_card, "cred": real_cred}


def ent(s):
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0


CARD_F = ["luhn_valid", "bin_id", "first_digit", "entropy", "distinct_digits", "max_digit_count"]
CRED_F = ["user_known", "pw_len", "vocab_rank", "alpha_frac", "digit_frac", "symbol_frac", "entropy", "vowel_ratio"]
FN = {"card": CARD_F, "cred": CRED_F}


def feats(wl, b):
    s = b.decode("latin1")
    if wl == "card":
        if not s.isdigit():
            return [0, -2, -1, 0, 0, 0]
        c = Counter(s)
        return [int(luhn_ok(s)), BINS.index(s[:6]) if s[:6] in BINS else -1, int(s[0]), ent(s), len(c), max(c.values())]
    u, _, p = s.partition(":")
    n = max(len(p), 1)
    return [int(u in USERS), len(p), VRANK.get(p, -1), sum(ch.isalpha() for ch in p) / n, sum(ch.isdigit() for ch in p) / n,
            sum(not ch.isalnum() for ch in p) / n, ent(p), sum(ch in "aeiou" for ch in p) / n]


def show(b):
    return "".join(c if 32 <= ord(c) < 127 else f"\\x{ord(c):02x}" for c in b.decode("latin1"))


# ---------- schemes ----------
class S:
    def __init__(s, name, wl, fam, enc, dec, kind):
        s.name, s.wl, s.fam, s.enc, s.dec, s.kind = name, wl, fam, enc, dec, kind


def build():
    out = []
    for wl in ("card", "cred"):
        for k, (e, d) in {"GCM": (pbe.enc_gcm, pbe.dec_gcm), "CBC": (pbe.enc_cbc, pbe.dec_cbc), "CTR": (pbe.enc_ctr, pbe.dec_ctr)}.items():
            out.append(S(f"PBE-{k}", wl, "PBE", e, d, k))
    cards = {"HE-naive": honey.CardCodec(BINS, [1] * 6, False), "HE-mismatch": honey.CardCodec(BINS, [1] * 6, True),
             "HE-matched": honey.CardCodec(BINS, REAL_W, True)}
    creds = {"HE-naive": honey.CredCodec(USERS, VOCAB, naive=True), "HE-mismatch": honey.CredCodec(USERS, VOCAB, [1] * len(VOCAB)),
             "HE-matched": honey.CredCodec(USERS, VOCAB, VW)}
    for wl, cs in (("card", cards), ("cred", creds)):
        for n, c in cs.items():
            h = honey.HoneyEnc(c); out.append(S(n, wl, "HE", h.enc, h.dec, "HE"))
    return out


def trial_setup(sch):
    return random.choice(SALTS), random.choices(range(D), ZW)[0], real[sch.wl]()


def train_clf(sch, n=3000):
    X, y, rows = [], [], []
    for _ in range(n):
        salt, r, msg = trial_setup(sch); blob = sch.enc(DICT[r], msg, salt)
        w = random.choice([i for i in random.sample(range(D), 3) if i != r])
        for lab, o in ((1, sch.dec(DICT[r], blob)), (0, sch.dec(DICT[w], blob))):
            f = feats(sch.wl, o); X.append(f); y.append(lab); rows.append((sch.name, sch.wl, lab, show(o), *f))
    X, y = np.array(X, float), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=1).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    art = reveng.artefacts(FN[sch.wl], X[y == 1], X[y == 0])
    return clf, {"auc": round(roc_auc_score(yte, p), 3), "acc": round(accuracy_score(yte, p > .5), 3), "artefacts": art}, rows


KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
curve = lambda pos: [round(float(np.mean([p < k for p in pos])), 3) for k in KS]


def attack(sch, clf, ds_writer):
    pos1, pos2, floor, ncand, samples = [], [], [], [], None
    for t in range(A.trials):
        salt, r, msg = trial_setup(sch); blob = sch.enc(DICT[r], msg, salt)
        outs = [sch.dec(g, blob) for g in DICT]
        ok = [i for i, o in enumerate(outs) if o is not None and valid[sch.wl](o)]
        assert r in ok
        pos1.append(sum(i < r for i in ok)); floor.append(r); ncand.append(len(ok))
        if clf is not None:
            X = np.array([feats(sch.wl, outs[i]) for i in ok], float)
            p = np.clip(clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
            sc = -np.log(np.array(ok) + 1.0) + np.log(p / (1 - p))
            pos2.append(int((sc > sc[ok.index(r)]).sum()))
        if t < 10:
            for g, o in enumerate(outs):
                ds_writer.writerow([sch.name, sch.wl, t, g, DICT[g], int(g == r), int(g in ok), "REJECTED" if o is None else show(o)])
        if t == 0:
            samples = {"true_rank": r, "true_pw": DICT[r], "rows": [
                {"rank": g, "pw": DICT[g], "out": "REJECTED" if outs[g] is None else show(outs[g]), "pass": g in ok, "correct": g == r}
                for g in sorted(set(list(range(8)) + [r]))]}
    res = {"name": sch.name, "wl": sch.wl, "fam": sch.fam, "avg_candidates": round(float(np.mean(ncand)), 1),
           "reduction_pct": round(100 * (1 - np.mean(ncand) / D), 1), "floor": curve(floor), "filter": curve(pos1), "samples": samples}
    res["ml"] = curve(pos2) if clf is not None else res["filter"]
    return res


def overhead(sch):
    salt, r, msg = random.choice(SALTS), 0, real[sch.wl](); blob = sch.enc(DICT[r], msg, salt)
    n = 3000; t = time.perf_counter()
    for _ in range(n): sch.enc(DICT[r], msg, salt)
    te = (time.perf_counter() - t) / n * 1e6; t = time.perf_counter()
    for _ in range(n): sch.dec(DICT[1], blob)
    td = (time.perf_counter() - t) / n * 1e6
    return {"enc_us": round(te, 1), "dec_us": round(td, 1), "blob_bytes": len(blob), "msg_bytes": len(msg)}


def kdf_cost():
    import hashlib
    out = {}
    for it in (2000, 100000, 600000):
        t = time.perf_counter()
        for _ in range(5): hashlib.pbkdf2_hmac("sha256", b"password", b"s" * 16, it, 32)
        out[it] = round((time.perf_counter() - t) / 5 * 1000, 2)
    return out


def main():
    t0 = time.time(); schemes = build(); results, clf_rows = [], []
    f = open(f"{OUT}/dataset_attack.csv", "w", newline=""); w = csv.writer(f)
    w.writerow(["scheme", "workload", "trial", "guess_rank", "guess_pw", "is_correct", "passes_filter", "output"])
    for sch in schemes:
        clf, cm = None, None
        if sch.fam == "HE":
            clf, cm, rows = train_clf(sch); clf_rows += rows
        r = attack(sch, clf, w); r["clf"] = cm; r["overhead"] = overhead(sch)
        # reverse engineering of ciphertext blobs
        blobs = []
        for _ in range(1200):
            salt, rk, msg = trial_setup(sch); blobs.append(sch.enc(DICT[rk], msg, salt))
        kind = sch.kind
        r["blob"] = {"hexdump": reveng.hexdump(blobs[0]), "layout": reveng.annotate(kind, blobs[0]),
                     "entropy": round(reveng.byte_entropy(blobs[0]), 2), **reveng.corpus_tests(blobs, kind)}
        results.append(r); print(f"{sch.wl:5} {sch.name:12} cands={r['avg_candidates']:7} f@10={r['floor'][3]} filt@10={r['filter'][3]} ml@10={r['ml'][3]} auc={cm and cm['auc']}  [{time.time()-t0:.0f}s]", flush=True)
    f.close()
    with open(f"{OUT}/dataset_classifier.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scheme", "workload", "is_real", "output", "features..."]); w.writerows(clf_rows)
    json.dump({"config": {"trials": A.trials, "dict_size": D, "pbkdf2_iters_sim": common.ITERS, "zipf_s": 1.0, "ks": KS},
               "kdf_ms": kdf_cost(), "schemes": results}, open(f"{OUT}/results.json", "w"))


main()

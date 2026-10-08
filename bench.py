"""Timing micro-benchmark (separate from the attack so KDF caching cannot distort it).
Reports median and IQR per call after warm-up. KDF cost is measured on its own; encrypt/decrypt exclude it (key cached).
CAVEAT: PBE runs in OpenSSL (C) via `cryptography`; HE's DTE is pure Python. Ratios reflect implementation language, not algorithmic cost."""
import hashlib, json, os, random, sys, time
import numpy as np
import common, pbe, groundtruth as G, features as FT, hemodels as HM, analysis as AN

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/loop"
random.seed(1); pbe.set_rng(random.Random(1).randbytes)


def bench(fn, reps=200, batches=40, warm=500):
    for _ in range(warm): fn()
    xs = []
    for _ in range(batches):
        t = time.perf_counter()
        for _ in range(reps): fn()
        xs.append((time.perf_counter() - t) / reps * 1e6)
    q = np.percentile(xs, [25, 50, 75]); return {"median_us": round(float(q[1]), 2), "q1_us": round(float(q[0]), 2), "q3_us": round(float(q[2]), 2)}


res = {"kdf_ms": {}, "schemes": {}, "attacker_analysis": {}}
for it in (1000, 2000, 100000, 600000):
    xs = []
    for _ in range(15 if it < 100000 else 5):
        t = time.perf_counter(); hashlib.pbkdf2_hmac("sha256", b"password", b"s" * 16, it, 32); xs.append((time.perf_counter() - t) * 1000)
    q = np.percentile(xs, [25, 50, 75]); res["kdf_ms"][str(it)] = {"median": round(float(q[1]), 3), "q1": round(float(q[0]), 3), "q3": round(float(q[2]), 3)}
salt, pw = b"s" * 16, "sunshine123"; rng = random.Random(3)
for wl in ("card", "cred"):
    msg = G.GEN[wl](rng).encode(); res["schemes"][wl] = {}
    for sch in HM.pbe_schemes() + [HM.make(HM.spec_v1(wl))]:
        blob = sch.enc(pw, msg, salt); sch.dec(pw, blob)
        res["schemes"][wl][sch.name] = {"enc": bench(lambda: sch.enc(pw, msg, salt)), "dec": bench(lambda: sch.dec(pw, blob)),
                                        "blob_bytes": len(blob), "msg_bytes": len(msg)}
    outs = [G.GEN[wl](rng) for _ in range(2000)]
    res["attacker_analysis"][wl] = {"feature_extract": bench(lambda: FT.extract(wl, outs[7]), reps=100, batches=30), "l1_rule": bench(lambda: FT.l1_ok(wl, outs[7]), reps=100, batches=30)}
    X = np.array([FT.extract(wl, s) for s in outs]); y = np.r_[np.ones(1000), np.zeros(1000)]
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(X, y)
    t = time.perf_counter(); [clf.predict_proba(X[:500]) for _ in range(5)]; res["attacker_analysis"][wl]["gb_us_per_row"] = round((time.perf_counter() - t) / 2500 * 1e6, 2)
res["notes"] = ["encrypt/decrypt exclude KDF (key cached); KDF is reported separately.", "PBE = OpenSSL (C); HE DTE = pure Python: compare shape, not ratio.",
                "attacker analysis cost is per guess for HE outputs only; PBE needs only the L1 rule."]
json.dump(res, open(f"{OUT}/bench.json", "w"), indent=1); print(json.dumps(res["kdf_ms"]))

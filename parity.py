"""Cross-implementation test vectors: the browser simulator must reproduce these Python results exactly."""
import json, random, sys
import numpy as np
import groundtruth as G, features as FT, hemodels as HM, analysis as AN
out = sys.argv[1] if len(sys.argv) > 1 else "results/loop"
mod = json.load(open(f"{out}/models.json")); rng = random.Random(99); vec = {}
for wl in ("card", "cred"):
    vec[wl] = {}
    for ver in ("v1", "v2"):
        spec = mod[wl][ver]; h = HM.make(spec); c = h.__self__.c if hasattr(h, "__self__") else None
        import honey
        codec = honey.CardCodec(spec["bins"], spec["bin_w"], spec["luhn"]) if wl == "card" else honey.CredCodec(spec["users"], spec["vocab"], spec["pw_w"])
        seeds = [rng.getrandbits(64) for _ in range(300)]
        dec = [codec.from_ij(*codec.dte.decode(s)) for s in seeds]
        nb = mod[wl]["nb"][ver]; sample = G.corpus(wl, 120, rng) + dec[:120] + ["zzz:" + "x" * 7, "12345", "alice:Hello#1", "4111111111111112"]
        X = np.array([FT.extract(wl, s) for s in sample], float)
        vec[wl][ver] = {"seeds": [str(s) for s in seeds], "decoded": dec, "sample": sample,
                        "l1": [bool(FT.l1_ok(wl, s)) for s in sample], "nb": [float(v) for v in AN.nb_score(wl, nb, X)]}
json.dump(vec, open(f"{out}/parity_vectors.json", "w"))
print("parity vectors written")

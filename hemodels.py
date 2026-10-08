"""HE model specs (JSON-serialisable) and scheme wrappers. HE-v1 is the ORIGINAL implementation (honey.py) with a crude
first-attempt decoy model; HE-v2 is produced by improve.py from analysis findings. The encryption layer is identical."""
import copy
import honey, pbe
import groundtruth as G


class Scheme:
    def __init__(self, name, fam, enc, dec):
        self.name, self.fam, self.enc, self.dec = name, fam, enc, dec


def spec_v1(wl):
    if wl == "card":   # random digits after a known BIN, no Luhn, BINs equally likely
        return {"version": "v1", "workload": "card", "bins": list(G.BINS), "bin_w": [1.0] * len(G.BINS), "luhn": False}
    return {"version": "v1", "workload": "cred", "users": list(G.USERS), "vocab": list(G.CRED_VOCAB),
            "pw_w": [1.0] * len(G.CRED_VOCAB)}   # uniform over the public password list


def make(spec):
    if spec["workload"] == "card":
        c = honey.CardCodec(spec["bins"], spec["bin_w"], spec["luhn"])
    else:
        c = honey.CredCodec(spec["users"], spec["vocab"], spec["pw_w"])
    h = honey.HoneyEnc(c)
    return Scheme("HE-" + spec["version"], "HE", h.enc, h.dec)


def pbe_schemes():
    return [Scheme(f"PBE-{k}", "PBE", getattr(pbe, f"enc_{k.lower()}"), getattr(pbe, f"dec_{k.lower()}")) for k in ("GCM", "CBC", "CTR")]

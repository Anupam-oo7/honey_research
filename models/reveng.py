"""Reverse-engineering helpers: inspect ciphertext blobs and decrypted outputs the way an attacker would."""
import math
from collections import Counter

LAYOUT = {
    "GCM": [("salt", 16), ("nonce", 12), ("ciphertext", None), ("auth tag", 16)],
    "CBC": [("salt", 16), ("iv", 16), ("ciphertext (PKCS7 blocks)", None)],
    "CTR": [("salt", 16), ("nonce", 16), ("ciphertext", None)],
    "HE":  [("salt", 16), ("masked seed", 8)],
}


def byte_entropy(b):
    c = Counter(b); n = len(b)
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def hexdump(b, w=16):
    return [f"{i:04x}  " + b[i:i + w].hex(" ") for i in range(0, len(b), w)]


def annotate(kind, blob):
    out, pos = [], 0
    for name, ln in LAYOUT[kind]:
        if ln is None:
            ln = len(blob) - pos - sum(l or 0 for n, l in LAYOUT[kind][LAYOUT[kind].index((name, ln)) + 1:])
        out.append({"field": name, "offset": pos, "length": ln})
        pos += ln
    return out


def corpus_tests(blobs, kind):
    """Static tests on many ciphertexts: length leakage, body uniformity (chi2/df ~ 1 = random)."""
    lens = Counter(len(b) for b in blobs)
    hdr = {"GCM": 28, "CBC": 32, "CTR": 32, "HE": 16}[kind]
    body = b"".join(b[hdr:] for b in blobs)
    cnt = Counter(body); e = len(body) / 256
    chi = sum((cnt.get(i, 0) - e) ** 2 / e for i in range(256)) / 255
    return {"distinct_lengths": len(lens), "length_values": sorted(lens)[:6],
            "body_chi2_per_df": round(chi, 2), "body_entropy_bits_per_byte": round(byte_entropy(body), 3)}


def artefacts(names, Xreal, Xdecoy, top=4):
    """Rank output features by standardised mean difference between real and decoy plaintexts."""
    import numpy as np
    r, d = np.asarray(Xreal, float), np.asarray(Xdecoy, float)
    rows = []
    for k, n in enumerate(names):
        sd = math.sqrt((r[:, k].var() + d[:, k].var()) / 2) + 1e-9
        rows.append({"feature": n, "real_mean": round(float(r[:, k].mean()), 3),
                     "decoy_mean": round(float(d[:, k].mean()), 3), "effect": round(abs(r[:, k].mean() - d[:, k].mean()) / sd, 2)})
    return sorted(rows, key=lambda x: -x["effect"])[:top]

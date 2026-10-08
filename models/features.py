"""Feature extraction. L1 = hard structural predicates (rules a parser could apply).
L2 = distributional features compared statistically. L3 classifier uses both."""
import math, re
from collections import Counter
from models.common import luhn_ok
from models import groundtruth as G

CRED_RE = re.compile(r"^[a-z]{3,10}:[\x21-\x7e]{1,16}$")
CARD_RE = re.compile(r"^[0-9]{16}$")
SPEC = {
    "card": {"L1": ["len16", "all_digits", "luhn_valid", "bin_known"],
             "L2": {"bin_id": "cat", "first_digit": "cat", "digit_entropy": "num", "distinct_digits": "num",
                    "max_digit_count": "num", "digit_mean": "num"}},
    "cred": {"L1": ["fmt_ok", "user_known", "pw_parses"],
             "L2": {"has_upper": "cat", "word_bucket": "cat", "word_rank": "num", "suffix_id": "cat", "sym_suffix": "cat",
                    "cap_x_sym": "cat", "pw_len": "num", "digit_frac": "num", "symbol_frac": "num", "entropy": "num",
                    "user_id": "cat"}}}
NB_FEATS = {"card": ["luhn_valid", "bin_id"], "cred": ["has_upper", "suffix_id", "word_bucket", "cap_x_sym"]}


def names(wl):
    return SPEC[wl]["L1"] + list(SPEC[wl]["L2"])


def _ent(seq):
    c = Counter(seq); n = len(seq)
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def bucket(rank):
    return -1 if rank < 0 else 0 if rank < 3 else 1 if rank < 10 else 2 if rank < 25 else 3


def feat_dict(wl, s):
    if wl == "card":
        digs = [int(c) for c in s if "0" <= c <= "9"]
        c = Counter(digs)
        return {"len16": float(len(s) == 16), "all_digits": float(bool(CARD_RE.match(s)) or (s.isascii() and s.isdigit())),
                "luhn_valid": float(s.isascii() and luhn_ok(s)), "bin_known": float(s[:6] in G.BIN_IDX),
                "bin_id": G.BIN_IDX.get(s[:6], -1), "first_digit": int(s[0]) if s[:1].isascii() and s[:1].isdigit() else -1,
                "digit_entropy": _ent(digs), "distinct_digits": len(c), "max_digit_count": max(c.values()) if c else 0,
                "digit_mean": sum(digs) / len(digs) if digs else 0.0}
    u, _, p = s.partition(":")
    pr = G.parse_pw(p); n = max(len(p), 1)
    cp, wr, si = pr if pr else (-1, -1, -1)
    sym = float(si >= 0 and G.SUF2[si] in G.SYM)
    return {"fmt_ok": float(bool(CRED_RE.match(s))), "user_known": float(u in G.USERS), "pw_parses": float(pr is not None),
            "has_upper": cp, "word_bucket": bucket(wr), "word_rank": wr, "suffix_id": si, "sym_suffix": sym,
            "cap_x_sym": float(cp == 1 and sym == 1), "pw_len": len(p), "digit_frac": sum(ch.isdigit() for ch in p) / n,
            "symbol_frac": sum(not ch.isalnum() for ch in p) / n, "entropy": _ent(p),
            "user_id": G.USERS.index(u) if u in G.USERS else -1}


def extract(wl, s):
    d = feat_dict(wl, s)
    return [float(d[k]) for k in names(wl)]


def l1_ok(wl, s):
    """Attacker's structural rule: accept only if every L1 predicate holds."""
    if wl == "card":
        return bool(CARD_RE.match(s)) and luhn_ok(s) and s[:6] in G.BIN_IDX
    u, _, p = s.partition(":")
    return bool(CRED_RE.match(s)) and u in G.USERS and G.parse_pw(p) is not None

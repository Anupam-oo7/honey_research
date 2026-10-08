"""Synthetic, controlled 'real user' data generators + public lists.
EVERYTHING here is synthetic: the distributions below are assumptions we chose, not measurements of real users."""
from models import common
from models.common import luhn_check

BINS = ["411111", "424242", "453201", "510510", "555555", "601100"]
BIN_W = [40, 25, 12, 10, 8, 5]                     # synthetic issuer mix (real cards)
BIN_IDX = {b: i for i, b in enumerate(BINS)}
USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
         "karan", "leela", "mohan", "nisha", "omar", "priya", "quinn", "rahul", "sana", "tarun"]
WORDS = list(common.BASE)                           # 56 base words, listed in assumed popularity order
WORD_IDX = {w: i for i, w in enumerate(WORDS)}
SUF2 = ["", "1", "12", "123", "1234", "2024", "99", "007", "!", "@123", "#1", "*"]
SYM = {"!", "@123", "#1", "*"}
P_CAP = 0.30                                        # synthetic: 30% of users capitalise the first letter
WORD_W = [1.0 / (i + 1) for i in range(len(WORDS))]  # Zipf(1) over words
# synthetic dependency: capitalised passwords tend to carry symbol suffixes, lower-case ones digit suffixes
SUF_W = {0: [6, 8, 3, 6, 3, 3, 2, 1, 1, .5, .3, .2], 1: [2, 2, 1, 3, 1, 2, 1, .5, 8, 6, 3, 2]}


def cap(w):
    return w[:1].upper() + w[1:]


CRED_VOCAB = [(cap(w) if c else w) + s for c in (0, 1) for w in WORDS for s in SUF2]  # public password list
assert len(set(CRED_VOCAB)) == len(CRED_VOCAB)


def real_card(rng):
    s = rng.choices(BINS, BIN_W)[0] + f"{rng.randrange(10 ** 9):09d}"
    return s + luhn_check(s)


def real_cred(rng):
    c = int(rng.random() < P_CAP)
    w = rng.choices(WORDS, WORD_W)[0]
    s = rng.choices(SUF2, SUF_W[c])[0]
    return f"{rng.choice(USERS)}:{cap(w) if c else w}{s}"


GEN = {"card": real_card, "cred": real_cred}


def corpus(wl, n, rng):
    return [GEN[wl](rng) for _ in range(n)]


def parse_pw(pw):
    """Split a password into (capitalised?, word_rank, suffix_id) using the public word/suffix lists; None if impossible."""
    best = None
    for si, s in enumerate(SUF2):
        if s and not pw.endswith(s):
            continue
        stem = pw[:len(pw) - len(s)] if s else pw
        w = stem.lower()
        if w in WORD_IDX and stem in (w, cap(w)):
            cand = (int(stem != w), WORD_IDX[w], si)
            if best is None or len(s) > len(SUF2[best[2]]):
                best = cand
    return best

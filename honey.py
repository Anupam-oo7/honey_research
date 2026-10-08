"""Honey Encryption (Juels-Ristenpart style) with a two-level Distribution-Transforming Encoder.
Ciphertext = salt(16) | (seed XOR PBKDF2(pw,salt)[:8]); any password decrypts to *some* seed -> decode -> plausible message."""
import bisect
import random
from common import kdf, luhn_check

TOTAL = 1 << 64


class TwoLevelDTE:
    """Seed space [0,2^64) is split into category intervals (proportional to `weights`);
    inside category i the interval is split uniformly into `sub` sub-messages j."""

    def __init__(self, weights, sub):
        weights = [max(1, int(round(w * 10 ** 9))) for w in weights]
        tot = sum(weights)
        acc, self.starts = 0, []
        for w in weights:
            self.starts.append(TOTAL * acc // tot)
            acc += w
        self.starts.append(TOTAL)
        self.sub = sub
        assert min(b - a for a, b in zip(self.starts, self.starts[1:])) >= sub

    def encode(self, i, j):  # randomised inverse sampling
        a, size = self.starts[i], self.starts[i + 1] - self.starts[i]
        lo = a + -(-j * size // self.sub)
        hi = a + -(-(j + 1) * size // self.sub) - 1
        return random.randint(lo, hi)

    def decode(self, seed):
        i = bisect.bisect_right(self.starts, seed) - 1
        size = self.starts[i + 1] - self.starts[i]
        return i, (seed - self.starts[i]) * self.sub // size


class CardCodec:
    """16-digit card: 6-digit BIN | 9-digit account | Luhn check. luhn=False is the naive 'random digits' encoder."""

    def __init__(self, bins, weights, luhn):
        self.bins, self.luhn = bins, luhn
        self.dte = TwoLevelDTE(weights, 10 ** 9 if luhn else 10 ** 10)

    def to_ij(self, m):
        return self.bins.index(m[:6]), int(m[6:15]) if self.luhn else int(m[6:16])

    def from_ij(self, i, j):
        if self.luhn:
            s = self.bins[i] + f"{j:09d}"
            return s + luhn_check(s)
        return self.bins[i] + f"{j:010d}"


CH = "abcdefghijklmnopqrstuvwxyz0123456789!@#*"
N8 = sum(40 ** k for k in range(1, 9))


def s2i(s):
    v = 0
    for c in s:
        v = v * 40 + CH.index(c)
    return sum(40 ** k for k in range(1, len(s))) + v


def i2s(n):
    l = 1
    while n >= 40 ** l:
        n -= 40 ** l
        l += 1
    out = []
    for _ in range(l):
        n, r = divmod(n, 40)
        out.append(CH[r])
    return "".join(reversed(out))


class CredCodec:
    """'user:password'. Model mode: password drawn from a vocabulary with `weights`. Naive mode: any string of length<=8."""

    def __init__(self, users, vocab, weights=None, naive=False):
        self.users, self.vocab, self.naive = users, vocab, naive
        self.idx = {w: i for i, w in enumerate(vocab)}
        self.dte = TwoLevelDTE([1], len(users) * N8) if naive else TwoLevelDTE(weights, len(users))

    def to_ij(self, m):
        u, p = m.split(":", 1)
        if self.naive:
            return 0, self.users.index(u) * N8 + s2i(p)
        return self.idx[p], self.users.index(u)

    def from_ij(self, i, j):
        if self.naive:
            return f"{self.users[j // N8]}:{i2s(j % N8)}"
        return f"{self.users[j]}:{self.vocab[i]}"


class HoneyEnc:
    def __init__(self, codec):
        self.c = codec

    def enc(self, pw, msg, salt):
        seed = self.c.dte.encode(*self.c.to_ij(msg.decode()))
        pad = int.from_bytes(kdf(pw, salt)[:8], "big")
        return salt + (seed ^ pad).to_bytes(8, "big")

    def dec(self, pw, blob):  # never fails: every wrong key yields a plausible-looking message
        seed = int.from_bytes(blob[16:24], "big") ^ int.from_bytes(kdf(pw, blob[:16])[:8], "big")
        return self.c.from_ij(*self.c.dte.decode(seed)).encode()

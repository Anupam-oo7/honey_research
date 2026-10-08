"""Shared utilities: password dictionary, cached PBKDF2, Luhn."""
import hashlib

ITERS = 2000  # PBKDF2 iterations used inside the simulation (see overhead benchmark for realistic 100k)
BASE = ("password iloveyou princess sunshine monkey dragon football baseball welcome master shadow superman "
        "letmein qwerty abc123 admin login hello freedom whatever trustno1 batman starwars michael jennifer "
        "jordan ashley charlie thomas hunter soccer harley ranger buster killer pepper summer winter cookie "
        "flower tigger secret ginger orange banana computer internet samsung manipal udupi india cricket "
        "mumbai delhi krishna ganesh").split()
SUF = ["", "1", "12", "123", "1234", "!", "@123", "2020", "2023", "2024", "2025", "007", "786", "99", "11",
       "69", "21", "*", "#1", "321", "12345", "1990", "2000", "01", "@"]


def dictionary(n=1000):
    """Attacker dictionary, ordered most- to least-popular (heuristic: base rank x suffix rank)."""
    cand = sorted(((bi + 1) * (si + 2), b + s) for bi, b in enumerate(BASE) for si, s in enumerate(SUF))
    return list(dict.fromkeys(w for _, w in cand))[:n]


def zipf(n, s=1.0):
    return [1.0 / (i + 1) ** s for i in range(n)]


_C = {}


def _as_bytes(value, name):
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"{name} must be str or bytes-like, got {type(value).__name__}")


def kdf(pw, salt, iters=None):
    iters = iters or ITERS
    pw_b = _as_bytes(pw, "password")
    salt_b = _as_bytes(salt, "salt")
    k = (pw_b, salt_b, iters)
    v = _C.get(k)
    if v is None:
        v = _C[k] = hashlib.pbkdf2_hmac("sha256", pw_b, salt_b, iters, 32)
    return v


def luhn_ok(s):
    if isinstance(s, (bytes, bytearray, memoryview)):
        s = s.decode("ascii")
    if not isinstance(s, str) or not s.isdigit():
        return False
    t = 0
    for i, c in enumerate(reversed(s)):
        d = int(c)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        t += d
    return t % 10 == 0


def luhn_check(d15):
    if isinstance(d15, (bytes, bytearray, memoryview)):
        d15 = d15.decode("ascii")
    for c in "0123456789":
        if luhn_ok(d15 + c):
            return c

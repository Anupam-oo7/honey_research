"""Conventional password-based encryption baselines (PBKDF2-HMAC-SHA256 + AES-256).
Blob layouts: GCM = salt|nonce|ct|tag ; CBC = salt|iv|ct (PKCS7) ; CTR = salt|nonce|ct (no integrity)."""
import os
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from common import kdf

_rb = os.urandom  # nonce/IV source; loop.py swaps in a seeded generator for reproducible runs


def set_rng(f):
    global _rb
    _rb = f


def enc_gcm(pw, msg, salt):
    n = _rb(12)
    return salt + n + AESGCM(kdf(pw, salt)).encrypt(n, msg, None)


def dec_gcm(pw, blob):
    try:
        return AESGCM(kdf(pw, blob[:16])).decrypt(blob[16:28], blob[28:], None)
    except InvalidTag:
        return None  # authentication failure -> perfect wrong-password oracle


def enc_cbc(pw, msg, salt):
    iv = _rb(16)
    p = padding.PKCS7(128).padder()
    e = Cipher(algorithms.AES(kdf(pw, salt)), modes.CBC(iv)).encryptor()
    return salt + iv + e.update(p.update(msg) + p.finalize()) + e.finalize()


def dec_cbc(pw, blob):
    d = Cipher(algorithms.AES(kdf(pw, blob[:16])), modes.CBC(blob[16:32])).decryptor()
    raw = d.update(blob[32:]) + d.finalize()
    u = padding.PKCS7(128).unpadder()
    try:
        return u.update(raw) + u.finalize()
    except ValueError:
        return None  # bad padding -> padding oracle (~255/256 rejection)


def enc_ctr(pw, msg, salt):
    n = _rb(16)
    e = Cipher(algorithms.AES(kdf(pw, salt)), modes.CTR(n)).encryptor()
    return salt + n + e.update(msg) + e.finalize()


def dec_ctr(pw, blob):
    d = Cipher(algorithms.AES(kdf(pw, blob[:16])), modes.CTR(blob[16:32])).decryptor()
    return d.update(blob[32:]) + d.finalize()  # always "succeeds": garbage on wrong key

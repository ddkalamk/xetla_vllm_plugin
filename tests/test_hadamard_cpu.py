"""CPU check of the plugin's Hadamard helpers against a butterfly FWHT and
ggml's parity construction. Run from xetla_vllm_plugin/: python tests/test_hadamard_cpu.py"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "xetla_vllm_plugin.py")).read()
start = src.index("# ---- Hadamard rotated basis")
end = src.index("class Timer")
ns = {"torch": torch, "os": os, "_xetla_prequant_meta": {},
      "_xetla_prequant_load_path": None}
exec(src[start:end], ns)  # noqa: S102 - pulls only the helper block, no vllm import


def fwht(x):
    x = x.clone()
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x / math.sqrt(n)


class L:
    xetla_hadamard = None
    xetla_hadamard_inverse = None


torch.manual_seed(0)
H = ns["_xetla_hadamard_matrix"](1024, "cpu")
x = torch.randn(3, 5120)
signs = torch.where(torch.rand(5120) > 0.5, 1.0, -1.0)
l = L()
l.xetla_hadamard = (signs, 1024)
y = ns["_xetla_hadamard_fwd"](l, x)
ref = fwht((x * signs).reshape(-1, 1024)).reshape(3, 5120)
print("fwd max err", (y - ref).abs().max().item())
l2 = L()
l2.xetla_hadamard_inverse = (signs, 1024)
print("inv(fwd(x)) err", (ns["_xetla_hadamard_inv"](l2, y) - x).abs().max().item())
print("H symmetric", torch.equal(H, H.t()), "H@H=I", (H @ H - torch.eye(1024)).abs().max().item())
row, col = np.meshgrid(np.arange(1024), np.arange(1024), indexing="ij")
p = row & col
for s in (16, 8, 4, 2, 1):
    p ^= p >> s
ref_H = np.where(p & 1, -1, 1) / 32.0
print("matches ggml parity matrix", np.abs(H.numpy() - ref_H).max())
assert (y - ref).abs().max() < 1e-4 and np.abs(H.numpy() - ref_H).max() == 0
print("OK")

#!/usr/bin/env python3
"""Diagnose the FLA chunked gated-delta-rule triton kernel on Intel XPU.

The Qwen3.5 / Bonsai-27B prefill path calls
``chunk_gated_delta_rule`` -> ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
which crashes the Intel triton backend for some autotune configs. This script
runs the op once per config so we can see which (BV, num_warps, num_stages)
combinations actually compile.
"""
from __future__ import annotations

import argparse
import itertools
import traceback

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=64)
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=48)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--all-configs", action="store_true",
                   help="Try every autotune config individually.")
    p.add_argument("--no-l2norm-in-kernel", action="store_true",
                   help="Apply the q/k L2 norm outside the triton kernel "
                        "(what newer vLLM does on XPU).")
    p.add_argument("--check-numerics", action="store_true",
                   help="Compare the triton output against a naive torch "
                        "gated-delta-rule reference.")
    return p.parse_args()


def naive_gated_delta_rule(q, k, v, g, beta, state, scale):
    """Sequential reference for chunk_gated_delta_rule (single sequence).

    q, k: [T, H, Dk]   v: [T, H, Dv]   g, beta: [T, H]
    state: [H, Dv, Dk]
    """
    t = q.shape[0]
    s = state.float().clone()
    outs = []
    for i in range(t):
        s = s * torch.exp(g[i].float())[:, None, None]
        ki = k[i].float()
        vi = v[i].float()
        sk = torch.einsum("hvk,hk->hv", s, ki)
        s = s + beta[i].float()[:, None] .unsqueeze(-1) * torch.einsum(
            "hv,hk->hvk", vi - sk, ki)
        outs.append(torch.einsum("hvk,hk->hv", s, q[i].float() * scale))
    return torch.stack(outs), s


def make_inputs(a):
    dev = "xpu"
    dt = torch.float16
    t = a.seqlen
    q = torch.randn(1, t, a.num_k_heads, a.head_k_dim, device=dev, dtype=dt)
    k = torch.randn(1, t, a.num_k_heads, a.head_k_dim, device=dev, dtype=dt)
    v = torch.randn(1, t, a.num_v_heads, a.head_v_dim, device=dev, dtype=dt)
    g = torch.rand(1, t, a.num_v_heads, device=dev, dtype=torch.float32).neg()
    beta = torch.rand(1, t, a.num_v_heads, device=dev, dtype=dt)
    state = torch.zeros(1, a.num_v_heads, a.head_v_dim, a.head_k_dim,
                        device=dev, dtype=torch.float32)
    cu = torch.tensor([0, t], device=dev, dtype=torch.int32)
    return q, k, v, g, beta, state, cu


def main():
    a = parse_args()
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops import chunk_delta_h

    kern = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    # triton wraps the jit fn in Heuristics(Autotuner(JITFunction)); walk down
    # to the object that owns the autotune configs.
    tuner = kern
    while not hasattr(tuner, "configs") and hasattr(tuner, "fn"):
        tuner = tuner.fn
    if not hasattr(tuner, "configs"):
        raise SystemExit("could not locate the triton Autotuner")
    all_cfgs = list(tuner.configs)
    print(f"[diag] {len(all_cfgs)} autotune configs on "
          f"{torch.xpu.get_device_properties(0).name}")

    q, k, v, g, beta, state, cu = make_inputs(a)

    def run():
        qq, kk = q, k
        if a.no_l2norm_in_kernel:
            qq = torch.nn.functional.normalize(q.float(), dim=-1, p=2).to(q.dtype)
            kk = torch.nn.functional.normalize(k.float(), dim=-1, p=2).to(k.dtype)
        return chunk_gated_delta_rule(
            q=qq, k=kk, v=v, g=g, beta=beta,
            initial_state=state, output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=not a.no_l2norm_in_kernel,
        )

    if not a.all_configs:
        try:
            o, s = run()
            print(f"[diag] OK with default autotune: out={tuple(o.shape)}")
            if a.check_numerics:
                import torch.nn.functional as F
                qn = F.normalize(q[0].float(), dim=-1, p=2)
                kn = F.normalize(k[0].float(), dim=-1, p=2)
                if a.no_l2norm_in_kernel:
                    qn = F.normalize(
                        F.normalize(q[0].float(), dim=-1, p=2), dim=-1, p=2)
                # GQA: k has fewer heads than v/q in this layout, expand
                rep = v.shape[2] // k.shape[2]
                kn = kn.repeat_interleave(rep, dim=1)
                qn = qn.repeat_interleave(rep, dim=1)
                ref_o, ref_s = naive_gated_delta_rule(
                    qn, kn, v[0].float(), g[0], beta[0].float(),
                    state[0], a.head_k_dim ** -0.5)
                got = o[0].float()
                err = (got - ref_o).abs().max().item()
                rel = ((got - ref_o).abs().mean()
                       / ref_o.abs().mean().clamp_min(1e-6)).item()
                serr = (s[0].float() - ref_s).abs().max().item()
                print(f"[diag] numerics: max_abs_err={err:.4g} "
                      f"mean_rel={rel:.4g} state_err={serr:.4g}")
                print("[diag] numerics " +
                      ("OK" if rel < 5e-2 else "MISMATCH"))
        except Exception:
            traceback.print_exc()
            print("[diag] FAILED with default autotune")
        return

    good, bad = [], []
    for cfg in all_cfgs:
        tuner.configs = [cfg]
        if hasattr(tuner, "cache"):
            tuner.cache.clear()
        desc = (f"BV={cfg.kwargs.get('BV')} warps={cfg.num_warps} "
                f"stages={cfg.num_stages}")
        try:
            run()
            torch.xpu.synchronize()
            good.append(desc)
            print(f"[diag] OK    {desc}", flush=True)
        except Exception as e:  # noqa: BLE001
            bad.append((desc, str(e).splitlines()[-1] if str(e) else type(e).__name__))
            print(f"[diag] FAIL  {desc}", flush=True)
    print(f"\n[diag] {len(good)} ok / {len(bad)} failed")
    for d in good:
        print(f"[diag]   ok  : {d}")
    for d, e in bad:
        print(f"[diag]   fail: {d}  ({e})")


if __name__ == "__main__":
    main()

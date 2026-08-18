"""Does an int8 lm_head change which token comes out?

    python scripts/check_lm_head_int8.py --model <export dir>

Speed is settled (torch._int_mm is 1.84x fp16 at M=1). The open question is
accuracy: lm_head produces logits, and only the ordering matters, so raw
relative error is the wrong metric. This reports top-1 agreement, top-5
overlap and the logit gap at the decision boundary, on the real weight matrix.

W8A8 (what _int_mm needs) is compared against weight-only int8, which would
need a custom kernel but leaves activations untouched.
"""
from __future__ import annotations

import argparse

import torch
from safetensors import safe_open


def load_lm_head(model_dir):
    import glob, os, json
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for f in files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if k.endswith("lm_head.weight"):
                    return fh.get_tensor(k)
    # tied embeddings: fall back to the input embedding
    for f in files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if k.endswith("embed_tokens.weight"):
                    return fh.get_tensor(k)
    raise SystemExit("no lm_head or embed_tokens found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rows", type=int, default=512, help="synthetic hidden states")
    p.add_argument("--hidden-rms", type=float, default=1.0,
                   help="rms of the hidden state entering lm_head")
    a = p.parse_args()

    w = load_lm_head(a.model).to(torch.float32)   # [vocab, hidden]
    V, H = w.shape
    print(f"lm_head: vocab {V}, hidden {H}, dtype on disk fp16")

    torch.manual_seed(0)
    x = torch.randn(a.rows, H) * a.hidden_rms

    ref = x @ w.T                                  # fp32 reference
    ref16 = (x.half() @ w.half().T).float()        # what we ship today

    # weight-only int8, per output row (per vocab entry)
    ws = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    w8 = torch.round(w / ws).clamp(-127, 127)
    wq_only = (x @ (w8 * ws).T)

    # W8A8: activations also quantised, per row (per token)
    xs = (x.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    x8 = torch.round(x / xs).clamp(-127, 127)
    w8a8 = (x8 @ w8.T) * xs * ws.T

    def report(name, out):
        t1_ref = ref.argmax(-1)
        t1 = out.argmax(-1)
        top1 = (t1 == t1_ref).float().mean().item()
        k = 5
        r5 = ref.topk(k, -1).indices
        o5 = out.topk(k, -1).indices
        overlap = sum(len(set(r.tolist()) & set(o.tolist())) for r, o in zip(r5, o5))
        overlap /= (k * len(r5))
        # margin between top1 and top2 in the reference, in units of the error
        gap = (ref.topk(2, -1).values[:, 0] - ref.topk(2, -1).values[:, 1])
        err = (out - ref).abs().max(-1).values
        print(f"{name:<22}{100*top1:>9.2f}%{100*overlap:>11.2f}%"
              f"{err.mean().item():>12.4f}{gap.mean().item():>12.4f}")

    print(f"{'variant':<22}{'top-1 match':>10}{'top-5 overlap':>12}"
          f"{'mean|err|':>12}{'mean gap':>12}")
    print("-" * 68)
    report("fp16 (shipped today)", ref16)
    report("int8 weight-only", wq_only)
    report("int8 W8A8 (_int_mm)", w8a8)


if __name__ == "__main__":
    main()

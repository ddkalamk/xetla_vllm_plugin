#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Combine N GIFs side-by-side into a single looping GIF.

Usage:
    python scripts/combine_gifs.py INPUT.gif [INPUT.gif ...] OUT.gif \
        [--label "fp16 baseline" --label "int2 (xetla)" ...]

Legacy 2-input form is still supported via --label-left / --label-right.
Inputs may have different durations and frame counts; shorter ones are held
on their last frame until the longest finishes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    img = Image.open(path)
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        while True:
            frames.append(img.copy().convert("RGBA"))
            durations.append(img.info.get("duration", 80))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames, durations


def cumulative_ms(durations: list[int]) -> list[int]:
    out = []
    s = 0
    for d in durations:
        s += d
        out.append(s)
    return out


def frame_at(frames: list[Image.Image], cum: list[int], t_ms: int) -> Image.Image:
    if t_ms >= cum[-1]:
        return frames[-1]
    for i, c in enumerate(cum):
        if t_ms < c:
            return frames[i]
    return frames[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="INPUT.gif [INPUT.gif ...] OUT.gif (last is output).")
    ap.add_argument("--label", action="append", default=[],
                    help="Label for the next panel (repeatable). "
                         "Use --label '' to skip a panel.")
    ap.add_argument("--label-left", default=None,
                    help="(legacy) label for the first panel.")
    ap.add_argument("--label-right", default=None,
                    help="(legacy) label for the second panel.")
    ap.add_argument("--gap", type=int, default=8, help="pixels between panels")
    ap.add_argument("--bg", default="#1e1e1e")
    ap.add_argument("--label-h", type=int, default=0,
                    help="height in pixels of the label bar (0 = auto, ~28).")
    ap.add_argument("--fps", type=int, default=20,
                    help="output frame rate (frames per second).")
    args = ap.parse_args()

    if len(args.paths) < 2:
        ap.error("need at least one input GIF and an output path")
    inputs = [Path(p) for p in args.paths[:-1]]
    out_path = Path(args.paths[-1])

    labels = list(args.label)
    if not labels and (args.label_left or args.label_right):
        labels = [args.label_left or "", args.label_right or ""]
    while len(labels) < len(inputs):
        labels.append("")

    panels = [load_frames(p) for p in inputs]
    cums = [cumulative_ms(d) for _, d in panels]
    total_ms = max(c[-1] for c in cums)

    sizes = [p[0][0].size for p in panels]
    H = max(h for _, h in sizes)
    has_label = any(l for l in labels)
    label_h = args.label_h or (28 if has_label else 0)

    out_w = sum(w for w, _ in sizes) + args.gap * (len(inputs) - 1)
    out_h = H + label_h

    # Resample at uniform fps to avoid an explosion of frames.
    step_ms = int(round(1000 / args.fps))
    n = max(1, total_ms // step_ms + 1)

    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
               "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(fp, 16)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    out_frames: list[Image.Image] = []
    durations: list[int] = []
    for i in range(n):
        t = i * step_ms
        canvas = Image.new("RGB", (out_w, out_h), args.bg)
        x = 0
        for (frames, _), cum, (w, _), label in zip(panels, cums, sizes, labels):
            f = frame_at(frames, cum, t)
            canvas.paste(f.convert("RGB"), (x, label_h))
            if label_h and label:
                d = ImageDraw.Draw(canvas)
                tw = d.textlength(label, font=font)
                d.text((x + (w - tw) / 2, 4), label, fill="#dddddd", font=font)
            x += w + args.gap
        out_frames.append(canvas)
        durations.append(step_ms)

    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {out_path}: {len(out_frames)} frames, {out_w}x{out_h}, "
          f"{total_ms/1000:.1f}s")


if __name__ == "__main__":
    main()

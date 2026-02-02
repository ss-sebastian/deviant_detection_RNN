from pathlib import Path
import subprocess
import json

import torch
import matplotlib.pyplot as plt


# -----------------------
# Config: match your CLI
# -----------------------
GM_PY = "gm_stimuli.py"

n_blocks = 10000
blocks_to_show = list(range(1, 11))  # 1..10

f_min, f_max, f_step = 1000, 2000, 5
save_root = Path("./gm_viz_10blocks")   # temp folder
save_root.mkdir(parents=True, exist_ok=True)

# If you changed these in gm_stimuli.py defaults, keep consistent
seed = 42
exclude_freqs = [1455.0, 1500.0, 1600.0]


def run_gm(block_index: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", GM_PY,
        "--n_blocks", str(n_blocks),
        "--block_index", str(block_index),
        "--f_min", str(f_min), "--f_max", str(f_max), "--f_step", str(f_step),
        "--seed", str(seed),
        "--save_dir", str(out_dir),
        "--exclude_freqs", *[str(v) for v in exclude_freqs],
    ]
    subprocess.run(cmd, check=True)


def load_block(out_dir: Path):
    x = torch.load(out_dir / "input_tensor.pt", map_location="cpu")   # (1,10,8)
    y = torch.load(out_dir / "labels_tensor.pt", map_location="cpu")  # (1,10)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return x[0], y[0], meta


# -----------------------
# Generate + load blocks
# -----------------------
blocks = []
print("Generating blocks...")
for b in blocks_to_show:
    out_dir = save_root / f"block_{b:02d}"
    run_gm(b, out_dir)
    X, Y, meta = load_block(out_dir)
    blocks.append((b, X, Y, meta))

print("Done.\n")


# -----------------------
# Print summary + sanity
# -----------------------
for b, X, Y, meta in blocks:
    uniq, cnt = torch.unique(X, return_counts=True)
    uniq = [float(u.item()) for u in uniq]
    cnt = [int(c.item()) for c in cnt]
    lab_counts = {int(v.item()): int((Y == v).sum().item()) for v in torch.unique(Y)}

    print(f"=== Block {b:02d} ===")
    print("std/dev (meta):", meta.get("block_standard_hz"), meta.get("block_deviant_hz"))
    print("unique freqs:", uniq, "counts:", cnt)
    print("label counts:", lab_counts)

    # Optional: print trials for each block (comment out if too verbose)
    for t in range(10):
        dp = int(Y[t].item())
        trial = X[t]
        print(f"  Trial {t+1:02d} dev_pos={dp} freqs={trial.tolist()}")
    print("")


# -----------------------
# Visualize: 10 heatmaps
# -----------------------
fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
axes = axes.flatten()

# use a common color scale across blocks so you can compare visually
global_min = min(float(X.min()) for _, X, _, _ in blocks)
global_max = max(float(X.max()) for _, X, _, _ in blocks)

for ax, (b, X, Y, meta) in zip(axes, blocks):
    im = ax.imshow(X.numpy(), aspect="auto", vmin=global_min, vmax=global_max)
    ax.set_title(f"Block {b}\nstd={meta.get('block_standard_hz'):.0f}, dev={meta.get('block_deviant_hz'):.0f}")
    ax.set_xticks(range(8))
    ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([str(i) for i in range(1, 11)], fontsize=8)

    # mark deviant positions
    for t in range(10):
        j = int(Y[t].item()) - 1
        ax.scatter(j, t, s=70, facecolors="none", edgecolors="white", linewidths=1.6)

# shared colorbar
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.9)
cbar.set_label("Frequency (Hz)")

fig.suptitle("10 Blocks: trial × position heatmaps (white circles mark deviant position)", y=1.02)
plt.show()

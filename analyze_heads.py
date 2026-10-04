import argparse
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from toy_model import ToyTransformer
from create_dataset import generate_test_sequence, vocabulary, itos
from plot_results import plot_scores, plot_ablation

CONDITIONS = {
    "standard": "prefix_length",
    "ooc_pos": "out_of_context_pos",
    "ooc_no_pos": "out_of_context_no_pos",
}
CHANCE = 1.0 / len(vocabulary)

def run_model(model, x, ablate=frozenset(), return_attn=False):
    B, T = x.shape
    h = model.embed(x) + model.pos_embed[:, :T, :]
    patterns = []

    for li, layer in enumerate(model.transformer.layers):
        mha = layer.self_attn
        H, D = mha.num_heads, mha.embed_dim
        dh = D // H

        qkv = F.linear(h, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)

        pattern = (q @ k.transpose(-2, -1) / math.sqrt(dh)).softmax(dim=-1)
        z = pattern @ v 

        for hd in range(H):
            if (li, hd) in ablate:
                z[:, hd] = 0.0

        z = z.transpose(1, 2).reshape(B, T, D)
        h = layer.norm1(h + mha.out_proj(z))
        h = layer.norm2(h + layer.linear2(layer.activation(layer.linear1(h))))
        patterns.append(pattern)

    logits = model.fc(h[:, -1])
    return (logits, patterns) if return_attn else logits

def make_samples(condition, n):
    test_type = CONDITIONS[condition]
    out = []
    for _ in range(n):
        x, y, prefix_len, *_ = generate_test_sequence(test_type)
        out.append((x, y, prefix_len))
    return out

def bucket_by_length(samples):
    buckets = {}
    for x, y, p in samples:
        buckets.setdefault(len(x), []).append((x, y, p))
    for T, items in buckets.items():
        xs = torch.stack([i[0] for i in items])
        ys = torch.stack([i[1] for i in items])
        ps = torch.tensor([i[2] for i in items])
        yield T, xs, ys, ps

@torch.no_grad()
def head_scores(model, samples, device):
    L = len(model.transformer.layers)
    H = model.transformer.layers[0].self_attn.num_heads
    target = np.zeros((L, H))
    prev = np.zeros((L, H))
    n_total = 0

    for T, xs, ys, ps in bucket_by_length(samples):
        xs, ps = xs.to(device), ps.to(device)
        _, patterns = run_model(model, xs, return_attn=True)
        B = xs.shape[0]
        for li, pat in enumerate(patterns):
            last = pat[:, :, T - 1, :]
            idx = ps.view(B, 1, 1).expand(B, H, 1) 
            target[li] += last.gather(2, idx).squeeze(-1).sum(0).cpu().numpy()
            prev[li] += torch.diagonal(pat, offset=-1, dim1=-2, dim2=-1).mean(-1).sum(0).cpu().numpy()
        n_total += B

    return target / n_total, prev / n_total

@torch.no_grad()
def accuracy(model, samples, device, ablate=frozenset()):
    correct, total = 0, 0
    for T, xs, ys, ps in bucket_by_length(samples):
        logits = run_model(model, xs.to(device), ablate=ablate)
        correct += (logits.argmax(-1).cpu() == ys).sum().item()
        total += len(ys)
    return correct / total


def ablation_configs(L, H):
    configs = [("baseline", frozenset())]
    for li in range(L):
        for hd in range(H):
            configs.append((f"L{li}H{hd}", frozenset({(li, hd)})))
    for li in range(L):
        configs.append((f"all L{li}", frozenset((li, hd) for hd in range(H))))
    return configs

def plot_patterns(model, device, out_dir, target_scores):
    x, y, p, *_ = generate_test_sequence("prefix_length", prefix_length=3)
    with torch.no_grad():
        _, patterns = run_model(model, x.unsqueeze(0).to(device), return_attn=True)
    T = len(x)
    labels = [f"{i}:{itos[t.item()]}" for i, t in enumerate(x)]
    L, H = len(patterns), patterns[0].shape[1]

    fig, axes = plt.subplots(L, H, figsize=(3.4 * H, 3.6 * L), squeeze=False)
    for li in range(L):
        for hd in range(H):
            ax = axes[li][hd]
            ax.imshow(patterns[li][0, hd].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"L{li}H{hd}  (target score {target_scores[li, hd]:.2f})", fontsize=9)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            ax.get_xticklabels()[p].set_color("red")
            ax.get_yticklabels()[T - 1].set_color("red")
            ax.set_xlabel("key", fontsize=7)
            ax.set_ylabel("query", fontsize=7)
    fig.suptitle(
        f"Attention patterns (example input, target B at index {p} in red; "
        "readout = last query, in red). Score = mean over many samples.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "attention_patterns.png"), dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n", type=int, default=2000, help="samples per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    model = ToyTransformer(len(vocabulary)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    L = len(model.transformer.layers)
    H = model.transformer.layers[0].self_attn.num_heads

    data = {c: make_samples(c, args.n) for c in CONDITIONS}

    target, prev = head_scores(model, data["standard"], device)
    print("\nTarget score (final position -> B), rows = layers:\n", np.round(target, 2))
    print("Previous-token score, rows = layers:\n", np.round(prev, 2))
    plot_patterns(model, device, args.out, target)
    plot_scores(target, prev, args.out)

    table, names = {}, []
    for name, ablate in ablation_configs(L, H):
        names.append(name)
        table[name] = {}
        for cond, samples in data.items():
            acc = accuracy(model, samples, device, ablate)
            se = math.sqrt(acc * (1 - acc) / len(samples))
            table[name][cond] = (acc, se)

    print(f"\n{'config':<10}" + "".join(f"{c:>20}" for c in CONDITIONS))
    for name in names:
        print(f"{name:<10}" + "".join(f"{table[name][c][0]:>13.3f} ±{table[name][c][1]:.3f}" for c in CONDITIONS))

    with open(os.path.join(args.out, "ablation_results.csv"), "w") as f:
        f.write("config," + ",".join(f"{c}_acc,{c}_se" for c in CONDITIONS) + "\n")
        for name in names:
            f.write(name + "," + ",".join(f"{table[name][c][0]:.4f},{table[name][c][1]:.4f}" for c in CONDITIONS) + "\n")
    plot_ablation(names, table, args.out)
    print(f"\nSaved figures and CSV to {args.out}/")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import random
import numpy as np
from toy_model import ToyTransformer
from create_dataset import get_training_or_val_batch, generate_training_sequence, generate_test_sequence, vocabulary
from plot_results import plot_training_curve, plot_accuracy_by_prefix_length, plot_gap_length, plot_out_of_context, plot_summary

### CONFIGS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPOCHS = 200
SAMPLES = 200
BATCH_SIZE = 64
EVAL_SAMPLES = 1000
TEST_SAMPLES = 500
GAP_LENGTHS = [1, 2, 3, 5, 8, 13, 20]
PREFIX_LENGTHS = [2, 3, 4, 5]
CHECKPOINT = "checkpoints/checkpoint.pth"
RESULT_FILE = "result.txt"
TRIALS = 10

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def write(f, s):
    print(s)
    f.write(s + "\n")

def run_eval(model, n=EVAL_SAMPLES):
    model.eval()
    correct = 0
    by_length = {}

    with torch.no_grad():
        for _ in range(n):
            x, y, pattern_length = generate_training_sequence()
            x = x.unsqueeze(0).to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x).argmax(dim=-1)
            hit = (pred.item() == y.item())
            correct += hit
            if pattern_length not in by_length.keys():
                by_length[pattern_length] = [hit]
            else:
                by_length[pattern_length].append(hit)

    acc = correct / n
    acc_by_length = {l: sum(v) / len(v) for l, v in sorted(by_length.items())}
    return acc, acc_by_length

def mean_std(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1))

def aggregate_runs(runs):
    keys = runs[0].keys()
    aggregated = {}
    for key in keys:
        mean, std = mean_std([run[key] for run in runs])
        aggregated[key] = (mean, std)
    return aggregated

def run_experiments_once(model):
    model.eval()
    acc_by_gap = {}
    with torch.no_grad():
        for gap in GAP_LENGTHS:
            hits = []
            for _ in range(TEST_SAMPLES):
                x, y, *_ = generate_test_sequence("gap_length", gap_length=gap)
                x    = x.unsqueeze(0).to(DEVICE)
                y    = y.to(DEVICE)
                pred = model(x).argmax(dim=-1)
                hits.append(pred.item() == y.item())
            acc_by_gap[gap] = sum(hits) / len(hits)


    acc_by_prefix = {}
    with torch.no_grad():
        for pl in PREFIX_LENGTHS:
            hits = []
            for _ in range(TEST_SAMPLES):
                x, y, *_ = generate_test_sequence("prefix_length", prefix_length=pl)
                x = x.unsqueeze(0).to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x).argmax(dim=-1)
                hits.append(pred.item() == y.item())
            acc_by_prefix[pl] = sum(hits) / len(hits)

    hits_std, hits_pos, hits_no_pos = [], [], []
    with torch.no_grad():
        for _ in range(TEST_SAMPLES):
            x, y, *_ = generate_test_sequence("prefix_length")
            x = x.unsqueeze(0).to(DEVICE)
            y = y.to(DEVICE)
            hits_std.append(model(x).argmax(dim=-1).item() == y.item())

            x, y, *_ = generate_test_sequence("out_of_context_pos")
            x = x.unsqueeze(0).to(DEVICE)
            y = y.to(DEVICE)
            hits_pos.append(model(x).argmax(dim=-1).item() == y.item())

            x, y, *_ = generate_test_sequence("out_of_context_no_pos")
            x = x.unsqueeze(0).to(DEVICE)
            y = y.to(DEVICE)
            hits_no_pos.append(model(x).argmax(dim=-1).item() == y.item())

    return {
        "acc_by_prefix": acc_by_prefix,
        "acc_by_gap": acc_by_gap,
        "acc_standard": sum(hits_std) / len(hits_std),
        "acc_ooc_pos": sum(hits_pos) / len(hits_pos),
        "acc_ooc_no_pos": sum(hits_no_pos) / len(hits_no_pos),
    }

def run_experiments(model, trials=TRIALS):
    runs = [run_experiments_once(model) for _ in range(trials)]

    acc_by_prefix = aggregate_runs([run["acc_by_prefix"] for run in runs])
    acc_by_gap = aggregate_runs([run["acc_by_gap"] for run in runs])
    acc_standard = mean_std([run["acc_standard"] for run in runs])
    acc_ooc_pos = mean_std([run["acc_ooc_pos"] for run in runs])
    acc_ooc_no_pos = mean_std([run["acc_ooc_no_pos"] for run in runs])

    return acc_by_prefix, acc_by_gap, acc_standard, acc_ooc_pos, acc_ooc_no_pos

def main():
    model = ToyTransformer(len(vocabulary)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # use_checkpoint = True

    losses = []
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        write(f, "═" * 50)
        write(f, "TRAINING")
        write(f, "═" * 50)

        best_loss = np.inf

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for _ in range(SAMPLES):
                x, y, _ = get_training_or_val_batch(BATCH_SIZE)
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / SAMPLES
            losses.append(avg_loss)

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), CHECKPOINT)

            write(f, f"epoch {epoch}  loss {avg_loss}")

        write(f, f"\nbest training loss: {best_loss}")

        write(f, "\n" + "═" * 50)
        write(f, "EVALUATION")
        write(f, "═" * 50)
        acc, acc_by_length = run_eval(model)
        write(f, f"overall accuracy : {acc}")
        write(f, "accuracy by prefix length:")
        for l, a in acc_by_length.items():
            write(f, f"len {l}: {a}")

        write(f, "\n" + "═" * 50)
        write(f, "TEST SUITE")
        write(f, "═" * 50)

        write(f, "\n── gap_length ──")
        write(f, "Hypothesis: induction don't care bout distance: accuracy flat across gaps")

        acc_by_prefix_test, acc_by_gap, acc_std, acc_ooc_pos, acc_ooc_no_pos = run_experiments(model, TRIALS)

        for g, a in acc_by_gap.items():
            mean, std = a
            write(f, f"gap {g}: {mean} ± {std}")

        write(f, "\n── prefix_length ──")
        write(f, "Hypothesis: induction care bout prefix length; frequency bias stays flat")
        for l, a in acc_by_prefix_test.items():
            mean, std = a
            write(f, f"prefix {l}: {mean} ± {std}")

        write(f, "\n── out_of_context ──")
        write(f, "standard >> chance; ooc_pos splits strategies; ooc_no_pos → chance")
        std_mean, std_std = acc_std
        pos_mean, pos_std = acc_ooc_pos
        nopos_mean, nopos_std = acc_ooc_no_pos
        write(f, f"standard: {std_mean} ± {std_std}")
        write(
            f,
            f"ooc_pos: {pos_mean} ± {pos_std}  "
            f"(potential pos heu: {pos_mean - nopos_mean})"
        )
        write(
            f,
            f"ooc_no_pos: {nopos_mean} ± {nopos_std}  "
            f"(potential ind con: {std_mean - pos_mean})"
        )

    plot_training_curve(losses)
    plot_accuracy_by_prefix_length(acc_by_length)
    plot_accuracy_by_prefix_length(acc_by_prefix_test, save_path="test_by_prefix_length.png")
    plot_gap_length(acc_by_gap)
    plot_out_of_context(acc_std, acc_ooc_pos, acc_ooc_no_pos)
    plot_summary(losses, acc_by_length, acc_by_gap, acc_std, acc_ooc_pos, acc_ooc_no_pos)

    print("Done")

if __name__ == "__main__":
    main()

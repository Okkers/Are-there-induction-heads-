import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

sequence_lengths = [x for x in range(2, 6)]
filler_lengths = [x for x in range(1, 10)]
vocabulary = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
stoi = {ch: i for i, ch in enumerate(vocabulary)}
itos = {i: ch for ch, i in stoi.items()}

def generate_prefix(length):
    return list(np.random.choice(vocabulary, size=length))

def generate_gap(length, prefix):
    while True:
        gap = list(np.random.choice(vocabulary, size=length))
        if not any(gap[i:i+len(prefix)] == prefix for i in range(len(gap) - len(prefix) + 1)):
            return gap

def to_tensor(seq, target):
    x = torch.tensor([stoi[c] for c in seq])
    y = torch.tensor(stoi[target])
    return x, y

def add_filler(seq):
    filler_len = random.choice(filler_lengths)
    filler = list(np.random.choice(vocabulary, size=filler_len))
    return seq + filler

def generate_training_sequence():
    pattern_len = random.choices(sequence_lengths, weights=[0.4, 0.3, 0.2, 0.1])[0]
    prefix = generate_prefix(pattern_len)
    target = random.choice(vocabulary)
    seq = add_filler(prefix + [target]) + prefix
    x, y = to_tensor(seq, target)
    return x, y, pattern_len

def generate_test_sequence(test_type, **kwargs):
    if test_type == "gap_length":
        prefix_len = random.choices(sequence_lengths, weights=[0.4, 0.3, 0.2, 0.1])[0]
        prefix = generate_prefix(prefix_len)
        target = random.choice(vocabulary)
        gap_len = kwargs.get("gap_length", random.randint(1, 10))
        gap = generate_gap(gap_len, prefix)
        seq = prefix + [target] + gap + prefix
        x, y = to_tensor(seq, target)
        return x, y, prefix_len, gap_len

    elif test_type == "prefix_length":
        prefix_len = kwargs.get("prefix_length", random.choices(sequence_lengths, weights=[0.4, 0.3, 0.2, 0.1])[0])
        prefix = generate_prefix(prefix_len)
        target = random.choice(vocabulary)
        gap = generate_gap(random.randint(1, 3), prefix)
        seq = prefix + [target] + gap + prefix
        x, y = to_tensor(seq, target)
        return x, y, prefix_len

    elif test_type == "out_of_context_pos":
        prefix_len = random.choices(sequence_lengths, weights=[0.4, 0.3, 0.2, 0.1])[0]
        prefix_a = generate_prefix(prefix_len)
        target = random.choice(vocabulary)
        prefix_b = generate_prefix(prefix_len)
        while prefix_b == prefix_a:
            prefix_b = generate_prefix(prefix_len)
        gap = generate_gap(random.randint(1, 3), prefix_b)
        seq = prefix_a + [target] + gap + prefix_b
        x, y = to_tensor(seq, target)
        return x, y, prefix_len
    
    elif test_type == "out_of_context_no_pos":
        prefix_len = random.choices(sequence_lengths, weights=[0.4, 0.3, 0.2, 0.1])[0]
        prefix_a = generate_prefix(prefix_len)
        target = random.choice(vocabulary)
        
        remaining_vocab = [v for v in vocabulary if v not in prefix_a]
        prefix_b = [random.choice(remaining_vocab) for _ in range(prefix_len)]
        
        decoy_len = random.randint(2, 6)
        decoy = list(np.random.choice(vocabulary, size=decoy_len))
        
        gap = generate_gap(random.randint(1, 3), prefix_b)
        seq = decoy + prefix_a + [target] + gap + prefix_b
        x, y = to_tensor(seq, target)
        return x, y, prefix_len

    else:
        return

def pad_batch(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    metadata = [item[2:] for item in batch]
    max_len = max(len(x) for x in xs)
    padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.stack(ys), metadata

def get_training_or_val_batch(batch_size):
    return pad_batch([generate_training_sequence() for _ in range(batch_size)])

def get_test_batch(batch_size, test_type, **kwargs):
    return pad_batch([generate_test_sequence(test_type, **kwargs) for _ in range(batch_size)])
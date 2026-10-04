# Are there induction heads in smaller transformers?

A controlled synthetic dataset for behaviorally testing whether a small transformer learns an induction-style mechanism ("if `A → B` appeared earlier and `A` appears again, predict `B`"), and whether it also relies on a positional shortcut.

**Write-up:** [Are there induction heads in smaller transformers?](TODO: link to post/PDF)

## TL;DR

A 2-layer, 4-head transformer trained on a prefix-retrieval task reaches ~0.91 test accuracy. Targeted test conditions suggest it uses **two strategies in parallel**:

| Condition | Accuracy | What it tells us |
|---|---|---|
| Standard (matching prefix) | 0.91 | Task is learned |
| OOC, same position (no matching prefix) | 0.48 | Large drop without a matchable prefix, but still far above chance → positional shortcut |
| OOC, offset position (no matching prefix) | 0.11 | Falls to chance (0.10) once position is shifted → no residual strategy |

Accuracy also degrades only slightly as the prefix gets longer, and is flat in gap length up to the longest gaps seen in training (collapse beyond that, consistent with a training-distribution boundary).

**Scope:** this is a *behavioral* test. It does not inspect attention patterns or ablate individual heads, so it shows behavior consistent with induction rather than identifying induction heads directly (see [Limitations](#limitations)).

## Task and dataset

Training and validation samples have the form

```
x = A + B + filler + A,    y = B
```

- `A`: random prefix of 2–5 letters
- `B`: a single random letter (the target)
- `filler`: random letters, length 1–10, to prevent trivial matching
- Vocabulary: small (10 tokens; chance accuracy = 0.10) <!-- TODO: confirm vocab size -->

Test sets modify the distribution to separate hypotheses:

| Test | Change | Question |
|---|---|---|
| Gap length | Filler extended to 1–20 | Is accuracy invariant to the distance between the two occurrences of `A`? |
| Prefix length | Prefix 2–5 | Does accuracy depend on how much must be matched? |
| Out-of-context (OOC) | `x = A + B + filler + C`, `C ≠ A` | Induction cannot solve this. What does the model do instead? |
| OOC with offset | As above, with filler prepended | Removes the fixed-position cue, isolating a positional heuristic |

## Results

<!-- TODO: save the figures from plot_results.py into a figures/ folder and commit them -->

**Out-of-context conditions** (induction vs. positional heuristic)

<img width="1034" height="515" alt="test_out_of_context" src="https://github.com/user-attachments/assets/ef1b5a03-7f66-4801-ac02-b6e1f6f565dd" />

**Prefix length**

<img width="888" height="508" alt="test_by_prefix_length" src="https://github.com/user-attachments/assets/b5dd5e67-6feb-44f5-86d4-ddccb47f8871" />

**Gap length**

![Gap length](figures/gap_length.png)

Training loss converges to ~0.25 (not 0), as expected for a partly stochastic task: see `figures/training_loss.png`.

**Attention Patterns**

<img width="2176" height="1152" alt="attention_patterns" src="https://github.com/user-attachments/assets/0ad9d738-44cd-4e3a-937b-40e635b4932e" />

## Model and training

- Encoder-only transformer, hidden dim 64, 4 heads, 2 layers
- Adam, learning rate 1e-3, cross-entropy loss
- 200 epochs, batch size 64, 128k training samples, no early stopping
- Validation: 1,000 samples. Test: 10 runs × 500 samples per test type; means and standard deviations reported

## Reproduce

Requires Python ≥ 3.10.11.

```bash
pip install -r requirements.txt
python main.py        # trains the model, runs all tests, produces the figures
```

## Repository structure

| File | Purpose |
|---|---|
| `create_dataset.py` | Modular sample generators for the training set and each test type |
| `toy_model.py` | Small transformer and training loop |
| `main.py` | Runs the full pipeline |
| `plot_results.py` | Produces the figures used in the write-up |

## Possible extensions

- Visualize layer-wise attention to look for the previous-token head → induction head composition described by Elhage et al.
- Ablate or patch individual heads and measure the effect on each test condition.
- Retrain with longer fillers and multiple seeds.

## References

- Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
- Olsson et al. (2022). [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895).

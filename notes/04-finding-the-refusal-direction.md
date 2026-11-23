# 04 — Finding the Refusal Direction

> Notes for `03_find_refusal_direction.ipynb` (+ one extra cell in `02_collect_activations.ipynb` for test activations).
> Goal: compute a candidate refusal direction for every layer, see where it emerges, and check whether it generalizes to unseen prompts.
> Runs on **CPU**. No model needed except for collecting test activations.

---

## 1. Difference of means

```python
mean_harmful  = harmful_acts.mean(dim=0)    # [128, 28, 2048] → [28, 2048]
mean_harmless = harmless_acts.mean(dim=0)
diff = mean_harmful - mean_harmless         # [28, 2048]: one candidate direction per layer
```

- `mean(dim=0)` averages over the **prompt** dimension → 128 prompts collapse into one "average prompt". That dimension disappears.
- Averaging cancels prompt-specific content ("Oakland", "weather"…). What survives is the **systematic difference** between the two groups.

---

## 2. Norm, and why relative size matters

```
‖v‖ = √(v₁² + v₂² + … + v₂₀₄₈²)      (Pythagoras in 2048 dims)
```

```python
diff_norm = diff.norm(dim=-1)                       # [28]
act_norm  = harmless_acts.norm(dim=-1).mean(dim=0)  # typical activation size per layer
relative  = diff_norm / act_norm
```

- **Absolute** diff norm grows from ~0 to ~1300, but mostly because the residual stream itself grows layer by layer (every layer adds to it). Misleading on its own.
- **Relative** difference (`diff_norm / act_norm`) is the meaningful one.

### Result (relative difference)

| Layers | Value | Interpretation |
|---|---|---|
| 0–10 | ~0.05–0.15 | only token-level info; no "harmful" judgment yet |
| 11–15 | rising, **big jump 14→15** | the concept forms |
| 16–21 | peak ~0.7 around 19–20 | clearest separation |
| 22–26 | slowly decreasing | stream shifts from "what do I think" to "what word do I say" |
| 27 | sudden spike | last layer prepares the output token: "I'm sorry" vs "Sure", a *consequence* of refusal, not the concept |

→ Abliteration usually targets **middle** layers, not the last ones.

---

## 3. Cosine similarity between layers

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖)       1 = same direction, 0 = unrelated, −1 = opposite
```

```python
dirs = diff / diff.norm(dim=-1, keepdim=True)   # unit vectors
cos_sim = dirs @ dirs.T                          # [28, 28]
```

For unit vectors, the dot product *is* the cosine similarity → one matrix multiply gives all 784 comparisons.

### Result (heatmap)
- Diagonal = 1 (each direction vs itself). Matrix is **symmetric**.
- **Layers 0–14:** thin band, each direction only resembles its immediate neighbours → the direction keeps changing, no stable concept yet.
- **Layers 15–18:** transition zone (~0.4–0.6 with later layers).
- **Layers 19–27:** a red **block** (~0.6–0.9) → the direction has **settled** and is carried forward roughly unchanged.

### Reading the two plots together
- The difference **jumps at layer 15** → the concept forms.
- The direction **stabilizes from ~19** → the concept settles.

This supports the "refusal is (roughly) a single direction" hypothesis, at least in middle–late layers. It's also why removing *one* direction from all writers can work: removing layer 19's direction also removes most of the later layers' (they're similar).

**Candidate layers for ablation: ~15–21.** The plots don't give a single answer. Following Arditi et al., we'll pick by **testing** a few candidates in stage 4.

Saved: `refusal_dirs.pt` (`diff`, `dirs`).

---

## 4. PCA

PCA finds the directions along which the data **varies the most**; we project onto the top 2 to get a 2D picture.

**PCA is unsupervised:** it does not know which points are harmful. If the groups separate anyway, the harmful/harmless difference is one of the *largest* sources of variation at that layer.

```python
X = torch.cat([harmful_acts[:, L], harmless_acts[:, L]]).numpy()   # [256, 2048]
Z = PCA(n_components=2).fit_transform(X)                            # [256, 2]
```

### Result

| Layer | What we see |
|---|---|
| 2 | already a *weak* separation (red lower, blue upper), mixed |
| 8 | mixed |
| 12 | red starts forming a tight cluster |
| 15 | two clusters + a "bridge" of a few red points |
| 19, 24 | full separation **along PC1**: the biggest variation in the data *is* harmful vs harmless |

### Observations
1. **Separation at layer 2** can't be a "judgment". It comes from **surface features**: AdvBench prompts share templates ("Write a script…", "Develop a plan…") and words like "hack", "steal". The difference-of-means vector contains more than refusal.
2. **Red is tight, blue is spread:** harmless prompts ask for very different things (poems, maths, recipes); harmful prompts, whatever their topic, **converge** to a shared "refusal mode".
3. **Bridge points:** a few harmful prompts sit between the clusters. Hypothesis: the model is uncertain about them, or may not refuse them. Testable in stage 4.

Saved: `pca_layers.png`.

---

## 5. Held-out test: does the direction generalize?

Test activations were collected in notebook 02 (extra cell), reading the **saved** test prompt lists to guarantee the exact same prompts. Saved: `activations_test.pt` (`[32, 28, 2048]` each).

### Projection
How much "refusal" is in a notebook? Dot product with the unit direction:

```
score = activation · r̂
```

Geometrically: the length of the activation's **shadow** on the refusal direction. 2048 numbers → 1 number.

```python
def project(acts, dirs):
    return (acts * dirs).sum(dim=-1)        # [N, 28, 2048] · [28, 2048] → [N, 28]

threshold = (tr_h.mean(0) + tr_b.mean(0)) / 2           # from TRAIN only
test_acc  = ((te_h > threshold).float().mean(0) +
             (te_b <= threshold).float().mean(0)) / 2   # balanced accuracy on TEST
```

- Threshold from **train**, measured on **test**. Computing the threshold on test data would leak test information.
- Balanced accuracy: 0.5 = coin flip, 1.0 = perfect.

### Result
- **Layer 0: 0.84** (!), rising to ~0.97 by layer 4.
- **Layers 13–15: 1.00.**
- **Layers 16–27: 0.984** → exactly one harmful test prompt misclassified (31/32).
- Layer 19 histogram: harmless scores ≈ −130…−10, harmful ≈ 260…490 → a huge margin. One harmful prompt sits at ~150, just below the threshold (~163): a **bridge point** again.

### ⚠️ The most important lesson of this stage: correlation ≠ causation

Even **layer 0** (pure embeddings, no processing at all) classifies at 84%. So "this direction separates harmful from harmless" does **not** prove "this direction *causes* refusal". At early layers it's just detecting words and templates.

- **Probing / projection** (what we did here) → *correlational*: is the information present?
- **Intervention / ablation** (stage 4) → *causal*: if we remove it, does the behaviour change?

A direction is the refusal direction only if **removing it makes the model stop refusing** while leaving other behaviour intact. That's what stage 4 tests.

Saved: `projection_test.png`.

---

## ➡️ Next: `04_ablation.ipynb`

1. A single `evaluate()` function → baseline refusal rate on test prompts (original model).
2. Ablate a candidate layer's direction (first with **hooks**, then by **editing the 57 writer matrices**: `W' = W − r̂ r̂ᵀ W`).
3. Compare refusal rates for candidate layers (e.g. 15, 17, 19, 21) + check harmless answers aren't broken.
4. Bonus: look at the "bridge" harmful prompt. Does the original model refuse it?

# 03 — Collecting Activations with Forward Hooks

> Notes for `02_collect_activations.ipynb`.
> Goal: run two prompt sets through the model and record the residual stream at the last prompt token, after every layer.

---

## 1. Prompt sets

| Set | Source | Total available | Used |
|---|---|---|---|
| **Harmful** | AdvBench (`harmful_behaviors.csv`, llm-attacks paper) | 520 | 128 train + 32 test |
| **Harmless** | Alpaca (`tatsu-lab/alpaca`), instructions with empty `input` | 31,323 (of 52,002) | 128 train + 32 test |

- Alpaca entries with an extra `input` field were dropped: instructions like "Summarize this text" make no sense on their own.
- `random.seed(42)` → same shuffle every run (reproducibility).
- **Train/test split:** the refusal direction is computed on *train*; its effect is evaluated on *test*, prompts the direction has never seen. Same idea as checking for overfitting in ML.
- **Equal set sizes:** both means come from the same number of samples, so one side isn't noisier than the other.
- Why only 128? We're computing a **mean**. Past a certain sample size the mean barely changes. (Possible experiment: compare the direction from 64 vs 128 prompts.)

---

## 2. Forward hooks

Normally `model(...)` only returns the final output; everything computed inside the 28 layers is thrown away.

A **forward hook** is a function attached to a module that PyTorch calls right after that module runs, passing it the module's output.

```python
captured = {}

def make_hook(layer_idx):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[layer_idx] = h.detach()
    return hook

handle = model.model.layers[10].register_forward_hook(make_hook(10))
# ... run the model ...
handle.remove()   # always remove hooks when done!
```

### Key points
- **The hook is an observer:** it copies the output and doesn't change the model. (Hooks *can* modify outputs; we'll use that for ablation.)
- **A decoder layer's output is the whole residual stream after that layer**, not just that layer's contribution:
  ```
  output of layer 10 = embedding + everything layers 0..10 added
  ```
  In the notebook analogy: a photo of the *entire* notebook right after person #10 writes their note.
- `isinstance(output, tuple)` handles a difference between transformers versions (tuple vs. plain tensor).
- Forgetting `handle.remove()` leaves an invisible hook that fires on every future call. Use `try/finally`.

---

## 3. Reading shapes

Single prompt, hook on layer 10:

```
input_ids:        [1, 16]
layer 10 output:  [1, 16, 2048]
                   │   │    └── d_model: each notebook is 2048 numbers
                   │   └─────── 16 tokens = 16 separate notebooks
                   └─────────── batch size: 1 prompt
```

We only want the **last token's** notebook, because that's where the model decides the first word of its answer (and whether to refuse):

```python
captured[10][0, -1, :]   # first prompt, last token, all 2048 numbers → [2048]
captured[10][:, -1, :]   # all prompts in the batch, last token       → [B, 2048]
```

> **Habit:** predict the shape *before* running the code. Most deep learning bugs are shape bugs. A wrong prediction means either the code or your mental model is wrong.

---

## 4. Batching and left padding

Prompts are processed in batches of 16 for speed. Prompts have different lengths, but tensors must be rectangular, so shorter prompts get **pad** tokens.

```
Right padding:                     Left padding:
[ A  B  C  D  E ]  ← -1 = E ✅      [ A  B  C  D  E ]  ← -1 = E ✅
[ X  Y  Z  _  _ ]  ← -1 = _ ❌      [ _  _  X  Y  Z ]  ← -1 = Z ✅
```

With **left padding**, every prompt's real last token sits at index `-1`. `attention_mask` tells the model to ignore the pads.

**Position ids:** left padding shifts the position numbers of real tokens. We recompute positions so they only count real tokens:

```python
position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
```

(Qwen's rotary position encoding is mostly insensitive to a constant shift, but this is the correct way.)

---

## 5. The collection function

```python
tokenizer.padding_side = "left"

def collect_last_token_acts(prompts, batch_size=16):
    """Returns [n_prompts, n_layers, d_model]: the residual stream
    at the last prompt token, after every decoder layer."""
    n_layers = len(model.model.layers)
    batch_acts = {}

    def make_hook(i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            batch_acts[i] = h[:, -1, :].detach().float().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.model.layers)]
    all_acts = []
    try:
        for start in range(0, len(prompts), batch_size):
            texts = [build_prompt(p) for p in prompts[start:start + batch_size]]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            position_ids = (inputs.attention_mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                model(**inputs, position_ids=position_ids)
            all_acts.append(torch.stack([batch_acts[i] for i in range(n_layers)], dim=1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(all_acts, dim=0)
```

- `.float()`: store in float32; averaging in float16 can lose precision.
- `.cpu()`: move to RAM so VRAM doesn't fill up.

---

## 6. Result

```
harmful:  torch.Size([128, 28, 2048])
harmless: torch.Size([128, 28, 2048])
```

Predicted before running ✅

| Dim | Meaning | Value |
|---|---|---|
| 0 | prompts | 128 |
| 1 | layers (one photo after each) | 28 |
| 2 | d_model | 2048 |

```python
harmful_acts[5, 14, :]   # 6th prompt, after layer 14 → [2048]
harmful_acts[:, 14, :]   # all prompts, after layer 14 → [128, 2048]
```

Saved to Drive as `MyDrive/abliteration/activations_train.pt` (~60 MB), together with the train/test prompt lists, so later notebooks evaluate on the **same** held-out prompts.

---

## 7. Practical lessons

- **Colab idle timeout:** after ~45 min idle the runtime was disconnected and all variables were lost (`NameError`). Code stays, memory doesn't. Fix: **Runtime → Run before** (Ctrl+F8).
- **Memory is temporary, Drive is permanent.** Save results as soon as they're computed.
- **Drive mount** fails with `credential propagation was unsuccessful` unless all requested permissions are granted. Permissions can be revoked later from Google Account → Security → Third-party connections.
- Every new runtime starts on a clean disk → the model (~4 GB) is re-downloaded each session.

---

## 8. Decision: baseline evaluation goes in notebook 04

The "refusal rate before vs. after" comparison must use **identical** evaluation code (same prompts, same `max_new_tokens`, same refusal criteria). Putting one `evaluate()` function in 04 and running it on both the original and the modified model guarantees that.

---

## ➡️ Next: `03_find_refusal_direction.ipynb`

For each of the 28 layers:
```
r_layer = mean(harmful_acts[:, layer, :]) − mean(harmless_acts[:, layer, :])
```
→ 28 candidate refusal directions. Then: which layer gives the best one, and how do we tell?

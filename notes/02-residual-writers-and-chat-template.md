# 02 — Residual Writers & the Chat Template

> Continuation of session 1. Answers the open exercise from `01-transformer-basics.md` and looks at what the model actually sees as input.

---

## 1. Readers vs. writers

Every `Linear` layer in the model either **reads** from the residual stream (takes the 2048-dim vector as input) or **writes** to it (its output is added back to the stream).

Two checks to tell them apart:
1. **Shape:** a writer's output must be 2048-dim. You can't add a 6144-dim vector to a 2048-dim one.
2. **Position in the flow:** a writer is the *last* layer of its sub-block. Shape alone is not enough: `q_proj` is also 2048 → 2048, but it's a reader.

### Attention

```
             ┌─ q_proj → Q   "what am I looking for?"
x (stream) ──┼─ k_proj → K   "what do I contain?"
             └─ v_proj → V   "what information do I carry?"

   Q · K  → attention weights (which tokens to look at, how much)
   weights × V → gathered information
   gathered information → o_proj → added to the stream
```

- `q_proj`, `k_proj`, `v_proj` → **readers** (used internally)
- `o_proj` ("output projection") → **writer**

### MLP (SwiGLU)

```
x (2048) ─┬─ gate_proj → 6144 → SiLU ─┐
          └─ up_proj   → 6144 ────────┴─ multiply → 6144 → down_proj → 2048 → added to the stream
```

The MLP **expands** to a wider space (2048 → 6144), computes there, then **compresses** back.

- `gate_proj`, `up_proj` → **readers**
- `down_proj` → **writer**

### Full list of writers

| Source | Writer | Count |
|---|---|---|
| Initial state | `embed_tokens` | 1 |
| Attention in each layer | `o_proj` | 28 |
| MLP in each layer | `down_proj` | 28 |
| **Total** | | **57** |

**These 57 matrices are exactly what abliteration modifies.** Everything else (q, k, v, gate, up, norms) only reads from the stream. If the stream never contains the refusal direction, the readers have nothing to react to.

`lm_head` is the final reader: it reads the last stream vector and produces next-token scores. We don't touch it.

---

## 2. What the model actually sees

The model doesn't "chat". It only continues text. The **chat template** turns a conversation into one long string:

```python
messages = [{"role": "user", "content": "Who are you?"}]
tokenizer.apply_chat_template(messages, tokenize=False,
                              add_generation_prompt=True, enable_thinking=False)
```

Output:

```
'<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

| Piece | Meaning |
|---|---|
| `<|im_start|>user` | user message begins |
| `Who are you?` | the message |
| `<|im_end|>` | message ends |
| `<|im_start|>assistant` | "your turn" — generation starts after this |
| `<think>\n\n</think>` | an **empty** thinking block |

### Two takeaways

1. **"Chat" is just formatting.** The template ends with the assistant tag, so the most natural continuation is the assistant's reply.
2. **`enable_thinking=False` doesn't switch anything off.** It *pre-fills* an empty `<think></think>` block. The model sees that thinking is already "done" and goes straight to the answer. We control the model by controlling where the text continues from.

### Why this matters for the project

We will read activations at the **last token of the prompt** (right after `</think>\n\n`). That's the moment just before the model picks the first word of its answer, where the "should I refuse?" decision is made.

### First output

```
>>> chat("Who are you?")
I am an AI assistant developed by Alibaba Group. I can help with various tasks
such as answering questions, providing information, writing text, and more.
I am here to assist you in any way I can.
```

`do_sample=False` → greedy decoding → same output every run. Important for reproducible experiments.

---

## 3. Project structure decision

One notebook per stage instead of one giant notebook:

```
qwen3-abliteration/
├── 01_setup_and_anatomy.ipynb
├── 02_collect_activations.ipynb
├── 03_find_refusal_direction.ipynb
├── 04_ablation.ipynb
├── 05_steering_and_eval.ipynb     (v2)
├── notes/
└── README.md
```

Why:
- Colab sessions reset. Each stage saves its outputs (activations, direction vector) to Drive; the next stage loads them instead of recomputing.
- Easier for a reader to jump to the part they care about.
- Easier to debug one stage in isolation.

Cost: the model-loading cell is repeated at the top of each notebook. Acceptable.

---

## 4. Releasing the model (decision deferred)

- Qwen3-1.7B is **Apache 2.0**: modification and redistribution are allowed, as long as the license is included and changes are stated.
- If released: model card should state the base model, the method, and that safety training was removed; consider Hugging Face's gated access.
- Releasing weights is optional. The portfolio value is in the notebooks, plots and write-up.

---

## ➡️ Next session

`02_collect_activations.ipynb`: use **forward hooks** to read the residual stream at every layer, at the last prompt token, for two prompt sets.

# 01 — Transformer Basics & Model Anatomy

> Notes from the first session of the abliteration project.
> Goal of this session: understand what is inside a decoder-only LLM before we start poking at it.

---

## 1. What does an LLM actually do?

A language model does exactly one thing: **predict the next token.**

- Text is split into **tokens** (roughly word pieces). `"Who are you?"` → a handful of token ids.
- Each token id is turned into a **vector of numbers** by an embedding table.
- The model processes these vectors and, at the end, outputs a score for every token in the vocabulary. The highest-scoring one is (roughly) the next token.
- Generation = repeat this one token at a time, appending each new token to the input.

---

## 2. The residual stream

Every token carries a vector (for Qwen3-1.7B: **2048 numbers**). As it moves through the model, this vector is **updated by addition**, never overwritten:

```
x ← x + F(x)
```

This is the same residual connection from ResNets: `H(x) = F(x) + x`.

The vector, as it flows through all the layers, is called the **residual stream**.

### Analogy: a notebook per token

- The embedding writes the **first note** in each token's notebook.
- 28 people (layers) take turns: each one **reads** the notebook and **appends** its own note at the bottom.
- Nobody erases what was written before.
- At the end, `lm_head` reads the final notebook and says "the next word is …".

### Unrolled

```
x₁ = x₀ + Attn₁(x₀)
x₂ = x₁ + MLP₁(x₁)
x₃ = x₂ + Attn₂(x₂)
...
x_final = x₀ + Attn₁(…) + MLP₁(…) + Attn₂(…) + …
```

So the final vector is literally a **sum** of the embedding and every block's output. This additive structure is what makes abliteration possible (see §6).

---

## 3. The decoder layer

One "person" in the analogy = one **decoder layer**. Each layer has two sub-blocks, and both add to the stream:

```
x = x + Attention(x)   # tokens look at each other
x = x + MLP(x)         # each token is processed on its own
```

| Sub-block | What it does |
|---|---|
| **Attention** | A token gathers information from *other* tokens (e.g. figuring out what "it" refers to). |
| **MLP** | Per-token computation; thought to store much of the model's "knowledge". |

**Detail — pre-norm:** in Qwen (and most modern LLMs) the input is normalized *before* each sub-block, but the *un-normalized* stream is what gets added to:

```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

That is what `input_layernorm` and `post_attention_layernorm` are.

"Decoder" is just the name for the transformer variant used by text-generating models (GPT, Qwen, Llama…).

---

## 4. Anatomy of Qwen3-1.7B (`print(model)`)

```
Qwen3ForCausalLM
├── model
│   ├── embed_tokens: Embedding(151936, 2048)     ← writes x₀
│   ├── layers: 28 × Qwen3DecoderLayer
│   │   ├── self_attn
│   │   │   ├── q_proj: 2048 → 2048
│   │   │   ├── k_proj: 2048 → 1024
│   │   │   ├── v_proj: 2048 → 1024
│   │   │   ├── o_proj: 2048 → 2048
│   │   │   ├── q_norm, k_norm (RMSNorm, 128)
│   │   ├── mlp
│   │   │   ├── gate_proj: 2048 → 6144
│   │   │   ├── up_proj:   2048 → 6144
│   │   │   ├── down_proj: 6144 → 2048
│   │   │   └── act_fn: SiLU
│   │   ├── input_layernorm (RMSNorm, 2048)
│   │   └── post_attention_layernorm (RMSNorm, 2048)
│   ├── norm (final RMSNorm, 2048)
│   └── rotary_emb (positional encoding)
└── lm_head: Linear(2048 → 151936)              ← reads the final stream
```

### Key numbers

| Name | Value | Meaning |
|---|---|---|
| `vocab_size` | 151,936 | number of distinct tokens |
| `hidden_size` / `d_model` | **2048** | width of the residual stream |
| `num_layers` | 28 | decoder layers |
| MLP inner size | 6144 | 3× expansion inside the MLP |
| head dim | 128 | size of each attention head |

### Side notes
- **Grouped Query Attention (GQA):** `k_proj`/`v_proj` output 1024 instead of 2048 → 16 query heads share 8 key/value heads. Saves memory during generation.
- **SwiGLU MLP:** `gate_proj` and `up_proj` run in parallel, are combined (`SiLU(gate) * up`), and `down_proj` brings the result back to 2048.

---

## 5. Hardware & memory

### Where things live in Colab

| Resource | Size (free T4) | What's stored there |
|---|---|---|
| Disk | ~112 GB | downloaded model files |
| RAM | ~12.7 GB | Python objects, CPU-side data |
| **VRAM** (GPU) | ~15 GB | the model weights we compute with |

Loading path: `Internet → Disk → RAM → VRAM`. The GPU can only compute on data in its own memory.

### Memory math

```
parameters × bytes per parameter
1.7B × 2 bytes (float16) ≈ 3.4 GB
```

Measured with `torch.cuda.memory_allocated()`: **3.44 GB** ✅

`nvidia-smi` shows slightly more, because PyTorch/CUDA itself reserves some GPU memory.

At 4-bit quantization: `1.7B × 0.5 byte ≈ 0.85 GB` (≈1.3 GB in practice with overhead).

---

## 6. Abliteration — the idea

1. **Refusal is (roughly) one direction.** When the model decides to refuse, the residual stream gets pushed along a consistent direction `r` in the 2048-dim space (Arditi et al., 2024, *Refusal in LLMs is mediated by a single direction*).

2. **Find it with a difference of means.** Run a set of refused prompts and a set of normal prompts, collect activations at some layer, then:
   ```
   r = mean(refused) − mean(normal)
   r = r / ‖r‖
   ```

3. **Remove it from the weights.** For every matrix that *writes* into the residual stream, project out the `r` component:
   ```
   W' = W − r rᵀ W
   ```
   After this, the matrix can no longer write anything along `r`.

### Why the residual structure matters

The stream is a **sum** of contributions from a fixed set of writers. If no writer can write along `r`, the sum cannot contain `r` either. In the notebook analogy: we find the pens that write the "refuse" note and make them unable to write it.

### What abliteration is *not*
- Not fine-tuning — no training, no dataset beyond a few hundred prompts.
- Adds no knowledge — the model only stops refusing what it already knew.
- Not free — small capability loss is common.
- Not perfect — refusal is not always exactly one direction.

---

## 7. Practical notes

- Colab free tier allows very few concurrent sessions. "Too many sessions" → **Manage sessions** → terminate old ones.
- When done: **Runtime → Disconnect and delete runtime** (closing the tab isn't enough).
- Don't re-run the model-loading cell unnecessarily.
- `##` inside a code cell is a Python comment; headers go in **Text** cells.
- Disable Qwen3's thinking mode for experiments: `enable_thinking=False` in `apply_chat_template`.

---

## ❓ Open exercise

In each decoder layer, which `Linear` layers produce the output that is **added to the residual stream**?
Hint: it's the *last* layer of each sub-block, and its output must be 2048-dim. Also: one writer lives outside the layers.

*(Answer goes in the next notes file.)*

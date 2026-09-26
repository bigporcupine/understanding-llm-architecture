# Language Model Architecture: Hands-On Learning Guide

The goal is not to memorize architecture names. Instead, build this chain of reasoning:

```text
design choice → tensor transformation → problem solved → experiment that verifies it
```

The companion implementation is `transformer_tutorial.py`:

```bash
.venv/bin/python transformer_tutorial.py --lesson embedding
```

## 0. What does a language model learn?

A language model predicts the next token from the tokens it has already seen. For a tokenized
sequence such as:

```text
to be or not to be
```

the aligned training example is:

```text
input:  to be or not to
target:    be or not to be
```

The target is shifted one position to the right. At every position, the model emits one score per
vocabulary item. These scores are called logits:

```text
token_ids: [B, T]
logits:    [B, T, V]
```

Cross-entropy encourages the correct next token to receive a higher logit than all alternatives.

## 1. Embedding: turn an ID into a learned vector

A token ID is an index, not a numerical quantity. ID 7 is not “four more” than ID 3. An embedding
table is a learned matrix of shape `[V, D]`; each token ID selects one row:

```text
[B,T] --lookup--> [B,T,D]
```

```bash
.venv/bin/python transformer_tutorial.py --lesson embedding
```

Question: if the vocabulary grows from 50,000 to 200,000 while `D` stays fixed, which parameter
matrix becomes four times larger?

## 2. Attention: content-addressed information retrieval

The three projections have useful interpretations:

- `Q` (query): what am I looking for?
- `K` (key): what kind of query should select me?
- `V` (value): what information should be retrieved if I am selected?

The complete operation is:

```text
scores  = Q @ K.T / sqrt(Dh)
weights = softmax(scores + mask)
output  = weights @ V
```

```bash
.venv/bin/python transformer_tutorial.py --lesson attention
.venv/bin/python transformer_tutorial.py --lesson scaling
```

### Why use a causal mask?

Training processes the whole sequence in parallel. Without a mask, an early position could read a
later token—the answer it is meant to predict. The mask replaces future-position scores with
negative infinity, which softmax maps to zero.

### Why divide by `sqrt(Dh)`?

If each component of Q and K has variance near 1, summing `Dh` products gives dot products with
variance near `Dh` and standard deviation near `sqrt(Dh)`. Scaling keeps attention logits in a
similar range as head width changes and prevents premature softmax saturation.

## 3. Multi-head attention: multiple representation subspaces

Heads are evaluated in parallel. Implementations usually construct Q/K/V and reshape them as:

```text
[B,T,D] → [B,T,H,Dh] → [B,H,T,Dh]
```

Each head produces its own `[T,T]` attention matrix. The head outputs are then joined back into
`[B,T,D]`.

```bash
.venv/bin/python transformer_tutorial.py --lesson multihead
```

The key constraint is `D = H * Dh`, so `D` must be divisible by `H`.

## 4. RMSNorm: control activation scale

Repeated linear transformations and residual additions can cause activation scales to drift.
RMSNorm divides the final dimension by its root mean square and applies a learned scale:

```text
rms(x) = sqrt(mean(x²) + eps)
y = weight * x / rms(x)
```

```bash
.venv/bin/python transformer_tutorial.py --lesson rmsnorm
```

Unlike LayerNorm, RMSNorm does not subtract the mean. It only controls scale and is simpler to
compute.

## 5. Pre-Norm: preserve a direct residual path

The tutorial block uses:

```python
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

The residual value can move directly to the next layer. During backpropagation, a gradient path also
exists that does not pass through attention, the FFN, or normalization. This generally makes deep
models easier to optimize than placing normalization after each residual addition.

## 6. RoPE: rotate Q and K instead of adding positions

Without positional information, attention sees content but not order. RoPE treats each pair of
features as a two-dimensional plane and rotates it by an angle determined by token position.

```bash
.venv/bin/python transformer_tutorial.py --lesson rope
```

Rotation preserves vector length, but the dot product between vectors at different positions
contains their relative positional offset. This is why RoPE is applied to Q and K: position directly
affects attention scores. V is normally left unchanged.

## 7. SwiGLU: a learned content gate

A standard FFN expands the representation, applies a nonlinearity, and projects it back down.
SwiGLU adds a second branch:

```text
gate    = SiLU(x @ W_gate)
content = x @ W_up
output  = (gate * content) @ W_down
```

```bash
.venv/bin/python transformer_tutorial.py --lesson swiglu
```

The elementwise product lets the network produce candidate features and independently decide how
much of each one to pass through.

## 8. Assemble the model

```bash
.venv/bin/python transformer_tutorial.py --lesson model
```

The complete data flow is:

```text
IDs → embedding
    → [RMSNorm → causal attention → residual
       RMSNorm → SwiGLU          → residual] × N
    → RMSNorm → lm_head → logits → cross-entropy
```

The implementation also uses weight tying: `embedding.weight` and `lm_head.weight` refer to the
same parameter. This saves `V*D` independent parameters and gives the input and output sides a
shared token representation.

## 9. The most useful test: overfit a tiny corpus

```bash
.venv/bin/python transformer_tutorial.py --lesson train --steps 200
```

This is not intended to produce a useful model. It is an integration test. A correct model with
enough capacity should memorize the repeated sentence:

- loss should decrease substantially;
- gradient norms should remain finite;
- generated text should become similar to the training corpus.

If it fails, check the target shift, causal-mask direction, softmax dimension, Q/K/V reshaping,
learning rate, and whether parameters receive gradients.

## Recommended workflow

Study one or two experiments per session. For every experiment:

1. Run the original and explain every tensor shape.
2. Change one variable, predict the result, and then run it.
3. Close the reference, implement a minimal version, and compare outputs.

Suggested modifications:

1. Remove the causal mask and inspect future-position weights.
2. Remove `sqrt(Dh)` scaling and inspect the largest softmax probability at larger dimensions.
3. Change two heads to four and record which shapes change.
4. Remove RMSNorm temporarily and compare loss and gradient norms.
5. Replace SwiGLU with `Linear → GELU → Linear` and compare parameter counts.
6. Stop shifting the targets and explain why the model learns to copy its input.

After these experiments, GQA, sliding-window attention, and MLA become easier to understand. They
mainly change which queries share keys and values, or which historical positions a query can access;
the core attention operation remains the same.

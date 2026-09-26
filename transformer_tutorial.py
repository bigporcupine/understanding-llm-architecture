"""From tensors to a trainable tiny language model.

Run:
    .venv/bin/python transformer_tutorial.py --lesson attention
    .venv/bin/python transformer_tutorial.py --lesson all
    .venv/bin/python transformer_tutorial.py --lesson train --steps 200

This implementation deliberately favors readability over training speed.
"""

import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F


torch.manual_seed(42)


def title(text):
    print(f"\n{'=' * 12} {text} {'=' * 12}")


def lesson_embedding():
    title("1. Token ID -> Embedding")
    embedding = nn.Embedding(num_embeddings=8, embedding_dim=4)
    token_ids = torch.tensor([[1, 3, 1, 6]])  # [B=1, T=4]
    x = embedding(token_ids)                   # [B=1, T=4, D=4]
    print("token_ids.shape:", tuple(token_ids.shape))
    print("x.shape:        ", tuple(x.shape))
    print("Same ID 1 produces the same vector:", torch.equal(x[0, 0], x[0, 2]))
    print("Embedding parameter count:", embedding.weight.numel(), "= vocab_size * d_model")


def causal_attention(q, k, v):
    """Single-head scaled dot-product causal attention."""
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    length = q.size(-2)
    future = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=q.device), diagonal=1
    )
    scores = scores.masked_fill(future, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


def lesson_attention():
    title("2. Attention = match, then retrieve a weighted value")
    # Q=K=identity makes the matching behavior easy to inspect.
    q = torch.eye(3).unsqueeze(0)  # [B=1, T=3, Dh=3]
    k = q.clone()
    v = torch.tensor([[[10.0, 0.0], [0.0, 20.0], [30.0, 30.0]]])
    output, weights = causal_attention(q, k, v)
    print("weights.shape:", tuple(weights.shape), "= [B, T_query, T_key]")
    print("Causal weights (rows=query, columns=key):\n", weights[0].round(decimals=3))
    print("Output:\n", output[0].round(decimals=3))
    print("All future-position weights are zero:", bool(torch.all(weights[0].triu(1) == 0)))
    print("Position 0 can only retrieve itself:", output[0, 0].tolist())


def lesson_scaling():
    title("3. Why divide by sqrt(d_head)?")
    for d_head in (8, 64, 512):
        q = torch.randn(20_000, d_head)
        k = torch.randn(20_000, d_head)
        raw = (q * k).sum(dim=-1)
        scaled = raw / math.sqrt(d_head)
        print(
            f"Dh={d_head:3d} | dot-product std={raw.std():6.2f} "
            f"| scaled std={scaled.std():.2f}"
        )
    print("Without scaling, larger dimensions create extreme logits and saturated softmax values.")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def lesson_rmsnorm():
    title("4. RMSNorm: control vector scale")
    norm = RMSNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]])
    y = norm(x)
    rms_before = x.pow(2).mean(-1).sqrt()
    rms_after = y.pow(2).mean(-1).sqrt()
    print("RMS before normalization:", rms_before.tolist())
    print("RMS after normalization: ", rms_after.tolist())
    print("Same direction gives the same normalized vector:", torch.allclose(y[0], y[1]))


def apply_rope(x):
    """Rotate pairs along the final dimension. D must be even."""
    length, dim = x.size(-2), x.size(-1)
    assert dim % 2 == 0
    positions = torch.arange(length, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2, device=x.device) / dim))
    angles = positions[:, None] * inv_freq[None, :]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def lesson_rope():
    title("5. RoPE: encode position through rotations")
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]]).repeat(4, 1)  # [T=4, D=4]
    y = apply_rope(x)
    print("The same content is rotated differently at each position:\n", y.round(decimals=3))
    print("Norm before rotation:", x.norm(dim=-1).tolist())
    print("Norm after rotation: ", y.norm(dim=-1).tolist())
    print("Key point: RoPE is normally applied to Q and K, but not V.")


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def lesson_swiglu():
    title("6. SwiGLU: one content branch and one gate branch")
    layer = SwiGLU(d_model=4, d_ff=8)
    x = torch.randn(2, 3, 4)
    gate = F.silu(layer.gate(x))
    content = layer.up(x)
    y = layer(x)
    print("Input:            ", tuple(x.shape))
    print("Gate/content:     ", tuple(gate.shape), tuple(content.shape))
    print("Elementwise gate: ", tuple((gate * content).shape))
    print("Down projection:  ", tuple(y.shape))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_weights=False):
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(batch, length, self.num_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))  # [B, H, T, Dh]
        q, k = apply_rope(q), apply_rope(k)
        y, weights = causal_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch, length, d_model)
        y = self.out(y)
        return (y, weights) if return_weights else y


def lesson_multihead():
    title("7. Multi-head attention: split features into subspaces")
    attention = CausalSelfAttention(d_model=8, num_heads=2)
    x = torch.randn(1, 4, 8)
    y, weights = attention(x, return_weights=True)
    print("Input X:          ", tuple(x.shape), "[B,T,D]")
    print("Attention weights:", tuple(weights.shape), "[B,H,T,T]")
    print("Final output:     ", tuple(y.shape), "[B,T,D]")
    print("Per-head Dh = D/H =", attention.d_head)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        # Pre-Norm keeps normalization off the direct residual path.
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=2, num_heads=4, d_ff=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying shares parameters between input embeddings and output classification.
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids, targets=None):
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss


def lesson_model():
    title("8. Assemble a complete decoder-only LM")
    model = TinyLM(vocab_size=20)
    tokens = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))
    logits, loss = model(tokens, targets)
    loss.backward()
    print("tokens: ", tuple(tokens.shape), "[B,T]")
    print("logits: ", tuple(logits.shape), "[B,T,V]")
    print("loss:   ", round(loss.item(), 4))
    print("Embedding received gradients:", model.embedding.weight.grad is not None)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))


def make_char_data(text):
    chars = sorted(set(text))
    to_id = {char: index for index, char in enumerate(chars)}
    to_char = {index: char for char, index in to_id.items()}
    ids = torch.tensor([to_id[char] for char in text], dtype=torch.long)
    return ids, to_id, to_char


@torch.no_grad()
def generate(model, prompt, to_id, to_char, max_new_tokens=40):
    model.eval()
    ids = torch.tensor([[to_id[c] for c in prompt]], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)
    return "".join(to_char[i] for i in ids[0].tolist())


def lesson_train(steps=200):
    title("9. Overfit a tiny corpus: a practical correctness test")
    text = "to be or not to be, that is the question. " * 12
    data, to_id, to_char = make_char_data(text)
    seq_len, batch_size = 16, 16
    model = TinyLM(len(to_id), d_model=48, num_layers=2, num_heads=4, d_ff=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    initial_loss = None

    model.train()
    for step in range(steps):
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s : s + seq_len] for s in starts])
        # The essential shift: each target is the next token after its input.
        y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        _, loss = model(x, y)
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step == steps - 1:
            print(
                f"step {step + 1:3d} | loss {loss.item():.4f} "
                f"| grad_norm {float(grad_norm):.3f}"
            )

    print("Loss decreased substantially:", loss.item() < initial_loss * 0.5)
    print("Greedy generation:", generate(model, "to ", to_id, to_char))


LESSONS = {
    "embedding": lesson_embedding,
    "attention": lesson_attention,
    "scaling": lesson_scaling,
    "rmsnorm": lesson_rmsnorm,
    "rope": lesson_rope,
    "swiglu": lesson_swiglu,
    "multihead": lesson_multihead,
    "model": lesson_model,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", choices=[*LESSONS, "train", "all"], default="all")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.lesson == "all":
        for lesson in LESSONS.values():
            lesson()
    elif args.lesson == "train":
        lesson_train(args.steps)
    else:
        LESSONS[args.lesson]()


if __name__ == "__main__":
    main()

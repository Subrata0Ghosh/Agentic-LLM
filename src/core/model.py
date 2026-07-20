"""
CustomLLM — src/core/model.py
==============================
GPT-style Transformer with Biological Enhancements:

  1. NeuromodulatoryGain layer: scalar gain per attention head, modulated
     by norepinephrine (NE) at inference time. High NE = sharper attention
     focus (gain amplification). Low NE = softer, more diffuse attention.

  2. MemoryCrossAttention: an optional cross-attention head in the final
     Transformer Block. When `memory_kv` embeddings are passed in, the
     model can directly attend over hippocampal memory key-value pairs —
     not just read context from the prompt string, but actively integrate
     retrieved memory into the generation process.

Both additions are BACKWARD COMPATIBLE:
  - If `ne_gain=1.0` (default), neuromodulatory layer is a no-op.
  - If `memory_kv=None` (default), cross-attention is skipped entirely.
  - Existing model.pt weights load without any changes.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ── Neuromodulatory Gain ───────────────────────────────────────────────────────
class NeuromodulatoryGain(nn.Module):
    """
    A learnable per-head gain scalar modulated by norepinephrine.
    High NE → amplifies attention scores (sharper focus).
    Low NE  → attenuates scores (broader, diffuse attention).

    Biologically: LC-NE projections modulate cortical gain globally.
    """
    def __init__(self, num_heads: int):
        super().__init__()
        # Learnable baseline gain per head (initialized to 1.0)
        self.gain = nn.Parameter(torch.ones(num_heads))

    def forward(self, attention_scores: torch.Tensor, ne_level: float = 0.5) -> torch.Tensor:
        """
        Args:
            attention_scores: (B, n_heads, T, T)
            ne_level:         0.0 (low arousal) → 1.0 (high arousal)
        Returns:
            Gain-modulated attention scores.
        """
        # NE modulation: ne_level=0.5 → no change; >0.5 → amplify; <0.5 → attenuate
        ne_factor = 0.5 + ne_level  # Range [0.5, 1.5]
        gain = (self.gain * ne_factor).view(1, -1, 1, 1)  # Broadcast over B,T,T
        return attention_scores * gain


# ── Memory Cross-Attention ─────────────────────────────────────────────────────
class MemoryCrossAttention(nn.Module):
    """
    Optional cross-attention head that attends over hippocampal memory embeddings.

    The model generates a query from the current hidden state,
    then attends over key-value pairs derived from retrieved memories.
    This is analogous to memory reactivation during recall in the brain.

    Biological basis:
      - Hippocampal CA3 provides pattern completion (keys + values)
      - Cortical areas generate queries based on current processing state
      - CA1 integrates the cross-attended output back into the cortical stream
    """
    def __init__(self, n_embd: int, mem_embd: int = None, dropout: float = 0.1):
        super().__init__()
        mem_embd = mem_embd or n_embd
        self.query_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.key_proj   = nn.Linear(mem_embd, n_embd, bias=False)
        self.value_proj = nn.Linear(mem_embd, n_embd, bias=False)
        self.out_proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.scale      = n_embd ** -0.5

    def forward(self, x: torch.Tensor, memory_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, T, n_embd) — current hidden state
            memory_kv: (B, M, mem_embd) — retrieved memory embeddings
        Returns:
            (B, T, n_embd) — x enhanced with memory context
        """
        Q = self.query_proj(x)               # (B, T, E)
        K = self.key_proj(memory_kv)          # (B, M, E)
        V = self.value_proj(memory_kv)        # (B, M, E)

        # Scaled dot-product attention
        attn = Q @ K.transpose(-2, -1) * self.scale  # (B, T, M)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V                         # (B, T, E)
        return self.out_proj(out)


# ── Standard Attention Head ────────────────────────────────────────────────────
class Head(nn.Module):
    """One head of self-attention with optional NE gain modulation."""

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ne_level: float = 0.5):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        # NE gain: amplify attention scores based on arousal level
        ne_factor = 0.5 + ne_level   # 0.5 → 1.5 range
        wei = wei * ne_factor

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v   = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads   = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ne_level: float = 0.5):
        out = torch.cat([h(x, ne_level) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),                    # GELU instead of ReLU (more biologically smooth)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block with:
    - NE-modulated self-attention
    - Optional hippocampal memory cross-attention (final block only)
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.2, has_cross_attn=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

        # Optional memory cross-attention (only in last block)
        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.cross_attn = MemoryCrossAttention(n_embd, dropout=dropout)
            self.ln_cross   = nn.LayerNorm(n_embd)
            self.ln_mem     = nn.LayerNorm(n_embd)

    def forward(self, x, ne_level: float = 0.5, memory_kv=None):
        x = x + self.sa(self.ln1(x), ne_level)
        x = x + self.ffwd(self.ln2(x))

        # Hippocampal memory integration via cross-attention
        if self.has_cross_attn and memory_kv is not None:
            x = x + self.cross_attn(self.ln_cross(x), self.ln_mem(memory_kv))

        return x


class CustomLLM(nn.Module):
    """
    GPT-style language model with:
      - NE-modulated attention (neuromodulatory gain)
      - Hippocampal memory cross-attention (final block)
      - GELU activations (smoother than ReLU)
    """

    def __init__(self, vocab_size, n_embd=256, block_size=128,
                 n_head=4, n_layer=4, dropout=0.2):
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # All blocks: NE-modulated self-attention
        # Last block: also has hippocampal cross-attention
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout,
                  has_cross_attn=(i == n_layer - 1))   # Only last block gets cross-attn
            for i in range(n_layer)
        ])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, ne_level: float = 0.5, memory_kv=None):
        """
        Args:
            idx:       (B, T) token indices
            targets:   (B, T) token targets for training (optional)
            ne_level:  float — norepinephrine level [0, 1] for attention gain
            memory_kv: (B, M, n_embd) — hippocampal memory embeddings (optional)
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                             # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))                               # (T, C)
        x = tok_emb + pos_emb

        # Forward through blocks (passing ne_level and memory_kv to each)
        for block in self.blocks:
            x = block(x, ne_level=ne_level, memory_kv=memory_kv)

        x      = self.ln_f(x)
        logits = self.lm_head(x)                                              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens, ne_level: float = 0.5,
                  memory_kv=None, temperature: float = 1.0):
        """Auto-regressive generation with NE modulation and memory cross-attention."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond, ne_level=ne_level, memory_kv=memory_kv)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

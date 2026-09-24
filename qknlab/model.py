"""Small causal language model for controlled QK-normalization interventions.

The denominator-detach interventions have exactly the same numerical forward
map. Only the training backward rule changes. A true-forward JVP must therefore
always be evaluated in ``standard`` mode, regardless of the training arm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("standard", "q_detach", "k_detach", "both_detach")


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    mlp_ratio: int = 4
    norm_eps: float = 1e-5

    def __post_init__(self):
        if min(self.vocab_size, self.max_seq_len, self.d_model,
               self.n_heads, self.n_layers, self.mlp_ratio) < 1:
            raise ValueError("All model dimensions must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.norm_eps <= 0:
            raise ValueError("Model norm_eps must be positive; use scripts/verify_geometry.py for epsilon=0 theory.")

    def to_dict(self):
        return asdict(self)


def qk_rms_normalize(x: torch.Tensor, eps: float, detach: bool = False) -> torch.Tensor:
    """Parameter-free RMS normalization, evaluated in the input precision.

    Under autocast, q/k projections can be fp16. Accumulate their squared norm
    in float32 to avoid fp16 overflow and return the original activation dtype.
    Float64 diagnostic and gradcheck inputs retain float64 throughout.
    """
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    denom = (work.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    if detach:
        denom = denom.detach()
    return (work / denom).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return qk_rms_normalize(x, self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, mode: str):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.eps = config.norm_eps
        self.mode = mode
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        batch, seq, width = x.shape
        def heads(y):
            return y.reshape(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        q = qk_rms_normalize(heads(self.q_proj(x)), self.eps,
                             self.mode in ("q_detach", "both_detach"))
        k = qk_rms_normalize(heads(self.k_proj(x)), self.eps,
                             self.mode in ("k_detach", "both_detach"))
        v = heads(self.v_proj(x))
        # Explicit matmul/softmax supports torch.func.jvp and makes the mask
        # visible. Do not replace this with an opaque fused kernel in audits.
        content_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
        attention_work = content_logits.float() if content_logits.dtype in (
            torch.float16, torch.bfloat16) else content_logits
        probabilities = attention_work.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (probabilities.to(v.dtype) @ v).transpose(1, 2).contiguous().reshape(batch, seq, width)
        output = self.o_proj(y)
        return (output, content_logits) if return_attn else output


class Block(nn.Module):
    def __init__(self, config: ModelConfig, mode: str):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config, mode)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_ratio * config.d_model, bias=False),
            nn.GELU(approximate="none"),
            nn.Linear(config.mlp_ratio * config.d_model, config.d_model, bias=False),
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            update, content_logits = self.attn(self.norm1(x), return_attn=True)
        else:
            update = self.attn(self.norm1(x))
        x = x + update
        x = x + self.mlp(self.norm2(x))
        return (x, content_logits) if return_attn else x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, mode: str = "standard"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")
        self.config = config
        self.mode = mode
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config, mode) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        # No separate output Parameter: functional_call/JVP sees each tied
        # parameter exactly once and output projection uses the embedding.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")
        self.mode = mode
        for block in self.blocks:
            block.attn.mode = mode
        return self

    def forward(self, input_ids: torch.Tensor, return_attn: bool = False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input exceeds configured maximum sequence length.")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        recorded = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                recorded.append(attn)
            else:
                x = block(x)
        if return_attn == "only":
            return torch.stack(recorded, dim=0)
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        if return_attn:
            return {"logits": logits, "attn_logits": torch.stack(recorded, dim=0)}
        return logits

    def attention_logits(self, input_ids):
        """Unmasked content logits without computing the vocabulary head."""
        return self.forward(input_ids, return_attn="only")

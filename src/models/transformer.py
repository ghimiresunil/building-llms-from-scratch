from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.inference.sampling import filter_logits
from src.models.transformer_block import Block


class Transformer(nn.Module):
    """The main Transformer model."""

    pos_idxs: (
        torch.Tensor
    )  # positions 0..context_length-1, registered as a buffer in __init__

    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int,
        vocab_size: int,
        N_BLOCKS: int,
    ) -> None:
        """Initializes the Transformer model."""
        super().__init__()
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        # Opt-in activation (gradient) checkpointing; off by default so behaviour and
        # numerics are unchanged unless a caller explicitly turns it on (e.g. the
        # pretraining script's --grad-checkpointing flag to save VRAM). See issue #5.
        self.gradient_checkpointing = False
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.attn_blocks = nn.ModuleList(
            [Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)]
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer("pos_idxs", torch.arange(context_length))

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """Combines token and position embeddings."""
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return the final hidden states AFTER the final layer norm."""
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            if self.gradient_checkpointing and self.training:
                # Recompute block activations in backward instead of storing them,
                # trading compute for a large activation-memory saving. use_reentrant=False
                # is the modern, correct variant.
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.layer_norm(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the Transformer."""
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # reshape (not view): targets come from a non-contiguous slice of the
            # batch tensor, which makes .view() raise on CPU (the cross-device .to('cuda')
            # copy happens to make it contiguous, which is why this only bit CPU runs).
            flat_logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass focusing on the embedding and attention blocks."""
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks[:-1]:
            x = block(x)
        last_block = cast(Block, self.attn_blocks[-1])
        return last_block.forward_embedding(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        context_window: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
    ) -> torch.Tensor:
        """Generates new tokens given a starting sequence."""
        window = min(context_window or self.context_length, self.context_length)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -window:]
            logits, _ = self(idx_cond)
            # temperature, then top-k / top-p / min-p filtering (src/inference/sampling.py)
            logits = filter_logits(logits[:, -1, :], temperature, top_k, top_p, min_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32
    num_heads = 4
    num_blocks = 2
    context_len = 5
    input_indices = torch.randint(0, vocab_size, (batch_size, sequence_length))

    transformer_model = Transformer(
        n_head=num_heads,
        n_embed=embedding_dim,
        context_length=context_len,
        vocab_size=vocab_size,
        N_BLOCKS=num_blocks,
    )
    logits, loss = transformer_model(
        input_indices, targets=input_indices
    )  # Using input as target for simplicity

    print("Transformer Logits Shape:", logits.shape)
    print("Transformer Loss:", loss)

    # Example of generating tokens
    start_indices = input_indices[
        :, :1
    ]  # Take the first token of each sequence as start
    generated_tokens = transformer_model.generate(start_indices, max_new_tokens=5)
    print("Generated Tokens Shape:", generated_tokens.shape)

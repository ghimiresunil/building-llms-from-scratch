"""Inference helpers shared by the chat CLI and the eval scripts."""

from __future__ import annotations

import torch

from src.checkpoint import (
    load_checkpoint,
    load_model_weights,
    model_config_from_checkpoint,
    model_state_from_checkpoint,
)
from src.models.factory import LanguageModel, build_model
from src.post_training.chat_template import encode_prompt, get_tokenizer
from src.post_training.evaluation import batched_generate


def load_model_from_ckpt(
    ckpt_path: str, device: str, overrides: dict | None = None
) -> LanguageModel:
    """Build a model from the settings stored in a checkpoint and load its weights."""
    ck = load_checkpoint(ckpt_path, map_location="cpu")
    defaults = {
        "n_head": 16,
        "n_embed": 1024,
        "context_length": 1024,
        "vocab_size": 50304,
        "n_blocks": 24,
    }
    cfg = {**defaults, **model_config_from_checkpoint(ck), **(overrides or {})}
    model = build_model(cfg)
    state = model_state_from_checkpoint(ck, extra_prefixes=("transformer.",))
    load_model_weights(model, state, source=ckpt_path)
    return model.to(device).eval()


@torch.no_grad()
def generate_reply(
    model,
    user_text: str,
    *,
    device: str,
    system: str | None = None,
    raw: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    greedy: bool = False,
) -> str:
    """Generate a response to ``user_text``."""
    if raw:
        ids = get_tokenizer().encode_ordinary(user_text)
        out = batched_generate(
            model,
            [ids],
            max_new_tokens,
            device=device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        return out[0]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    prompt_ids = encode_prompt(messages)
    out = batched_generate(
        model,
        [prompt_ids],
        max_new_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
    )
    return out[0]

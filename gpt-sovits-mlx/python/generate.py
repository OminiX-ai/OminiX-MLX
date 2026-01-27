"""Generation utilities for GPT-SoVITS.

Provides functions for autoregressive semantic token generation with:
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- Repetition penalty
- Async evaluation for pipelining
"""

from typing import Optional, Iterator, List, Callable
from dataclasses import dataclass
import mlx.core as mx

from python.models.gpt import GPTSoVITS
from python.models.cache import KVCache


@dataclass
class GenerationConfig:
    """Configuration for token generation."""

    max_tokens: int = 500
    min_tokens: int = 10
    temperature: float = 0.8
    top_k: int = 3
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    eos_token_id: int = 1024  # End of semantic sequence


@dataclass
class GenerationOutput:
    """Output from generation."""

    tokens: mx.array  # Generated token IDs
    num_tokens: int  # Number of tokens generated
    finished: bool  # Whether generation finished (EOS or max_tokens)


def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> mx.array:
    """Sample a token from logits with temperature, top-k, and top-p.

    Args:
        logits: Logits tensor [batch, vocab_size]
        temperature: Temperature for scaling (0 = argmax)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p

    Returns:
        Sampled token IDs [batch]
    """
    # Temperature scaling
    if temperature == 0:
        # Greedy decoding
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < logits.shape[-1]:
        # Get top-k indices and values
        top_k_indices = mx.argpartition(-logits, top_k, axis=-1)[..., :top_k]
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

        # Create mask for non-top-k tokens
        logits = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
        logits = mx.put_along_axis(logits, top_k_indices, top_k_logits, axis=-1)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        # Sort by descending probability
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

        # Compute cumulative probabilities
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove = mx.concatenate([
            mx.zeros_like(sorted_indices_to_remove[..., :1]),
            sorted_indices_to_remove[..., :-1]
        ], axis=-1)

        # Scatter back to original order
        indices_to_remove = mx.put_along_axis(
            mx.zeros_like(sorted_indices_to_remove),
            sorted_indices,
            sorted_indices_to_remove,
            axis=-1,
        )
        logits = mx.where(indices_to_remove, float("-inf"), logits)

    # Sample from distribution
    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))


def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: mx.array,
    penalty: float = 1.0,
) -> mx.array:
    """Apply repetition penalty to discourage repeated tokens.

    Args:
        logits: Current logits [batch, vocab_size]
        generated_tokens: Previously generated tokens [batch, seq]
        penalty: Penalty factor (1.0 = no penalty, >1.0 = discourage)

    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits

    # Get unique tokens that have been generated
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]

    for b in range(batch_size):
        unique_tokens = mx.unique(generated_tokens[b])
        for token in unique_tokens:
            token_id = int(token.item())
            if 0 <= token_id < vocab_size:
                # Reduce probability of repeated tokens
                if logits[b, token_id] > 0:
                    logits = logits.at[b, token_id].set(logits[b, token_id] / penalty)
                else:
                    logits = logits.at[b, token_id].set(logits[b, token_id] * penalty)

    return logits


def generate_semantic_tokens(
    model: GPTSoVITS,
    phoneme_ids: mx.array,
    bert_features: mx.array,
    config: Optional[GenerationConfig] = None,
    start_token_id: int = 0,
    callback: Optional[Callable[[int, mx.array], bool]] = None,
) -> GenerationOutput:
    """Generate semantic tokens autoregressively.

    Args:
        model: GPT model
        phoneme_ids: Phoneme token IDs [batch, phoneme_seq]
        bert_features: BERT/RoBERTa features [batch, text_seq, 1024]
        config: Generation configuration
        start_token_id: Token to start generation with
        callback: Optional callback(step, token) -> should_stop

    Returns:
        GenerationOutput with generated tokens
    """
    if config is None:
        config = GenerationConfig()

    batch_size = phoneme_ids.shape[0]

    # Initialize with start token
    current_token = mx.full((batch_size, 1), start_token_id, dtype=mx.int32)
    all_tokens = [current_token]

    # Create KV caches
    caches = model.create_caches()

    # Prefill: process phonemes and BERT features
    # For the first step, we need to process the full context
    logits, caches = model(
        phoneme_ids=phoneme_ids,
        semantic_ids=current_token,
        bert_features=bert_features,
        cache=caches,
    )

    # Get logits for next token prediction
    next_logits = logits[:, -1, :]  # [batch, vocab]

    # Sample first token
    next_token = sample_token(
        next_logits,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    next_token = next_token.reshape(batch_size, 1)
    all_tokens.append(next_token)

    # Check for early stopping
    finished = mx.all(next_token == config.eos_token_id)

    # Autoregressive generation
    for step in range(1, config.max_tokens):
        if finished:
            break

        # Process only the new token (use cache for previous)
        logits, caches = model(
            phoneme_ids=phoneme_ids,
            semantic_ids=next_token,
            bert_features=bert_features,
            cache=caches,
        )

        next_logits = logits[:, -1, :]

        # Apply repetition penalty if enabled
        if config.repetition_penalty != 1.0:
            generated_so_far = mx.concatenate(all_tokens, axis=1)
            next_logits = apply_repetition_penalty(
                next_logits, generated_so_far, config.repetition_penalty
            )

        # Sample next token
        next_token = sample_token(
            next_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
        next_token = next_token.reshape(batch_size, 1)

        # Async eval for pipelining (start GPU work before CPU continues)
        mx.async_eval(next_token)

        all_tokens.append(next_token)

        # Check for EOS
        finished = mx.all(next_token == config.eos_token_id)

        # Callback for progress/early stopping
        if callback is not None:
            should_stop = callback(step, next_token)
            if should_stop:
                break

    # Concatenate all tokens
    tokens = mx.concatenate(all_tokens, axis=1)

    # Remove start token and EOS if present
    tokens = tokens[:, 1:]  # Remove start token

    # Find EOS position and truncate
    eos_mask = tokens == config.eos_token_id
    if mx.any(eos_mask):
        # Find first EOS in each batch
        eos_positions = mx.argmax(eos_mask.astype(mx.int32), axis=1)
        # Create mask for valid tokens
        positions = mx.arange(tokens.shape[1])
        valid_mask = positions < eos_positions.reshape(-1, 1)
        # Mask out tokens after EOS
        tokens = mx.where(valid_mask, tokens, 0)

    return GenerationOutput(
        tokens=tokens,
        num_tokens=tokens.shape[1],
        finished=bool(finished),
    )


def generate_streaming(
    model: GPTSoVITS,
    phoneme_ids: mx.array,
    bert_features: mx.array,
    config: Optional[GenerationConfig] = None,
    start_token_id: int = 0,
) -> Iterator[mx.array]:
    """Generate semantic tokens with streaming output.

    Yields tokens one at a time as they are generated.

    Args:
        model: GPT model
        phoneme_ids: Phoneme token IDs [batch, phoneme_seq]
        bert_features: BERT/RoBERTa features [batch, seq_len, 1024]
        config: Generation configuration
        start_token_id: Token to start generation with

    Yields:
        Token ID arrays [batch, 1]
    """
    if config is None:
        config = GenerationConfig()

    batch_size = phoneme_ids.shape[0]
    current_token = mx.full((batch_size, 1), start_token_id, dtype=mx.int32)
    all_tokens = [current_token]

    # Create caches
    caches = model.create_caches()

    # Prefill
    logits, caches = model(
        phoneme_ids=phoneme_ids,
        semantic_ids=current_token,
        bert_features=bert_features,
        cache=caches,
    )

    next_logits = logits[:, -1, :]
    next_token = sample_token(
        next_logits,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    next_token = next_token.reshape(batch_size, 1)
    all_tokens.append(next_token)

    yield next_token

    # Check EOS
    if mx.all(next_token == config.eos_token_id):
        return

    # Continue generation
    for step in range(1, config.max_tokens):
        logits, caches = model(
            phoneme_ids=phoneme_ids,
            semantic_ids=next_token,
            bert_features=bert_features,
            cache=caches,
        )

        next_logits = logits[:, -1, :]

        if config.repetition_penalty != 1.0:
            generated_so_far = mx.concatenate(all_tokens, axis=1)
            next_logits = apply_repetition_penalty(
                next_logits, generated_so_far, config.repetition_penalty
            )

        next_token = sample_token(
            next_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
        next_token = next_token.reshape(batch_size, 1)

        mx.async_eval(next_token)
        all_tokens.append(next_token)

        yield next_token

        if mx.all(next_token == config.eos_token_id):
            return

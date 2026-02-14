"""Benchmark Qwen3-8B inference speed using mlx-lm for comparison with SALA."""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

FILLER = (
    "The annual rainfall in the Pacific Northwest varies between 150 and 200 centimeters "
    "depending on elevation and proximity to the coast. Meteorologists track these patterns "
    "using a network of ground stations and satellite imagery. The data collected helps farmers "
    "plan their crop rotations and irrigation schedules throughout the growing season. "
    "Modern bread baking techniques combine traditional fermentation methods with precise "
    "temperature control. The ideal proofing temperature for sourdough ranges from 24 to 27 "
    "degrees Celsius. Professional bakers monitor dough hydration levels carefully, as even "
    "small variations can significantly affect the final crumb structure and crust development. "
)

def build_prompt(tokenizer, target_tokens):
    question = "Summarize the key facts above in one sentence."
    base = PROMPT_TEMPLATE.format("")
    base_len = len(tokenizer.encode(base + question))
    filler_enc = tokenizer.encode(FILLER)
    filler_len = len(filler_enc)
    repeats = max(1, (target_tokens - base_len) // filler_len)
    body = (FILLER + "\n\n") * repeats
    prompt = PROMPT_TEMPLATE.format(body + question)
    actual_len = len(tokenizer.encode(prompt))
    return prompt, actual_len

def bench_model(model_name, target_lengths, max_tokens=64):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load(model_name)
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    results = []
    for target_len in target_lengths:
        prompt, actual_len = build_prompt(tokenizer, target_len)
        print(f"--- Context: {actual_len} tokens (target {target_len}) ---")

        prompt_tokens = mx.array(tokenizer.encode(prompt))

        # Time TTFT and decode separately using generate_step
        ttft = None
        decode_times = []
        t_start = time.time()
        n = 0
        for token, _ in generate_step(prompt_tokens, model, max_tokens=max_tokens):
            mx.eval(token)
            now = time.time()
            if ttft is None:
                ttft = now - t_start
            else:
                decode_times.append(now - t_prev)
            t_prev = now
            n += 1

        prefill_tps = actual_len / ttft if ttft else 0
        if decode_times:
            avg_decode = sum(decode_times) / len(decode_times)
            decode_tps = 1.0 / avg_decode
        else:
            decode_tps = 0

        print(f"  Prefill: {ttft:.2f}s ({prefill_tps:.1f} tok/s)")
        print(f"  Decode:  {decode_tps:.1f} tok/s ({n-1} tokens)")

        results.append({
            "context": actual_len,
            "prefill_tps": prefill_tps,
            "prefill_time": ttft,
            "decode_tps": decode_tps,
        })

        # Reset model cache
        mx.clear_cache()

    return results

if __name__ == "__main__":
    targets = [1000, 4000, 8000, 16000, 32000]
    max_tok = 64

    qwen_results = bench_model("Qwen/Qwen3-8B-MLX-8bit", targets, max_tok)

    print(f"\n{'='*60}")
    print("Qwen3-8B-8bit (mlx-lm, Apple M3 Max 128GB)")
    print(f"{'='*60}")
    print(f"{'Context':>8} | {'Prefill':>14} | {'Decode':>13} | {'TTFT':>8}")
    print(f"{'-'*8}-+-{'-'*14}-+-{'-'*13}-+-{'-'*8}")
    for r in qwen_results:
        print(f"{r['context']:>7}  | {r['prefill_tps']:>10.1f} t/s | {r['decode_tps']:>9.1f} t/s | {r['prefill_time']:>5.1f}s")

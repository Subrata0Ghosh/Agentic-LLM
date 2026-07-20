"""
generate.py — Qwen2.5-0.5B-Instruct text generation
=====================================================

Root cause of blank responses on HF Spaces (confirmed via transformers GitHub issues):
  - transformers `pipeline(return_full_text=False)` uses CHARACTER-LEVEL slicing to remove
    the prompt from the output. With chat-template models (Qwen, LLaMA-Instruct, etc.),
    the decoded prompt string differs from the original input due to special tokens
    (<|im_start|>, <|im_end|>, etc.), so the slice is wrong → empty or garbage output.

Fix: Use `AutoModelForCausalLM.generate()` with `apply_chat_template` and TOKEN-LEVEL
slicing (output_ids[len(input_ids):]), which is always correct regardless of special tokens.
Reference: https://huggingface.co/docs/transformers/model_doc/qwen2
"""

import os
import time
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None
_load_lock = threading.Lock()


def log_debug(message: str):
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/generation_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception as e:
        print(f"Failed to write debug log: {e}")


def get_model():
    """Load tokenizer + model once, cache globally. Thread-safe."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    with _load_lock:
        if _model is not None:          # Double-checked locking
            return _tokenizer, _model

        print(f"[generate] Loading {MODEL_NAME}... (first load only)")

        # Force CPU on HF Spaces to avoid ZeroGPU thread allocation errors
        if "SPACE_ID" in os.environ:
            device_map = "cpu"
            dtype = torch.float32
            print("[generate] Running on Hugging Face Spaces - forcing CPU execution")
        else:
            device_map = "auto"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map=device_map,
        )
        _model.eval()
        print(f"[generate] {MODEL_NAME} loaded and cached.")

    return _tokenizer, _model


def generate_text_api(
    messages,
    max_new_tokens: int = 200,
    temperature: float = 0.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
) -> str:
    """
    Generate text using Qwen2.5-0.5B-Instruct.

    Uses apply_chat_template + model.generate() with TOKEN-LEVEL slicing —
    the only approach guaranteed to return non-empty output with chat models.

    Args:
        messages: list of {"role": ..., "content": ...} dicts
    Returns:
        str — the assistant's reply (never empty; falls back to a safe message)
    """
    try:
        tokenizer, model = get_model()

        # Build prompt using the model's official chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,   # Adds <|im_start|>assistant\n
        )
        log_debug(f"PROMPT (first 200 chars): {text[:200]}")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_len = model_inputs.input_ids.shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

        # TOKEN-LEVEL slicing: strip the input tokens, decode only new tokens
        new_tokens = generated_ids[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        log_debug(f"SUCCESS: response={response[:120]}")

        if not response:
            log_debug("WARNING: empty response after decode, using fallback")
            return "I processed your message but couldn't formulate a response. Please try again."

        return response

    except Exception as e:
        import traceback
        err = f"ERROR in generate_text_api: {e}\n{traceback.format_exc()}"
        log_debug(err)
        print(err)
        return "I'm sorry, I encountered an error while trying to think of a response."


def generate_with_timeout(
    messages,
    max_new_tokens: int = 200,
    temperature: float = 0.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
    timeout_s: float = 60.0,
) -> str:
    """
    Timeout-protected wrapper around generate_text_api.
    If generation exceeds timeout_s seconds, returns a safe fallback.
    """
    result_container = [None]
    error_container  = [None]

    def _run():
        try:
            result_container[0] = generate_text_api(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        except Exception as e:
            error_container[0] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        print(f"[generate] WARNING: Generation timed out after {timeout_s}s")
        return "I'm taking too long to think. Please try a shorter or simpler query."

    if error_container[0]:
        print(f"[generate] Generation error: {error_container[0]}")
        return "I encountered an error while generating a response."

    return result_container[0] or "I was unable to generate a response."


if __name__ == "__main__":
    test_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain what a neural network is in one sentence."},
    ]
    print(generate_text_api(test_msgs))

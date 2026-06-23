#!/usr/bin/env python3
"""
Small local console chatbot using a quantized GGUF model and llama.cpp.

Examples:
    python chat.py
    python chat.py --model 1.5b
    python chat.py --model-file /path/to/model.gguf
    python chat.py --ctx-size 1024 --threads 4
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any

MODEL_OPTIONS = {
    "0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "label": "Qwen2.5 0.5B Instruct Q4_K_M",
    },
    "1.5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "label": "Qwen2.5 1.5B Instruct Q4_K_M",
    },
}

SYSTEM_PROMPT = (
    "You are a concise, helpful local assistant. "
    "Answer directly. When uncertain, say that you are uncertain."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small quantized LLM as a local console chatbot."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_OPTIONS,
        default="0.5b",
        help="Built-in model selection. Default: 0.5b",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        help="Use an existing local GGUF file instead of downloading a built-in model.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=1536,
        help="Context-window allocation. Lower values use less RAM. Default: 1536",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens generated per answer. Default: 256",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) - 1)),
        help="CPU worker threads.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help=(
            "Layers offloaded to a supported GPU. "
            "Default: all layers on Apple Silicon; CPU-only elsewhere."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature. Default: 0.4",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold. Default: 0.9",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=6,
        help="Number of recent user/assistant exchanges retained. Default: 6",
    )
    return parser.parse_args()


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def resolve_gpu_layers(requested: int | None) -> int:
    if requested is not None:
        return requested
    return -1 if is_apple_silicon() else 0


def obtain_model(args: argparse.Namespace) -> tuple[Path, str]:
    if args.model_file:
        model_path = args.model_file.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        return model_path, model_path.name

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is not installed. Run: pip install -r requirements.txt"
        ) from exc

    selected = MODEL_OPTIONS[args.model]
    print(f"Model: {selected['label']}")
    print("Checking the local model cache; the first run downloads the GGUF file...")

    path = hf_hub_download(
        repo_id=selected["repo_id"],
        filename=selected["filename"],
    )
    return Path(path), selected["label"]


def load_llm(model_path: Path, args: argparse.Namespace) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Follow the installation section "
            "in README.md, then run this program again."
        ) from exc

    gpu_layers = resolve_gpu_layers(args.gpu_layers)
    print(
        f"Loading {model_path.name} | context={args.ctx_size} | "
        f"threads={args.threads} | gpu_layers={gpu_layers}"
    )

    return Llama(
        model_path=str(model_path),
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        n_threads_batch=args.threads,
        n_gpu_layers=gpu_layers,
        n_batch=min(256, args.ctx_size),
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help   Show commands\n"
        "  /clear  Clear conversation memory\n"
        "  /exit   Exit the program\n"
    )


def stream_answer(llm: Any, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    chunks = llm.create_chat_completion(
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=1.1,
        stream=True,
    )

    collected: list[str] = []
    print("\nAssistant: ", end="", flush=True)

    for chunk in chunks:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        text = delta.get("content", "")
        if text:
            print(text, end="", flush=True)
            collected.append(text)

    print("\n")
    return "".join(collected).strip()


def trim_history(messages: list[dict[str, str]], history_turns: int) -> None:
    # Retain the system message plus the latest N user/assistant pairs.
    maximum_messages = 1 + (max(0, history_turns) * 2)
    if len(messages) > maximum_messages:
        del messages[1 : len(messages) - (maximum_messages - 1)]


def main() -> int:
    args = parse_args()

    if args.ctx_size < 512:
        print("Error: --ctx-size must be at least 512.", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("Error: --max-tokens must be positive.", file=sys.stderr)
        return 2

    try:
        model_path, model_label = obtain_model(args)
        llm = load_llm(model_path, args)
    except Exception as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        return 1

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print(f"\nReady: {model_label}")
    print_help()

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_text:
            continue

        command = user_text.lower()
        if command in {"/exit", "/quit"}:
            print("Goodbye.")
            return 0
        if command == "/help":
            print_help()
            continue
        if command == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation memory cleared.\n")
            continue

        messages.append({"role": "user", "content": user_text})
        trim_history(messages, args.history_turns)

        try:
            answer = stream_answer(llm, messages, args)
        except Exception as exc:
            # The most common runtime failure on small machines is context exhaustion.
            print(f"\nGeneration failed: {exc}\n", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": answer})
        trim_history(messages, args.history_turns)


if __name__ == "__main__":
    raise SystemExit(main())

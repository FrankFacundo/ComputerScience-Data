"""Run Qwen3-Embedding-0.6B with the local pure-torch implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qwen_embedding_0_6B_torch import (
    Qwen2Tokenizer,
    Qwen3EmbeddingModel,
    embed_texts,
)


DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    if requested is not None:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            requested
        ]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-torch Qwen3-Embedding-0.6B retrieval embedding demo."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto-detect if omitted).")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Parameter dtype. Defaults to fp32 on CPU, fp16 on MPS, bf16/fp16 on CUDA.",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--query",
        action="append",
        help="Query text. May be passed multiple times. Defaults to two sample queries.",
    )
    parser.add_argument(
        "--document",
        action="append",
        help="Document text. May be passed multiple times. Defaults to two sample documents.",
    )
    return parser.parse_args()


def _default_queries(task: str) -> list[str]:
    return [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]


def _default_documents() -> list[str]:
    return [
        "The capital of China is Beijing.",
        (
            "Gravity is a force that attracts two bodies towards each other. It gives weight "
            "to physical objects and is responsible for the movement of planets around the sun."
        ),
    ]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path, padding_side="left")
    model = Qwen3EmbeddingModel.from_pretrained(model_path, dtype=dtype, device=device).eval()

    queries = [get_detailed_instruct(args.task, q) for q in args.query] if args.query else _default_queries(args.task)
    documents = args.document if args.document else _default_documents()
    input_texts = queries + documents

    embeddings = embed_texts(
        model,
        tokenizer,
        input_texts,
        max_length=args.max_length,
        normalize=True,
    )
    scores = embeddings[: len(queries)] @ embeddings[len(queries) :].T
    print(scores.tolist())


if __name__ == "__main__":
    main()

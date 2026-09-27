"""CLI entry point: answer a single question or run the sample batch.

Run from the repo root as:
    python -m src.main --question "..."
    python -m src.main --batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import answer

REPO_ROOT = Path(__file__).resolve().parent.parent


def _print_result(question: str) -> None:
    result = answer(question)
    print(json.dumps(result.model_dump(), indent=2))


def main() -> None:
    """Parse CLI args and run the agent on one question or the batch file."""
    parser = argparse.ArgumentParser(description="Policy-aware support agent CLI")
    parser.add_argument("--question", type=str, help="A single question to answer")
    parser.add_argument(
        "--batch", action="store_true", help="Run all questions in data/questions.json"
    )
    args = parser.parse_args()

    if args.batch:
        questions = json.loads((REPO_ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
        for item in questions:
            print(f"\n--- {item['question']} ---")
            _print_result(item["question"])
    elif args.question:
        _print_result(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

from google_search import GoogleCustomSearchClient

QUERIES_FILE = Path("queries_gemini.json")
OUT_FILE = Path("search_results.jsonl")

api_key = os.getenv("GOOGLE_SEARCH_KEY")
cx = os.getenv("GOOGLE_CSE_ID")
client = GoogleCustomSearchClient(api_key=api_key, cx=cx)


def _load_queries() -> list[str]:
    """Return the list of queries from queries_gemini.json."""
    with QUERIES_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def _already_processed() -> set[str]:
    """Return the set of queries already present in the jsonl file."""
    if not OUT_FILE.exists():
        return set()
    processed = set()
    with OUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.update(record.keys())
            except json.JSONDecodeError:
                # malformed line ⇒ ignore but keep going
                continue
    return processed


def _append_result(query: str, res: dict) -> None:
    """Append a single query → result mapping to the jsonl file."""
    with OUT_FILE.open("a", encoding="utf-8") as f:
        json.dump({query: res}, f, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    queries = _load_queries()
    done = _already_processed()

    for q in queries:
        if q in done:
            continue
        try:
            res = client.search_site(q, site="bgl.lu", hl="fr", gl="LU")
        except Exception as exc:  # network, quota, etc.
            print(f"Error for '{q}': {exc}")
            continue

        _append_result(q, res)
        print(f"Saved results for: {q}")


if __name__ == "__main__":
    main()

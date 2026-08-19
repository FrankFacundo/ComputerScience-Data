#!/usr/bin/env python3
"""Check that an Anthropic API key works, by sending a minimal request to each model.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python check_anthropic_api_key.py

    # or pass the key explicitly / test other models
    python check_anthropic_api_key.py --api-key sk-ant-... --models claude-opus-5 claude-haiku-4-5

Exit codes: 0 = every model answered, 1 = at least one model failed, 2 = no key / SDK missing.
"""

import argparse
import os
import sys
import time

DEFAULT_MODELS = ["claude-opus-5", "claude-sonnet-5"]
PROMPT = "Reply with the single word: pong"

try:
    import anthropic
except ImportError:
    print("anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(2)


def mask(key):
    return f"{key[:11]}...{key[-4:]}" if len(key) > 19 else "***"


def probe(client, model, max_tokens, effort):
    """Send one minimal request. Returns (ok, detail_dict)."""
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": PROMPT}],
        )
    except anthropic.AuthenticationError as e:  # 401 — key invalid, revoked or missing
        return False, {"error": "401 authentication_error — key is invalid or revoked", "detail": e.message}
    except anthropic.PermissionDeniedError as e:  # 403 — key has no access to this model
        return False, {"error": f"403 {e.type} — key lacks access to {model} (or billing issue)", "detail": e.message}
    except anthropic.NotFoundError as e:  # 404 — unknown model id
        return False, {"error": f"404 not_found_error — no such model '{model}'", "detail": e.message}
    except anthropic.RateLimitError as e:  # 429 — key is valid, just throttled
        return False, {"error": "429 rate_limit_error — key is valid but throttled", "detail": e.message}
    except anthropic.APIStatusError as e:  # any other non-2xx
        return False, {"error": f"HTTP {e.status_code} {e.type}", "detail": e.message}
    except anthropic.APIConnectionError as e:  # network failure before a response
        return False, {"error": "connection error — could not reach the API", "detail": str(e.__cause__ or e)}

    elapsed = time.perf_counter() - start
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return True, {
        "seconds": elapsed,
        "text": text or f"(no text — stop_reason={response.stop_reason})",
        "stop_reason": response.stop_reason,
        "usage": response.usage,
        "request_id": response._request_id,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", help="key to test (default: $ANTHROPIC_API_KEY)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"models to test (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens per probe (default: 256)")
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"], help="thinking/effort level (default: low, the cheapest)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print token usage and request ids")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("No API key. Pass --api-key, or export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(2)

    client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=60.0)
    print(f"Key:    {mask(api_key)}")
    print(f"Models: {', '.join(args.models)}\n")

    failures = 0
    for model in args.models:
        print(f"{model:<20} ", end="", flush=True)
        ok, info = probe(client, model, args.max_tokens, args.effort)
        if ok:
            print(f"OK    {info['seconds']:5.2f}s  -> {info['text'][:60]!r}")
            if args.verbose:
                u = info["usage"]
                print(f"{'':<20}       in={u.input_tokens} out={u.output_tokens} stop={info['stop_reason']} req={info['request_id']}")
        else:
            failures += 1
            print(f"FAIL  {info['error']}")
            if args.verbose and info.get("detail"):
                print(f"{'':<20}       {info['detail']}")

    print()
    if failures:
        print(f"{failures}/{len(args.models)} model(s) failed.")
        return 1
    print(f"Key works on all {len(args.models)} model(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

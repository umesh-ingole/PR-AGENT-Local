"""Smoke-test the configured local Ollama provider."""
import time

import config  # Load project .env settings before creating the LLM client.
from llm_gateway import call_llm


def main() -> None:
    started = time.perf_counter()
    try:
        response, provider = call_llm(
            "You are a concise assistant.",
            "reply with the word ready",
            force_local=True,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"Ollama check failed after {elapsed:.2f} seconds: {exc}")
        raise

    elapsed = time.perf_counter() - started
    print(f"Provider: {provider}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
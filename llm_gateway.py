"""LLM Gateway (purple box): primary Gemini, with Ollama as the offline fallback."""
import logging
import os
import re
import time
from collections.abc import Callable


GEMINI_COOLDOWN_SECONDS = 10 * 60
_gemini_unavailable_until = 0.0
_call_observer: Callable[[str, str, str, float, bool | None], None] | None = None
_logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    def __init__(self, failures: list[tuple[str, str]]):
        self.failures = tuple(failures)
        summary = " | ".join(
            f"{provider}: {error_type}" for provider, error_type in failures
        )
        super().__init__("All LLM providers failed -> " + summary)


def set_call_observer(
    observer: Callable[[str, str, str, float, bool | None], None] | None,
) -> Callable[[str, str, str, float, bool | None], None] | None:
    """Set the optional per-attempt observer, returning the previous observer."""
    global _call_observer
    previous = _call_observer
    _call_observer = observer
    return previous


def _model_name(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    return provider


def _notify_call(
    stage: str,
    provider: str,
    model: str,
    duration: float,
    success: bool | None,
) -> None:
    if _call_observer is not None:
        _call_observer(stage, provider, model, duration, success)


def _is_quota_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    code = getattr(exc, "code", None)
    return (
        status == 429
        or code == 429
        or "429" in str(exc)
        or "RESOURCE_EXHAUSTED" in str(exc).upper()
    )


def _safe_quota_reason(exc: Exception) -> str:
    reason = str(exc)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        reason = reason.replace(api_key, "[redacted]")
    reason = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[redacted]", reason)
    reason = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", reason)
    reason = " ".join(reason.split())
    return reason[:100]


def _provider_order() -> list[str]:
    raw = os.getenv("LLM_PROVIDERS", "gemini,ollama")
    providers = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return providers or ["gemini", "ollama"]


def _make(provider: str):
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
            max_retries=1,
            timeout=90,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            num_predict=2048,
            client_kwargs={"timeout": 600},
            temperature=0.1)
    raise ValueError(provider)


def call_llm(system: str, user: str, force_local: bool = False) -> tuple[str, str]:
    """Return (text, provider_used). Raises RuntimeError if every provider fails."""
    global _gemini_unavailable_until
    order = ["ollama"] if force_local else _provider_order()
    if "gemini" in order and time.monotonic() < _gemini_unavailable_until:
        order = [provider for provider in order if provider != "gemini"]
    order = order[:2]
    errors = []
    for provider in order:
        model = _model_name(provider)
        call_started = time.perf_counter()
        _notify_call("start", provider, model, 0.0, None)
        try:
            llm = _make(provider)
            reply = llm.invoke([("system", system), ("human", user)])
            content = reply.content
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            _notify_call("end", provider, model, time.perf_counter() - call_started, True)
            return content, provider
        except Exception as exc:
            duration = time.perf_counter() - call_started
            _notify_call("end", provider, model, duration, False)
            errors.append((provider, type(exc).__name__))
            _logger.warning(
                "LLM provider failed: provider=%s error=%s",
                provider,
                type(exc).__name__,
            )
            if provider == "gemini" and _is_quota_error(exc):
                _gemini_unavailable_until = time.monotonic() + GEMINI_COOLDOWN_SECONDS
                reason = _safe_quota_reason(exc)
                fallback = "ollama" in order[order.index(provider) + 1:]
                suffix = ", falling back to ollama" if fallback else ", no fallback configured"
                reason_text = f" ({reason})" if reason else ""
                print(f"gemini quota exhausted (429){reason_text}{suffix}")
    raise LLMUnavailableError(errors)

"""LLM Gateway (purple box): primary Gemini, with Ollama as the offline fallback."""
import os


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
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.1)
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
    order = ["ollama"] if force_local else _provider_order()
    errors = []
    for provider in order:
        try:
            llm = _make(provider)
            reply = llm.invoke([("system", system), ("human", user)])
            content = reply.content
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            return content, provider
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))

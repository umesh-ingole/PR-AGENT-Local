import sys
import types

import pytest

import llm_gateway


class QuotaError(Exception):
    status_code = 429


class FakeReply:
    content = "mock reply"


def test_gemini_client_uses_one_retry_and_90_second_timeout(monkeypatch):
    captured = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeGemini
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")

    llm_gateway._make("gemini")

    assert captured["max_retries"] == 1
    assert captured["timeout"] == 90


def test_quota_error_falls_back_and_skips_gemini_during_cooldown(
    monkeypatch, capsys,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-secret-value")
    monkeypatch.setenv("LLM_PROVIDERS", "gemini,ollama")
    monkeypatch.setattr(llm_gateway, "_gemini_unavailable_until", 0.0)
    monotonic_time = {"now": 100.0}
    monkeypatch.setattr(
        llm_gateway.time, "monotonic", lambda: monotonic_time["now"]
    )
    attempted = []

    class FailingGemini:
        def invoke(self, _messages):
            attempted.append("gemini")
            raise QuotaError(
                "HTTP 429 RESOURCE_EXHAUSTED mock-secret-value " + "x" * 150
            )

    class WorkingOllama:
        def invoke(self, _messages):
            attempted.append("ollama")
            return FakeReply()

    monkeypatch.setattr(
        llm_gateway,
        "_make",
        lambda provider: FailingGemini() if provider == "gemini" else WorkingOllama(),
    )

    assert llm_gateway.call_llm("system", "user") == ("mock reply", "ollama")
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 1
    assert printed[0].startswith("gemini quota exhausted (429)")
    assert "falling back to ollama" in printed[0]
    assert "mock-secret-value" not in printed[0]
    reason = printed[0].split("(", 2)[-1].split(")", 1)[0]
    assert len(reason) <= 100
    assert llm_gateway._gemini_unavailable_until == (
        100.0 + llm_gateway.GEMINI_COOLDOWN_SECONDS
    )

    monotonic_time["now"] += 1
    assert llm_gateway.call_llm("system", "user") == ("mock reply", "ollama")
    assert attempted == ["gemini", "ollama", "ollama"]
    assert capsys.readouterr().out == ""


def test_no_fallback_raises_after_gemini_quota_error(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")
    monkeypatch.setenv("LLM_PROVIDERS", "gemini")
    monkeypatch.setattr(llm_gateway, "_gemini_unavailable_until", 0.0)

    class FailingGemini:
        def invoke(self, _messages):
            raise QuotaError("RESOURCE_EXHAUSTED")

    monkeypatch.setattr(llm_gateway, "_make", lambda _provider: FailingGemini())

    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        llm_gateway.call_llm("system", "user")

    output = capsys.readouterr().out.strip()
    assert output.startswith("gemini quota exhausted (429)")
    assert output.endswith(", no fallback configured")


def test_call_observer_receives_provider_model_and_elapsed_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "mock-model")
    monkeypatch.setattr(
        llm_gateway, "_make", lambda _provider: types.SimpleNamespace(
            invoke=lambda _messages: FakeReply()
        )
    )
    events = []
    previous = llm_gateway.set_call_observer(
        lambda *event: events.append(event)
    )
    try:
        assert llm_gateway.call_llm("system", "user") == ("mock reply", "ollama")
    finally:
        llm_gateway.set_call_observer(previous)

    assert events[0][:3] == ("start", "ollama", "mock-model")
    assert events[1][:3] == ("end", "ollama", "mock-model")
    assert events[1][3] >= 0
    assert events[1][4] is True


def test_json_format_is_passed_to_ollama_only(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    captured = {}

    def fake_make(provider, format=None):
        captured.update({"provider": provider, "format": format})
        return types.SimpleNamespace(
            invoke=lambda _messages: FakeReply()
        )

    monkeypatch.setattr(llm_gateway, "_make", fake_make)

    assert llm_gateway.call_llm("system", "user", format="json") == (
        "mock reply", "ollama",
    )
    assert captured == {"provider": "ollama", "format": "json"}


def test_ollama_client_receives_json_format(monkeypatch):
    captured = {}

    class FakeOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("langchain_ollama")
    fake_module.ChatOllama = FakeOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)

    llm_gateway._make("ollama", format="json")

    assert captured["format"] == "json"

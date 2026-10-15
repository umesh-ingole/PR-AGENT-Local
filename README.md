# AI-Based Multilingual Code Review and Auto-Correction System

Streamlit app that follows the architecture diagram box for box.

| Diagram box | File |
|---|---|
| User Uploads Code (Website) / User Downloads | `app.py` |
| Static Analysis (MCP: cppcheck / pylint / checkstyle) | `static_analysis.py`, `mcp_server.py`, `tools.py` |
| RAG Retrieval + Vector DB | `rag.py`, `knowledge/*.md` (Chroma) |
| LLM Gateway (Gemini, with Ollama as the offline fallback) | `llm_gateway.py` |
| Security, Bug-Detection, Summary, Coder agents | `agents.py` |
| Score good enough? loop | `graph.py` (LangGraph) |

## Run locally (VS Code)
1. `python -m venv .venv` then activate it
2. `pip install -r requirements.txt`
3. Install the linters: cppcheck, a JRE + checkstyle, rubocop (pylint comes from pip). Missing ones are skipped with a note.
4. Copy `.env.example` to `.env` and add `GOOGLE_API_KEY` and/or `GROQ_API_KEY`
5. Optional offline fallback: install Ollama, then `ollama pull qwen2.5-coder:7b`
6. `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo
2. Create the app from `app.py`. `packages.txt` installs the linters; `requirements.txt` installs Python packages
3. App settings > Secrets:
   ```toml
   GOOGLE_API_KEY = "..."
   GROQ_API_KEY = "..."
   ```
4. Cloud has no Ollama, so the local fallback only works when you run the app yourself.
   For a hosted fallback, the Groq key is the practical second provider.

## Review safeguards
- After a Coder Agent fix, Python AST and secrets scans run again. Remaining critical/high findings trigger one bounded remediation call, separate from the configured review-round count; unresolved findings are reported for manual repair.
- Security and Bug Agent reviews request JSON from Ollama. Invalid replies are retained in the run folder by agent and round with detected secrets redacted.

## Tuning
- `config.py`: pass score (80), max fix rounds (3), RAG top-k (2)
- Add a language: add a tool in `tools.py`, expose it in `mcp_server.py`, map it in `static_analysis.py` and `config.py`
- Swap Chroma for FAISS or pgvector: change `_collection()` and the query in `rag.py` only

## Copilot prompts that work well here
- "Add a unit test for `merge_and_score` in agents.py covering duplicate findings."
- "Add a pgvector backend to rag.py behind a VECTOR_BACKEND env var."

# Troubleshooting

## Common Issues

### `EnvironmentError: GROQ_API_KEY is not set`

**Cause:** The application validated config before finding a key.

**Fix:**
```bash
cp .env.example .env
# Open .env and set GROQ_API_KEY=your_actual_key
```

Get a free key at https://console.groq.com/keys.

---

### `Search encountered an error: ...`

**Cause:** DuckDuckGo rate-limited the request or network is unavailable.

**Fix:**
- Wait 30–60 seconds and retry — DDG applies per-IP rate limits.
- Check your internet connection.
- Try reducing `SEARCH_MAX_RESULTS` and `NEWS_MAX_RESULTS` in `.env`.

The search tools are designed to fail gracefully — the pipeline will continue with the partial results it has.

---

### `json.JSONDecodeError` in Supervisor logs

**Cause:** The LLM returned a non-JSON response for the supervisor decision.

**Fix:** This is handled automatically — the Supervisor falls back to heuristic routing. If it happens repeatedly:
- Lower `GROQ_TEMPERATURE` to `0.0` for more deterministic JSON output.
- Enable debug logging (`LOG_LEVEL=DEBUG`) to see the raw LLM response.

---

### Workflow runs but `final_answer` is empty

**Cause:** The Supervisor hit `MAX_ITERATIONS` before reaching the reflect node.

**Fix:**
- Increase `MAX_ITERATIONS` in `.env` (e.g. `MAX_ITERATIONS=8`).
- Check the `messages` log in the output — it shows exactly what each iteration decided.

---

### Tests fail with `ImportError`

**Cause:** Dependencies not installed or wrong Python version.

**Fix:**
```bash
python --version          # must be 3.10, 3.11, or 3.12
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

---

### `ModuleNotFoundError: No module named 'ddgs'`

**Cause:** The DuckDuckGo search package is not installed.

**Fix:**
```bash
pip install ddgs
# or for the legacy package name:
pip install duckduckgo-search
```

Both are listed in `requirements.txt` — this should not happen after a fresh install.

---

### Slow responses / timeouts

**Cause:** Groq API latency or complex queries requiring many tool iterations.

**Fix:**
- Increase `GROQ_TIMEOUT` (e.g. `GROQ_TIMEOUT=120`).
- Use `llama-3.1-8b-instant` for faster but less detailed responses: `GROQ_MODEL=llama-3.1-8b-instant`.
- Reduce `MAX_TOOL_ITERATIONS=2` to limit search breadth.

---

## Debug Mode

Enable full debug logging to see every LLM prompt, tool call, and state update:

```bash
LOG_LEVEL=DEBUG python main.py "Your query"
```

This will print the complete message history for each agent, which is useful for understanding exactly what the LLM is reasoning about.

---

## Reporting Issues

Please include the following in bug reports:
1. Python version (`python --version`)
2. Package versions (`pip freeze | grep -E "langgraph|langchain|groq|ddgs"`)
3. Full error traceback (with `LOG_LEVEL=DEBUG`)
4. Whether the issue is reproducible with the default query

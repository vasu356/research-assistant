# API Reference

## State Schema

All agents communicate exclusively through `ResearchState` defined in `graph/state.py`.

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original user research query (immutable after init) |
| `search_results` | `Optional[str]` | Raw consolidated web search output |
| `summary` | `Optional[str]` | Structured markdown summary from Summarizer |
| `fact_check` | `Optional[str]` | Confidence-scored fact-check report |
| `reflection_notes` | `Optional[str]` | Self-assessment from Supervisor reflection pass |
| `final_answer` | `Optional[str]` | Polished final answer ready to return to user |
| `next_action` | `Optional[str]` | Routing signal: `search \| summarize \| fact_check \| reflect \| finish` |
| `iteration_count` | `int` | Number of Supervisor decision loops completed |
| `messages` | `List[str]` | Append-only agent activity log |
| `error` | `Optional[str]` | Last error message, if any |

---

## Agent Node Contracts

All node functions share the same signature:

```python
async def agent_node(state: ResearchState, llm: ChatGroq) -> Dict[str, Any]:
    ...
```

They receive the full state and return a **partial** state dict — only the fields they update. LangGraph merges the partial update into the shared state.

### `search_agent_node`

**Reads:** `state["query"]`  
**Writes:** `search_results`, `messages`, (optionally) `error`

Executes a ReAct tool-calling loop using `duckduckgo_search` and `duckduckgo_news_search`. Terminates when the LLM stops calling tools or `MAX_TOOL_ITERATIONS` is reached.

**Output format:** Begins with `=== WEB SEARCH RESULTS ===` followed by labelled result blocks.

---

### `summarizer_agent_node`

**Reads:** `state["query"]`, `state["search_results"]`  
**Writes:** `summary`, `messages`, (optionally) `error`

Guards: If `search_results` is empty, returns immediately with a skip message without calling the LLM.

**Output format:** Structured markdown with `### 📋 Summary`, `**Key Findings:**`, `**Detailed Analysis:**` sections.

---

### `fact_checker_agent_node`

**Reads:** `state["query"]`, `state["summary"]`, `state["search_results"]`  
**Writes:** `fact_check`, `messages`, (optionally) `error`

Guards: If `summary` is empty, returns immediately.

**Output format:** Markdown with `### 🔍 Fact-Check Report`, a confidence-level table, flagged issues, and recommendations.

---

### `supervisor_decision_node`

**Reads:** All fields  
**Writes:** `next_action`, `iteration_count`, `messages`

Calls the LLM and expects a JSON response:
```json
{
  "reasoning": "...",
  "next_action": "search|summarize|fact_check|reflect|finish",
  "rationale": "..."
}
```

Falls back to `_heuristic_next_action` if the response cannot be parsed as JSON.
Hard-stops at `MAX_ITERATIONS` without calling the LLM.

---

### `supervisor_reflect_node`

**Reads:** All fields  
**Writes:** `final_answer`, `reflection_notes`, `next_action`, `messages`, (optionally) `error`

Splits the LLM response on `### 🔄 Self-Reflection` to separate the answer from the self-assessment. If `NEEDS_IMPROVEMENT` is found in the reflection, sets `next_action = "search"`.

---

## Routing Functions

### `route_after_decision(state) -> str`

Called by LangGraph as the conditional edge from `supervisor_decision`.

| `next_action` value | Destination node |
|---------------------|-----------------|
| `"search"` | `"search_agent"` |
| `"summarize"` | `"summarizer_agent"` |
| `"fact_check"` | `"fact_checker_agent"` |
| `"reflect"` | `"reflect"` |
| `"finish"` or unknown | `"__end__"` |

### `route_after_reflection(state) -> str`

Called by LangGraph as the conditional edge from `reflect`.

| Condition | Destination |
|-----------|-------------|
| `next_action == "search"` AND `iteration_count < MAX_ITERATIONS` | `"search_agent"` |
| Otherwise | `"__end__"` |

---

## Search Tools

### `duckduckgo_search`

```python
@tool
def duckduckgo_search(query: str, max_results: int = 6) -> str
```

Returns formatted web search results. Each result block:
```
[Result N]
Title: ...
URL: ...
Snippet: ...
```

On failure, returns a descriptive error string (never raises).

---

### `duckduckgo_news_search`

```python
@tool
def duckduckgo_news_search(query: str, max_results: int = 5) -> str
```

Returns formatted news results. Each result block:
```
[News N]
Title: ...
URL: ...
Date: ...
Source: ...
Snippet: ...
```

On failure, returns a descriptive error string (never raises).

---

## Configuration Reference

See `config/settings.py` and `.env.example` for the full list. Key values:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Groq model | `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM to use |
| Temperature | `GROQ_TEMPERATURE` | `0.1` | Sampling temperature |
| Max tokens | `GROQ_MAX_TOKENS` | `4096` | Per-response token limit |
| Max iterations | `MAX_ITERATIONS` | `6` | Supervisor loop cap |
| Tool iterations | `MAX_TOOL_ITERATIONS` | `4` | Search ReAct loop cap |
| Search results | `SEARCH_MAX_RESULTS` | `6` | Web results per query |
| News results | `NEWS_MAX_RESULTS` | `5` | News results per query |
| Output directory | `OUTPUT_DIR` | `research_outputs` | Markdown export path |
| Log level | `LOG_LEVEL` | `INFO` | Logging verbosity |

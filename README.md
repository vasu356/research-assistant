# Multi-Agent Research Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-brightgreen?logo=langchain)](https://python.langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20(Llama%203.3%2070B)-orange)](https://console.groq.com)
[![Search](https://img.shields.io/badge/Search-DuckDuckGo_(Free)-red)](https://duckduckgo.com)
[![Async](https://img.shields.io/badge/Async-asyncio-purple)](https://docs.python.org/3/library/asyncio.html)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://opensource.org/licenses/MIT)

A **production-grade multi-agent research system** built with **LangGraph** that uses hierarchical delegation to orchestrate specialised AI agents — each with a single responsibility — to produce deep, fact-checked research answers with self-reflection and iterative improvement.

---

## Features

- 🧠 **Multi-Agent Architecture** — Four specialised agents coordinated by a supervisor
- 🔄 **ReAct Pattern** — Agents reason about state, act via tools, and observe results
- 🏗️ **Hierarchical Delegation** — Supervisor routes work to specialists, never does work itself
- 🔍 **Self-Reflection** — Final answer is critically evaluated; can trigger re-search for improvement
- 🛠️ **Tool-Calling** — Search agent autonomously decides search strategy
- ✅ **Fact-Checking** — Every claim is validated with confidence scoring
- 📊 **LangGraph State Management** — Typed, append-only state propagates through the graph
- 📁 **Markdown Export** — Results saved to timestamped markdown files
- 🌐 **Free Web Search** — DuckDuckGo integration (no API key required)
- 🧪 **Comprehensive Tests** — 38 tests with mocked LLM and tools (no network/API needed)

---

## Architecture

```
                         ┌─────────────────────────────────┐
                         │         USER QUERY               │
                         └──────────────┬──────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────┐
                         │       SUPERVISOR AGENT           │
                         │  (ReAct: Reason + Act Loop)      │
                         │                                  │
                         │  Evaluates state completeness    │
                         │  and decides next worker         │
                         └──────────┬──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │ "search"            │ "summarize"          │ "fact_check"
              ▼                     ▼                      ▼
   ┌────────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
   │   SEARCH AGENT     │ │  SUMMARIZER      │ │  FACT CHECKER       │
   │                    │ │  AGENT           │ │  AGENT              │
   │  • DuckDuckGo web  │ │                  │ │                     │
   │    + news search   │ │  • Structures    │ │  • Validates claims │
   │  • ReAct tool loop │ │    raw results   │ │  • Flags uncertain  │
   │  • Multi-query     │ │  • Markdown      │ │    information      │
   │    strategy        │ │    output        │ │  • Confidence       │
   │  • Auto-decides    │ │  • Source refs   │ │    scoring          │
   │    when done       │ │                  │ │                     │
   └─────────┬──────────┘ └────────┬─────────┘ └──────────┬──────────┘
             │                     │                       │
             └─────────────────────┴───────────────────────┘
                                    │ (all return to supervisor)
                                    │
                                    ▼ "reflect"
                         ┌─────────────────────────────────┐
                         │    SELF-REFLECTION NODE          │
                         │  (Supervisor writes + reviews    │
                         │   final answer, assesses         │
                         │   quality, decides verdict)      │
                         └──────────┬──────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ COMPLETE                       │ NEEDS_IMPROVEMENT
                    ▼                                ▼
              ┌──────────┐              ┌─────────────────────┐
              │   END    │              │  Loop back to       │
              │  (return │              │  search_agent       │
              │  answer) │              │  (max 3 iterations) │
              └──────────┘              └─────────────────────┘
```

### Agent Responsibilities

| Agent | Role | Pattern | Tools Used |
|-------|------|---------|------------|
| **Supervisor** | Orchestrates workflow, routes tasks, evaluates completeness | ReAct + Self-Reflection | LLM reasoning only |
| **Search** | Retrieves web information, formulates search queries | ReAct with tool-calling | `duckduckgo_search`, `duckduckgo_news_search` |
| **Summarizer** | Converts raw results to structured markdown summary | Single LLM call | LLM only |
| **Fact Checker** | Validates claims, assigns confidence levels | Single LLM call | LLM only |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Orchestration** | LangGraph 0.2+ |
| **LLM Framework** | LangChain 0.3+ |
| **LLM Provider** | Groq (LLaMA 3.3 70B) |
| **Web Search** | DuckDuckGo (free, no API key) |
| **State Schema** | TypedDict with append-only annotations |
| **Async Runtime** | Python asyncio |
| **Environment Management** | python-dotenv |
| **Testing** | unittest with mocked LLM/tools |

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- A free [**Groq API key**](https://console.groq.com/keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/vasu356/research-assistant.git
cd research-assistant

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### Usage

```bash
# Run with the default sample query
python main.py

# Run with a custom research query
python main.py "What are the main differences between GPT-4o and Claude 3.5 Sonnet?"

# Run tests
python tests/test_end_to_end.py
```

### Example Output

```
                          MULTI-AGENT RESEARCH ASSISTANT
================================================================================
Query: What are the latest developments in LLM reasoning models in 2025
        and how do they compare to traditional chain-of-thought approaches?
================================================================================
  [  2.0s] [Supervisor] Iteration 1: Action=search | search_results missing
  [ 50.9s] [SearchAgent] Completed 2 tool iteration(s). Retrieved 7 result block(s).
  [ 55.3s] [Supervisor] Iteration 2: Action=summarize | search results available
  [102.3s] [SummarizerAgent] Summary complete (4394 chars).
  [105.5s] [Supervisor] Iteration 3: Action=fact_check | summary available
  [119.8s] [FactCheckerAgent] Fact-check complete (3891 chars).
  [122.5s] [Supervisor] Iteration 4: Action=reflect | all components ready
  [137.7s] [Supervisor] Reflection complete. Verdict: COMPLETE

  ✓ Workflow complete
```

Results are also saved to `research_outputs/research_YYYYMMDD_HHMMSS_query.md`.

---

## Project Structure

```
research-assistant/
├── agents/
│   ├── __init__.py
│   ├── fact_checker.py      # Fact Checker Agent
│   ├── search_agent.py      # Web Search Agent
│   ├── summarizer.py        # Summarizer Agent
│   └── supervisor.py        # Supervisor (orchestrator + self-reflection)
├── graph/
│   ├── __init__.py
│   ├── state.py             # ResearchState TypedDict
│   └── workflow.py          # LangGraph StateGraph definition
├── tools/
│   ├── __init__.py
│   └── search_tools.py      # DuckDuckGo tool definitions
├── tests/
│   ├── __init__.py
│   └── test_end_to_end.py   # 38 tests (no network/API needed)
├── main.py                  # Entry point
├── pyproject.toml           # Project metadata and build config
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GROQ_TEMPERATURE` | `0.1` | Sampling temperature (lower = more deterministic) |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Agentic Patterns Implemented

### 1. 🔄 ReAct (Reason + Act)
Both the **Supervisor** and **Search Agent** implement the ReAct pattern:
- **Reason**: Analyse current state/query, identify what's needed
- **Act**: Call a tool or dispatch a worker agent
- **Observe**: Review the result and decide next step

### 2. 🏗️ Hierarchical Delegation
The **Supervisor** never does research itself. It acts as a pure orchestrator:
- Evaluates state after every worker node completion
- Routes to the appropriate specialist based on what's missing
- Enforces a max iteration limit (6 loops) to prevent infinite loops

### 3. 🔍 Self-Reflection
After all three workers have run, the Supervisor enters a reflection node where it:
- Writes a comprehensive final answer synthesising all components
- Critically evaluates the answer's quality (coverage, accuracy, depth)
- Returns a **COMPLETE** or **NEEDS_IMPROVEMENT** verdict
- If improvement needed, loops back to re-search (capped by max iterations)

### 4. 🛠️ Tool-Calling
The Search Agent uses LangChain's `bind_tools` to give the LLM autonomous control over:
- `duckduckgo_search` — general web search
- `duckduckgo_news_search` — recent news search

The LLM decides how many searches to run and with what queries.

### 5. 📊 State Management (LangGraph)
All agents share a typed `ResearchState` managed by LangGraph:
- Append-only `messages` list (via `operator.add` annotation)
- Atomic state updates returned as dicts from each node
- No global mutable state — everything flows through the graph

---

## Extending the System

- **Add a new agent**: Create `agents/my_agent.py`, register it in `graph/workflow.py`, add a routing case in `agents/supervisor.py`.
- **Swap the search tool**: Replace DuckDuckGo in `tools/search_tools.py` with Tavily, SerpAPI, or Bing.
- **Swap the LLM**: Change `get_llm()` in `graph/workflow.py` to use OpenAI, Anthropic, or another provider.
- **Add memory**: Inject a `checkpointer` into `graph.compile()` for persistent conversation history.
- **Add human-in-the-loop**: Use LangGraph's `interrupt_before` to pause and request user input.

---

## Testing

The test suite runs **38 tests** covering every component:

```
python tests/test_end_to_end.py
```

Tests use mocked LLM and tools — no network access or API key required. The suite validates:
- State schema correctness
- Tool registry and error handling
- Supervisor routing logic
- Graph structure and node connectivity
- Full end-to-end workflow with canned responses
- Individual agent node error handling
- Max iteration guard (infinite loop prevention)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Vasu Agrawal**

- [GitHub](https://github.com/vasu356)
- [LinkedIn](https://www.linkedin.com/in/vasu-agrawal-m26a2003y)

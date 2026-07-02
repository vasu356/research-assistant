# Contributing

Thank you for considering a contribution! This document explains the workflow for making changes.

---

## Development Setup

```bash
git clone https://github.com/vasu356/research-assistant.git
cd research-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest pytest-asyncio pytest-cov
cp .env.example .env   # add your GROQ_API_KEY
```

---

## Coding Standards

- **Style:** [ruff](https://docs.astral.sh/ruff/) enforces formatting and linting.
- **Types:** All public functions should have type annotations.
- **Docstrings:** Follow the existing Google-style format with `Args:` and `Returns:` sections.
- **Error handling:** Never let exceptions propagate out of agent node functions — return an error string in state.

Run checks before committing:

```bash
ruff check .
ruff format .
python -m pytest tests/ -v
```

---

## Adding a New Agent

1. Create `agents/my_agent.py` with an async node function:
   ```python
   async def my_agent_node(state: ResearchState, llm: ChatGroq) -> Dict[str, Any]:
       ...
   ```
2. Register in `graph/workflow.py`.
3. Add a routing case in `agents/supervisor.py` → `route_after_decision`.
4. Update the Supervisor's system prompt to describe the new agent.
5. Write tests in `tests/test_end_to_end.py` covering the new node.

---

## Pull Request Guidelines

- **Branch naming:** `feat/description`, `fix/description`, `docs/description`
- **Commit style:** Conventional commits — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- **Tests required:** All PRs must include or update tests. No PR merges with failing tests.
- **PR description:** Explain the problem, the solution, and how to test it.

---

## Issue Reporting

Use the issue templates in `.github/ISSUE_TEMPLATE/` for bug reports and feature requests. Include `LOG_LEVEL=DEBUG` output for bugs.

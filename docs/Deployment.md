# Deployment Guide

## Prerequisites

- Python 3.10, 3.11, or 3.12
- A free [Groq API key](https://console.groq.com/keys)
- Outbound internet access (for DuckDuckGo search)

---

## Local Development

### 1. Clone and configure

```bash
git clone https://github.com/vasu356/research-assistant.git
cd research-assistant
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run

```bash
# Default query
python main.py

# Custom query
python main.py "What is the current state of quantum computing in 2025?"

# Debug logging
LOG_LEVEL=DEBUG python main.py "Your query"
```

---

## Docker

### Build and run

```bash
docker build -t research-assistant .
docker run --env-file .env research-assistant
docker run --env-file .env research-assistant "Your query here"
```

### With volume mount (persist outputs)

```bash
docker run --env-file .env \
  -v $(pwd)/research_outputs:/app/research_outputs \
  research-assistant "Your query"
```

### docker-compose

```bash
docker-compose up
```

To pass a custom query, edit `docker-compose.yml` or override the command:

```bash
docker-compose run research-assistant python main.py "Your query"
```

---

## Environment Variables

See `.env.example` for the complete list. All variables except `GROQ_API_KEY` are optional.

```bash
# Minimal — only required variable
GROQ_API_KEY=gsk_...

# Full example
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.1
MAX_ITERATIONS=6
SEARCH_MAX_RESULTS=6
OUTPUT_DIR=research_outputs
LOG_LEVEL=INFO
```

---

## Running Tests

```bash
# All tests (no network or API key required)
python -m pytest tests/ -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=agents --cov=graph --cov=tools --cov-report=term-missing

# Run directly (alternative runner with coloured output)
python tests/test_end_to_end.py
```

All 38 tests run offline — the LLM and DuckDuckGo tools are fully mocked.

---

## Linting

```bash
pip install ruff
ruff check .         # lint
ruff format --check . # format check
ruff format .        # auto-fix formatting
```

---

## Cloud Deployment Options

### AWS Lambda / GCP Cloud Run

The pipeline is stateless and runs to completion in a single async call, making it suitable for serverless deployment. Approximate p95 latency: 45–90 seconds depending on query complexity.

Key considerations:
- Set a function timeout of at least 120 seconds.
- Set `OUTPUT_DIR` to `/tmp/research_outputs` (writable in Lambda).
- Expose via an API Gateway endpoint accepting `{ "query": "..." }`.

### Scheduled Research Digest

Run on a schedule (cron) to generate regular research reports:

```bash
# cron example: daily at 06:00
0 6 * * * cd /app && python main.py "Daily AI news digest" >> /var/log/research.log 2>&1
```

---

## Troubleshooting

See [Troubleshooting.md](./Troubleshooting.md).

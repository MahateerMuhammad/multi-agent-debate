# Multi-Agent Debate System

A framework for multi-agent reasoning, retrieval, debate, and evaluation using LangGraph and FastAPI.

## Directory Structure

```
multi-agent-debate/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   ├── agents/
│   │   ├── researcher.py
│   │   ├── proponent.py
│   │   ├── opponent.py
│   │   ├── critic.py
│   │   ├── evidence.py
│   │   └── judge.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── judge.py
│   │   └── benchmarks.py
│   ├── tools/
│   │   ├── web_search.py
│   │   └── document_search.py
│   └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
├── experiments/
│   ├── baseline/
│   ├── debate/
│   └── results/
├── frontend/
├── tests/
├── docker/
├── .env.example
├── docker-compose.yml
├── requirements.txt
├── README.md
└── pyproject.toml
```

## Quickstart

1. Copy `.env.example` to `.env` and configure environment variables:
   ```bash
   cp .env.example .env
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```bash
   uvicorn app.api.main:app --reload
   ```

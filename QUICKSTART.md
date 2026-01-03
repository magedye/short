# EasyData Tier-2 Vanna 2.0.1 OSS — Quick Start

## What This Is

A single-file, production-ready FastAPI backend integrating **all official Vanna 2.0.1 tools**:

- `RunSqlTool` — Oracle SQL execution
- `VisualizeDataTool` — Matplotlib code generation
    - `SaveQuestionToolArgsTool` — Q↔SQL pair storage
- `SaveTextMemoryTool` — Generic memory persistence
- ChromaDB memory with persistent storage
- Feedback loop for continuous learning
- Structured assumptions extraction

**No SQL firewall, no auth required by default** — productivity-first.

---

## Installation

### 1. Clone/Copy the Files

```bash
cd /home/mfadmin/short
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn python-dotenv oracledb pandas openai vanna chromadb pydantic
```

Or use the one-liner:

```bash
pip install fastapi uvicorn python-dotenv oracledb pandas openai vanna chromadb pydantic
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env` with your Oracle credentials and OpenAI API key:

```
OPENAI_API_KEY=sk-...
ORACLE_USER=system
ORACLE_PASSWORD=your_pass
ORACLE_DSN=localhost:1521/XEPDB1
```

### 4. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access:
- **API Docs**: http://localhost:8000/docs
- **Root**: http://localhost:8000/

---

## API Quick Reference

### Health Check

```bash
curl http://localhost:8000/api/v2/health
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v2/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total revenue in Q3 2024?"}'
```

**Response** (sealed contract):
```json
{
  "success": true,
  "conversation_id": "tier2-a1b2c3d4e5f6",
  "timestamp": "2026-01-03T12:34:56.789012",
  "question": "What is the total revenue in Q3 2024?",
  "assumptions": [
    {
      "key": "time_scope",
      "value": "Question references a time scope; filtering by date"
    },
    {
      "key": "aggregation",
      "value": "Question requests aggregated data; using GROUP BY and aggregate functions"
    }
  ],
  "sql": "SELECT SUM(amount) FROM orders WHERE created_date >= '2024-07-01' AND created_date < '2024-10-01'",
  "rows": [{"SUM(AMOUNT)": 450000.50}],
  "row_count": 1,
  "chart_code": null,
  "memory_used": true,
  "meta": {"streaming_available": false}
}
```

### Train on Schema

```bash
curl -X POST http://localhost:8000/api/v2/train
```

Train a specific table:

```bash
curl -X POST "http://localhost:8000/api/v2/train?table_name=ORDERS"
```

**Response**:
```json
{
  "success": true,
  "trained": ["ORDERS", "CUSTOMERS", "PRODUCTS"],
  "failed": [],
  "timestamp": "2026-01-03T12:34:56.789012"
}
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v2/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total revenue in Q3 2024?",
    "sql_generated": "SELECT SUM(amount) FROM orders WHERE created_date >= \"2024-07-01\"",
    "sql_corrected": "SELECT SUM(amount) FROM orders WHERE EXTRACT(YEAR FROM created_date) = 2024 AND EXTRACT(QUARTER FROM created_date) = 3",
    "is_correct": false,
    "notes": "Need to use EXTRACT() for Oracle DATE handling"
  }'
```

### Get Agent State

```bash
curl http://localhost:8000/api/v2/state
```

**Response**:
```json
{
  "memory_items_count": 42,
  "trained_tables": ["ORDERS", "CUSTOMERS"],
  "agent_ready": true,
  "llm_connected": true,
  "db_connected": true,
  "timestamp": "2026-01-03T12:34:56.789012"
}
```

---

## Architecture Highlights

### 1. Single File, No Approximations

- **main.py**: All logic, no hidden state or approximations
- Sealed response contracts via Pydantic
- Exact memory counts from ChromaDB, not estimates

### 2. All Official Vanna 2.0.1 Tools Registered

- Tools are discoverable via `ToolRegistry`
- No private API usage (`_get_collection`, etc.)
- Agent can call any registered tool autonomously

### 3. FastAPI Lifespan (Modern Best Practice)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup here
    yield
    # Shutdown here
```

- Replaces deprecated `@app.on_event("startup")`
- Cleaner resource management

### 4. Custom OracleRunner Implementing Vanna Interface

```python
def run_sql(self, sql: str) -> pd.DataFrame:
    """Vanna-compatible interface."""
```

- Integrates seamlessly with `RunSqlTool`
- Sanitizes Oracle legacy encoding
- Handles NaN/Infinity edge cases

### 5. Assumptions + Feedback Loop

- **Assumptions**: Extracted via heuristics during query processing
- **Feedback**: `/api/v2/feedback` stores corrections in memory
- Agent learns from feedback over time

### 6. No Firewall, No Auth by Default

- Productivity-first: Query any table, any SQL
- All endpoints open
- Optional: Add auth/firewall via environment flags if needed

---

## Files Created

```
main.py                 Main application (this is all you need)
.env.example           Environment template
QUICKSTART.md          This file
requirements.txt       (optional) All dependencies listed
```

---

## Docker (Optional)

If you have Docker Compose set up:

```bash
docker-compose up -d
```

(Assumes `docker-compose.yaml` exists; customize as needed.)

---

## Troubleshooting

### "Agent not initialized" (503)

- Check `.env` file is copied and filled in
- Verify OPENAI_API_KEY is valid
- Check Oracle DSN connectivity

### "SQL generation failed"

- Ensure you've trained the schema first: `POST /api/v2/train`
- Check LLM is responding (test OPENAI_API_KEY)
- Review logs for detailed errors

### Memory Not Persisting

- ChromaDB uses `CHROMA_PATH` directory (default: `./vanna_memory`)
- Ensure directory has write permissions
- Check `ls -la vanna_memory/` for collection files

### Unicode Errors from Oracle

- OracleRunner sanitizes cp1252 fallback automatically
- Check `LOG_LEVEL=DEBUG` for detailed sanitization logs

---

## Next Steps

1. Train your schema: `POST /api/v2/train`
2. Ask a question: `POST /api/v2/ask`
3. Provide feedback on incorrect SQL: `POST /api/v2/feedback`
4. Monitor with `/api/v2/state` and `/api/v2/health`

---

## Notes

- **Streaming Mode** (`STREAMING_MODE=true`): Pseudo-streams via NDJSON (each stage as a JSON line)
- **MAX_ROWS**: Hard limit on query results (default: 1000)
- **Feedback Integration**: Corrections are stored in ChromaDB and influence future generations
- **No Approximations**: State endpoints return exact counts, not estimates

Enjoy your productivity-first Vanna integration!

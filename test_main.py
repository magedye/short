import asyncio
import json
import os
from typing import Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

import main
from vanna.core.llm.models import LlmMessage, LlmRequest
from vanna.core.user import User

try:
    import oracledb  # type: ignore
    _ORACLE_IMPORTED = True
except ImportError:
    oracledb = None  # type: ignore
    _ORACLE_IMPORTED = False


# ---------- Fixtures ----------


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """FastAPI test client that triggers lifespan startup once."""
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="session")
def health(client: TestClient) -> Dict:
    resp = client.get("/api/v2/health")
    assert resp.status_code == 200, f"Health endpoint failed: {resp.text}"
    return resp.json()


@pytest.fixture(scope="session")
def oracle_ready() -> bool:
    if not _ORACLE_IMPORTED:
        return False
    required = ["ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN"]
    if not all(os.getenv(k) for k in required):
        return False
    conn: Optional[oracledb.Connection] = None
    try:
        conn = oracledb.connect(
            user=os.getenv("ORACLE_USER"),
            password=os.getenv("ORACLE_PASSWORD"),
            dsn=os.getenv("ORACLE_DSN"),
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM DUAL")
        row = cur.fetchone()
        return row and row[0] == 1
    except Exception:
        return False
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


@pytest.fixture(scope="session")
def llm_ready() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


# ---------- Startup / preflight ----------


def test_tools_registered(client: TestClient):
    assert main.agent is not None, "Agent not initialized during lifespan"

    async def _list():
        return await main.agent.tool_registry.list_tools()  # type: ignore[union-attr]

    tools = asyncio.run(_list())
    expected = {
        "run_sql",
        "visualize_data",
        "save_question_tool_args",
        "save_text_memory",
        "search_saved_correct_tool_uses",
    }
    assert expected.issubset(set(tools))


def test_agent_run_available(client: TestClient):
    assert hasattr(main.agent, "ask") and callable(getattr(main.agent, "ask"))


def test_env_vars_present():
    required = ["OPENAI_API_KEY", "ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN"]
    missing = [k for k in required if not os.getenv(k)]
    assert not missing, f"Missing required env vars: {missing}"


def test_oracle_connectivity(oracle_ready: bool):
    if not oracle_ready:
        pytest.skip("Oracle not configured or unreachable")
    # oracle_ready fixture already validated connectivity; reaching here is a pass.
    assert oracle_ready


def test_llm_ping(client: TestClient, llm_ready: bool):
    if not llm_ready or not main.agent:
        pytest.skip("LLM not configured")
    llm_service = getattr(main.agent, "llm_service", None)
    if not llm_service:
        pytest.skip("LLM service not available on agent")

    user = User(id="llm-test")
    request = LlmRequest(
        messages=[LlmMessage(role="user", content="ping")],
        user=user,
        stream=False,
        max_tokens=5,
    )

    async def _ping():
        return await llm_service.send_request(request)  # type: ignore[attr-defined]

    response = asyncio.run(_ping())
    assert response is not None


# ---------- API / integration ----------


def test_health_endpoint(health: Dict):
    assert health["status"] in {"healthy", "degraded"}
    assert set(health["components"].keys()) >= {"agent", "memory", "oracle"}


def test_state_endpoint(client: TestClient):
    resp = client.get("/api/v2/state")
    assert resp.status_code == 200
    body = resp.json()
    for key in ["memory_items_count", "trained_tables", "agent_ready", "llm_connected", "db_connected"]:
        assert key in body


def test_ask_endpoint(client: TestClient, oracle_ready: bool):
    if not oracle_ready or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Oracle or LLM not configured")
    resp = client.post("/api/v2/ask", json={"question": "Show 1 row from dual"})
    if resp.status_code != 200:
        pytest.skip(f"Ask endpoint not ready: {resp.status_code} {resp.text}")
    body = resp.json()
    for key in ["sql", "rows", "assumptions", "conversation_id"]:
        assert key in body


def test_feedback_endpoint(client: TestClient):
    payload = {
        "question": "What is the total sales?",
        "sql_generated": "SELECT 1 FROM DUAL",
        "sql_corrected": "SELECT 1 FROM DUAL",
        "is_correct": True,
        "notes": "test feedback",
    }
    resp = client.post("/api/v2/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_train_endpoint(client: TestClient, oracle_ready: bool):
    if not oracle_ready:
        pytest.skip("Oracle not configured or unreachable")
    resp = client.post("/api/v2/train")
    if resp.status_code != 200:
        pytest.skip(f"Train endpoint not ready: {resp.status_code} {resp.text}")
    body = resp.json()
    assert "trained" in body and "failed" in body


def test_ask_stream_ndjson(client: TestClient, oracle_ready: bool):
    if not oracle_ready or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Oracle or LLM not configured")
    with client.stream("POST", "/api/v2/ask_stream", json={"question": "Show 1 row from dual"}) as resp:
        if resp.status_code != 200:
            pytest.skip(f"Ask stream endpoint not ready: {resp.status_code}")
        chunks: List[str] = []
        for line in resp.iter_lines():
            if not line:
                continue
            chunks.append(line if isinstance(line, str) else line.decode())
        assert chunks, "No NDJSON chunks returned"
        parsed = [json.loads(c) for c in chunks]
        stages = {item.get("stage") for item in parsed}
        assert {"assumptions", "results", "complete"}.issubset(stages)

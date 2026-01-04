import pytest
from unittest.mock import MagicMock

from main import generate_safe_sql, create_request_context, agent, vn


@pytest.fixture
def dummy_context():
    return create_request_context(conversation_id="test-convo")


def test_agentic_success(monkeypatch, dummy_context):
    """Agentic returns valid SQL → no fallback expected."""
    mock_response = MagicMock()
    mock_response.sql = "SELECT * FROM TRANSACTS_T2"
    monkeypatch.setattr(agent, "ask", lambda question, context: mock_response)

    sql = generate_safe_sql("give me 10 records", dummy_context)
    assert sql is not None
    assert "DUAL" not in sql.upper()
    assert "FROM TRANSACTS_T2" in sql


def test_agentic_returns_dummy_fallbacks_to_legacy(monkeypatch, dummy_context):
    """Agentic returns dummy SQL → fallback to Legacy should trigger."""
    mock_response = MagicMock()
    mock_response.sql = "SELECT 1 FROM DUAL"
    monkeypatch.setattr(agent, "ask", lambda question, context: mock_response)
    monkeypatch.setattr(vn, "generate_sql", lambda question: "SELECT * FROM FALLBACK_TABLE")

    sql = generate_safe_sql("list fallback data", dummy_context)
    assert sql == "SELECT * FROM FALLBACK_TABLE"


def test_agentic_exception_fallbacks_to_legacy(monkeypatch, dummy_context):
    """Agentic raises Exception → fallback to Legacy."""
    monkeypatch.setattr(agent, "ask", lambda question, context: 1 / 0)
    monkeypatch.setattr(vn, "generate_sql", lambda question: "SELECT * FROM LEGACY_OK")

    sql = generate_safe_sql("simulate error", dummy_context)
    assert sql == "SELECT * FROM LEGACY_OK"


def test_both_paths_fail(monkeypatch, dummy_context):
    """Both Agentic and Legacy fail → SQL should be None."""
    monkeypatch.setattr(agent, "ask", lambda question, context: 1 / 0)
    monkeypatch.setattr(vn, "generate_sql", lambda question: None)

    sql = generate_safe_sql("fail everything", dummy_context)
    assert sql is None

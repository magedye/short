import json
from pathlib import Path

import sqlglot


class ValidationError(Exception):
    """Raised when SQL fails policy validation."""


class SQLGatekeeper:
    """Policy-based SQL validator using SQLGlot parsing and memory-based ACL."""

    def __init__(self, vn, config_path: str = "config/gatekeeper_config.json"):
        self.vn = vn
        with Path(config_path).open() as f:
            self.cfg = json.load(f)["validation"]

    def validate(self, sql: str) -> str:
        sql = sql.strip().rstrip(";")

        # Oracle-only guardrails
        if any(ch >= "\u0600" and ch <= "\u06FF" for ch in sql):
            raise ValidationError("Oracle SQL must use canonical (non-Arabic) identifiers.")

        if len(sql) > self.cfg["max_query_length"]:
            raise ValidationError("SQL too long")

        upper = sql.upper()
        if " LIMIT " in upper or upper.endswith("LIMIT") or upper.startswith("LIMIT"):
            raise ValidationError("LIMIT is not supported by Oracle. Use FETCH FIRST.")

        for kw in self.cfg["forbidden_keywords"]:
            if kw in upper:
                raise ValidationError(f"Forbidden keyword: {kw}")

        if self.cfg.get("require_where_or_limit", False):
            if "WHERE" not in upper and "LIMIT" not in upper and "FETCH" not in upper:
                raise ValidationError("Missing WHERE or LIMIT")

        try:
            expr = sqlglot.parse_one(sql, read="oracle")
        except Exception as e:
            raise ValidationError(f"Invalid SQL syntax: {e}")

        # Table whitelist based on memory contents (sql_collection)
        used_tables = {
            t.name.upper()
            for t in expr.find_all(sqlglot.expressions.Table)
            if getattr(t, "name", None)
        }
        allowed_tables = get_allowed_tables_from_memory(self.vn)
        unknown = used_tables - allowed_tables
        if unknown:
            raise ValidationError(f"Access denied to table(s): {', '.join(sorted(unknown))}")

        return sql


def get_allowed_tables_from_memory(vn) -> set[str]:
    """
    Tables that appeared in stored SQL = allowed.
    Memory is the only source of truth.
    """
    tables: set[str] = set()
    try:
        data = vn.sql_collection.get(include=["documents"])  # type: ignore[attr-defined]
    except Exception:
        return tables

    docs = data.get("documents", []) if data else []
    for doc in docs:
        if not doc:
            continue
        try:
            payload = json.loads(doc) if isinstance(doc, str) else doc
            sql = payload.get("sql") or payload.get("text") or payload.get("content")
        except Exception:
            sql = None
        if not sql:
            continue
        try:
            parsed = sqlglot.parse_one(sql, read="oracle")
            for t in parsed.find_all(sqlglot.exp.Table):
                if getattr(t, "name", None):
                    tables.add(t.name.upper())
        except Exception:
            continue
    return tables

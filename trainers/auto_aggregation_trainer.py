"""
Auto Aggregation Trainer (GROUP BY)
-----------------------------------
Generates deterministic aggregation examples (COUNT/SUM) per table based on DDL
column metadata. Oracle-only. Skips tables without column metadata.
"""

import os
from typing import Dict, List

import chromadb
from dotenv import load_dotenv
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.openai import OpenAI_Chat

load_dotenv(dotenv_path=".env")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

NUMERIC_TYPES = {"NUMBER", "INTEGER", "FLOAT", "DECIMAL"}
TEXT_TYPES = {"VARCHAR2", "VARCHAR", "CHAR"}
DATE_TYPES = {"DATE", "TIMESTAMP"}


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


def get_vn():
    config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "path": os.getenv("CHROMA_PATH", "./vanna_memory"),
        "collection_name": os.getenv("CHROMA_COLLECTION", "tier2_vanna"),
    }
    return MyVanna(config=config)


def extract_table_columns() -> Dict[str, List[Dict]]:
    """Pull column metadata from DDL metadatas (requires columns listed there)."""
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./vanna_memory"))
    coll = client.get_collection(os.getenv("CHROMA_COLLECTION", "tier2_vanna"))
    res = coll.get(include=["metadatas"])

    tables: Dict[str, List[Dict]] = {}
    for meta in res.get("metadatas", []):
        if not meta or meta.get("type") != "ddl":
            continue
        table = meta.get("table")
        cols = meta.get("columns") or []
        if table and cols:
            tables[table.upper()] = cols
    return tables


def train_aggregations():
    vn = get_vn()
    tables = extract_table_columns()
    if not tables:
        print("⚠️ No column metadata in DDL; aggregation trainer skipped.")
        return

    total = 0
    for table, columns in tables.items():
        numeric_cols = []
        group_cols = []
        for c in columns:
            ctype = c.get("type", "").upper()
            name = c.get("name", "").upper()
            if ctype in NUMERIC_TYPES:
                numeric_cols.append(name)
            if ctype in TEXT_TYPES or ctype in DATE_TYPES or c.get("key") == "foreign":
                group_cols.append(name)

        for g in group_cols:
            # COUNT
            sql_count = (
                f"SELECT {g}, COUNT(*) AS CNT FROM {table} "
                f"GROUP BY {g} FETCH FIRST 5 ROWS ONLY"
            )
            for q in [f"عدد السجلات لكل {g}", f"count records per {g}"]:
                vn.train(question=q, sql=sql_count)
                total += 1

            # SUM on numeric columns
            for n in numeric_cols:
                sql_sum = (
                    f"SELECT {g}, SUM({n}) AS TOTAL_{n} FROM {table} "
                    f"GROUP BY {g} FETCH FIRST 5 ROWS ONLY"
                )
                for q in [f"إجمالي {n} حسب {g}", f"total {n} by {g}"]:
                    vn.train(question=q, sql=sql_sum)
                    total += 1

        print(f"✅ Aggregations trained for table: {table}")

    print(f"🎯 Done. Trained {total} aggregation Q&A pairs.")


if __name__ == "__main__":
    train_aggregations()

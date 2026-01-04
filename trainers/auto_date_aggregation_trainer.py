"""
Auto Date Aggregation Trainer
-----------------------------
Generates date-based aggregations (daily/monthly/yearly) from DDL metadata.
Oracle-only. No LLM. Skips tables without DATE/TIMESTAMP columns.
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

DATE_TYPES = {"DATE", "TIMESTAMP"}
NUMERIC_TYPES = {"NUMBER", "INTEGER", "FLOAT", "DECIMAL"}


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


def train_date_aggregations():
    vn = get_vn()
    tables = extract_table_columns()
    if not tables:
        print("⚠️ No column metadata in DDL; date aggregation trainer skipped.")
        return

    total = 0
    for table, columns in tables.items():
        date_cols = []
        numeric_cols = []
        for c in columns:
            t = c.get("type", "").upper()
            n = c.get("name", "").upper()
            if t in DATE_TYPES:
                date_cols.append(n)
            if t in NUMERIC_TYPES:
                numeric_cols.append(n)

        if not date_cols:
            continue

        for d in date_cols:
            for level, trunc in [
                ("daily", f"TRUNC({d})"),
                ("monthly", f"TRUNC({d}, 'MM')"),
                ("yearly", f"TRUNC({d}, 'YYYY')"),
            ]:
                sql_count = (
                    f"SELECT {trunc} AS PERIOD, COUNT(*) AS CNT FROM {table} "
                    f"GROUP BY {trunc} FETCH FIRST 5 ROWS ONLY"
                )
                for q in [f"عدد السجلات {level}", f"{level} count of records"]:
                    vn.train(question=q, sql=sql_count)
                    total += 1

                for n in numeric_cols:
                    sql_sum = (
                        f"SELECT {trunc} AS PERIOD, SUM({n}) AS TOTAL_{n} FROM {table} "
                        f"GROUP BY {trunc} FETCH FIRST 5 ROWS ONLY"
                    )
                    for q in [f"إجمالي {n} {level}", f"{level} total {n}"]:
                        vn.train(question=q, sql=sql_sum)
                        total += 1

        if date_cols:
            print(f"✅ Date aggregations trained for table: {table}")

    print(f"🎯 Done. Trained {total} date aggregation Q&A pairs.")


if __name__ == "__main__":
    train_date_aggregations()

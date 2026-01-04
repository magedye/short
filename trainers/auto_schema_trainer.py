"""
Auto Schema Q&A Trainer (DDL-driven)
-----------------------------------
Generates deterministic schema questions (columns) for each table present in DDL
stored in ChromaDB. No LLM, no guessing. Oracle-only.
"""

import os
from typing import Set, List

import chromadb
from dotenv import load_dotenv
from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.openai import OpenAI_Chat

load_dotenv(dotenv_path=".env")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"


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


def get_tables_from_ddl() -> Set[str]:
    """Extract table names from DDL metadata only."""
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./vanna_memory"))
    coll = client.get_collection(os.getenv("CHROMA_COLLECTION", "tier2_vanna"))
    res = coll.get(include=["documents", "metadatas"])
    tables: Set[str] = set()
    for meta in res.get("metadatas", []):
        if meta and meta.get("type") == "ddl" and meta.get("table"):
            tables.add(meta["table"].upper())
    return tables


def build_sql(table: str) -> str:
    """Oracle-only column introspection SQL."""
    return (
        "SELECT column_name, data_type, data_length, nullable "
        f"FROM USER_TAB_COLUMNS WHERE table_name = '{table}' ORDER BY column_id"
    )


def build_questions(table: str) -> List[str]:
    return [
        f"اعرض أعمدة جدول {table}",
        f"ما هي أعمدة جدول {table}",
        f"ما بنية جدول {table}",
        f"show columns of {table}",
        f"list columns of {table}",
        f"describe table {table}",
    ]


def train_schema_questions():
    vn = get_vn()
    tables = get_tables_from_ddl()
    if not tables:
        print("⚠️ No DDL tables found. Skipping.")
        return

    total = 0
    for table in tables:
        sql = build_sql(table)
        for q in build_questions(table):
            vn.train(question=q, sql=sql)
            total += 1
        print(f"✅ Schema Q&A trained for table: {table}")

    print(f"🎯 Done. Trained {total} schema Q&A pairs for {len(tables)} table(s).")


if __name__ == "__main__":
    train_schema_questions()

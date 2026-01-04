"""
Auto JOIN Trainer (DDL-driven)
------------------------------
Generates deterministic JOIN examples from foreign key relations present in DDL.
No LLM, no guessing. Oracle-only. Skips if no FK info is present.
"""

import os
import re
from typing import List, Dict

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


def extract_fk_relations() -> List[Dict[str, str]]:
    """
    Extract FK relations either from metadatas.foreign_keys or by parsing DDL text.
    """
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./vanna_memory"))
    coll = client.get_collection(os.getenv("CHROMA_COLLECTION", "tier2_vanna"))
    res = coll.get(include=["documents", "metadatas"])

    relations: List[Dict[str, str]] = []
    for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
        if not meta or meta.get("type") != "ddl":
            continue
        table = (meta.get("table") or "").upper()
        if not table:
            continue

        # Preferred: structured foreign_keys
        for fk in meta.get("foreign_keys", []) or []:
            relations.append(
                {
                    "from_table": table,
                    "from_column": fk.get("column", "").upper(),
                    "to_table": fk.get("ref_table", "").upper(),
                    "to_column": fk.get("ref_column", "").upper(),
                }
            )

        # Fallback: parse raw DDL
        if isinstance(doc, str):
            matches = re.findall(
                r"FOREIGN KEY\\s*\\((\\w+)\\)\\s*REFERENCES\\s*(\\w+)\\s*\\((\\w+)\\)",
                doc,
                flags=re.IGNORECASE,
            )
            for col, ref_table, ref_col in matches:
                relations.append(
                    {
                        "from_table": table,
                        "from_column": col.upper(),
                        "to_table": ref_table.upper(),
                        "to_column": ref_col.upper(),
                    }
                )

    # Deduplicate
    dedup = {}
    for r in relations:
        key = (
            r.get("from_table"),
            r.get("from_column"),
            r.get("to_table"),
            r.get("to_column"),
        )
        if all(key):
            dedup[key] = r
    return list(dedup.values())


def build_questions(a: str, b: str) -> List[str]:
    return [
        f"اعرض بيانات {a} مع {b}",
        f"اعرض {a} مع معلومات {b}",
        f"join {a} with {b}",
        f"list {a} with {b} details",
        f"show {a} along with {b}",
    ]


def train_join_examples():
    vn = get_vn()
    relations = extract_fk_relations()
    if not relations:
        print("⚠️ No FK relations found. Skipping.")
        return

    total = 0
    for r in relations:
        a = r["from_table"]
        b = r["to_table"]
        a_col = r["from_column"]
        b_col = r["to_column"]
        if not all([a, b, a_col, b_col]):
            continue

        sql = (
            f"SELECT A.*, B.* FROM {a} A JOIN {b} B "
            f"ON A.{a_col} = B.{b_col} FETCH FIRST 5 ROWS ONLY"
        )
        for q in build_questions(a, b):
            vn.train(question=q, sql=sql)
            total += 1
        print(f"✅ JOIN trained: {a} ↔ {b}")

    print(f"🎯 Done. Trained {total} JOIN Q&A pairs.")


if __name__ == "__main__":
    train_join_examples()

"""
Minimal question→SQL training to seed Vanna legacy memory (Oracle-only).

Usage:
    source ./venv/bin/activate
    python train_examples.py

Effect:
    - Adds a few Q/SQL pairs to the Chroma SQL collection.
    - Enables gatekeeper to authorize the referenced tables.

Notes:
    - Does NOT execute SQL against Oracle; it just stores the mapping.
    - Update TABLE_NAME below to match real Oracle tables you have DDL for.
"""

import os
from dotenv import load_dotenv
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


def main() -> int:
    load_dotenv()

    table = os.getenv("TRAIN_TABLE", "TRANSACTS_T2")

    config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "path": os.getenv("CHROMA_PATH", "./vanna_memory"),
        "collection_name": os.getenv("CHROMA_COLLECTION", "easydata_memory"),
    }

    vn = MyVanna(config=config)

    examples = [
        (
            "Show the first 5 rows from the main table",
            f"SELECT * FROM {table} FETCH FIRST 5 ROWS ONLY",
        ),
        (
            "اعرض أول خمسة صفوف من الجدول الرئيسي",
            f"SELECT * FROM {table} FETCH FIRST 5 ROWS ONLY",
        ),
        (
            "List latest 5 records",
            f"SELECT * FROM {table} ORDER BY 1 DESC FETCH FIRST 5 ROWS ONLY",
        ),
    ]

    for q, s in examples:
        vn.train(question=q, sql=s)
        print(f"✓ Trained: {q} -> {s}")

    print("\nDone. Tables referenced are now authorized via memory ACL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

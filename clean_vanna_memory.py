import os
import re

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./vanna_memory")
COLLECTION = os.getenv("CHROMA_SQL_COLLECTION", "sql")

client = chromadb.PersistentClient(path=CHROMA_PATH)
coll = client.get_collection(COLLECTION)

res = coll.get(include=["documents", "metadatas"], limit=1000)
ids = res.get("ids", [])
docs = res.get("documents", [])
metas = res.get("metadatas", [])

to_delete = []

FORBIDDEN_PATTERNS = [
    r"\bSELECT\s+1\b",
    r"\bDUAL\b",
    r"\bLIMIT\b",
    r"TABLE_A",
    r"TABLE_B",
    r"TRANSACT_AMNT",
    r"\bAMOUNT\b",
]


def has_from(sql: str) -> bool:
    return bool(re.search(r"\bFROM\b", sql, re.IGNORECASE))


for _id, doc, meta in zip(ids, docs, metas):
    if not doc or not isinstance(doc, str):
        continue

    # Never touch DDL
    if meta and meta.get("type") == "ddl":
        continue

    # Drop auto-training markers or any marker-only docs
    if doc.strip().lower() == "marker" or (meta and meta.get("type") == "auto_semantic_training"):
        to_delete.append(_id)
        continue

    upper = doc.upper()

    # Dummy or forbidden patterns
    if any(re.search(p, upper) for p in FORBIDDEN_PATTERNS):
        to_delete.append(_id)
        continue

    # SELECT without FROM
    if "SELECT" in upper and not has_from(upper):
        to_delete.append(_id)
        continue

if to_delete:
    coll.delete(ids=to_delete)
    print(f"🧹 Deleted {len(to_delete)} polluted memory entries")
else:
    print("✅ Memory is clean — no polluted entries found")

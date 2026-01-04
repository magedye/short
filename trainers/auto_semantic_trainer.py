"""
Auto Semantic Trainer
---------------------
Generates safe SQL training examples from metadata (DDL-driven).
No hallucination. Oracle-only syntax (FETCH FIRST ...).
"""

from typing import List, Dict

from config.metadata_loader import metadata

FETCH_N = 5


def _date_columns(table_spec: Dict) -> List[str]:
    return [
        c["name"]
        for c in table_spec.get("columns", [])
        if c.get("type", "").upper() in ("DATE", "TIMESTAMP")
    ]


def _numeric_columns(table_spec: Dict) -> List[str]:
    return [
        c["name"]
        for c in table_spec.get("columns", [])
        if "NUMBER" in c.get("type", "").upper()
    ]


def generate_training_pairs() -> List[Dict[str, str]]:
    """
    Returns a list of {question, sql} pairs based purely on metadata.
    """
    pairs: List[Dict[str, str]] = []

    for table_name, table_spec in metadata.tables.items():
        date_cols = _date_columns(table_spec)
        num_cols = _numeric_columns(table_spec)

        # 1) Latest records (DATE-based)
        if date_cols:
            col = date_cols[0]
            pairs.extend(
                [
                    {
                        "question": f"list latest records from {table_name}",
                        "sql": (
                            f"SELECT * FROM {table_name} "
                            f"ORDER BY {col} DESC FETCH FIRST {FETCH_N} ROWS ONLY"
                        ),
                    },
                    {
                        "question": f"اعرض آخر سجلات {table_name}",
                        "sql": (
                            f"SELECT * FROM {table_name} "
                            f"ORDER BY {col} DESC FETCH FIRST {FETCH_N} ROWS ONLY"
                        ),
                    },
                ]
            )

        # 2) Top numeric values
        if num_cols:
            col = num_cols[0]
            pairs.extend(
                [
                    {
                        "question": f"list top records by {col} from {table_name}",
                        "sql": (
                            f"SELECT * FROM {table_name} "
                            f"ORDER BY {col} DESC FETCH FIRST {FETCH_N} ROWS ONLY"
                        ),
                    },
                    {
                        "question": f"اعرض أعلى القيم حسب {col} من {table_name}",
                        "sql": (
                            f"SELECT * FROM {table_name} "
                            f"ORDER BY {col} DESC FETCH FIRST {FETCH_N} ROWS ONLY"
                        ),
                    },
                ]
            )

        # 3) Generic first rows
        pairs.extend(
            [
                {
                    "question": f"show first rows from {table_name}",
                    "sql": f"SELECT * FROM {table_name} FETCH FIRST {FETCH_N} ROWS ONLY",
                },
                {
                    "question": f"اعرض أول صفوف من {table_name}",
                    "sql": f"SELECT * FROM {table_name} FETCH FIRST {FETCH_N} ROWS ONLY",
                },
            ]
        )

    return pairs

from config.metadata_loader import metadata


def build_prompt() -> str:
    lines = [
        "You are a SQL expert assistant.",
        "Generate SQL using ONLY the schema below.",
        "",
        "KNOWN TABLES:",
    ]

    for table, spec in metadata.tables.items():
        cols = ", ".join(c["name"] for c in spec.get("columns", []))
        lines.append(f"- {table} ({cols})")

    if metadata.relationships:
        lines.append("")
        lines.append("RELATIONSHIPS:")
        for rel in metadata.relationships:
            lines.append(f"- {rel['from']} = {rel['to']}")

    lines.extend(
        [
            "",
            "RULES:",
            "1. Never invent table or column names.",
            "2. Use only listed tables.",
            "3. Always include WHERE or LIMIT.",
            "4. Return SQL only.",
        ]
    )

    return "\n".join(lines)


PROMPT_SYSTEM = build_prompt()

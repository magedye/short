"""
Quick LLM connectivity check for OpenAI-compatible endpoints (Groq).

Usage:
    python verify_llm.py

Reads env:
    OPENAI_API_KEY
    OPENAI_BASE_URL (default: https://api.openai.com/v1)
    OPENAI_MODEL    (default: gpt-4o-mini)
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("❌ OPENAI_API_KEY is not set")
        return 1

    print(f"🔗 Testing LLM connectivity: base_url={base_url}, model={model}")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
            temperature=0.0,
        )
        choice = resp.choices[0].message.content
        print(f"✅ LLM responded: {choice!r}")
        return 0
    except Exception as e:
        print(f"❌ LLM connectivity failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

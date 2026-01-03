
# ===============================================
# EasyData Tier-2 UI — Streamlit (Final Version)
# Fully Compatible with main.py (100%)
# ===============================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# ===============================================
# Logging
# ===============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EasyData-UI")

# ===============================================
# Configuration
# ===============================================

BACKEND_URL = os.getenv("BACKEND_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BACKEND_URL}/api/v2"
API_KEY = os.getenv("TIER2_ACCESS_KEY")

# Session-based timeout (dynamic)
if "request_timeout" not in st.session_state:
    st.session_state.request_timeout = 30

# ===============================================
# Helpers
# ===============================================

def _headers() -> Dict[str, str]:
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def ask_question(question: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            headers=_headers(),
            timeout=st.session_state.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def execute_chart_code(chart_code: str):
    try:
        plt.clf()
        safe_globals = {
            "__builtins__": {},
            "plt": plt,
            "pd": pd,
        }
        exec(chart_code, safe_globals)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning("⚠️ Failed to render chart. Showing code instead.")
        st.code(chart_code, language="python")
        st.error(str(e))


# ===============================================
# Page Setup
# ===============================================

st.set_page_config(
    page_title="EasyData Tier-2",
    layout="wide",
)

st.title("🤖 EasyData Tier-2")
st.caption("Vanna 2.0.1 | Oracle | Agentic SQL")

# ===============================================
# Sidebar — Settings
# ===============================================

with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.request_timeout = st.number_input(
        "Request Timeout (seconds)",
        min_value=5,
        max_value=300,
        value=st.session_state.request_timeout,
    )

    st.markdown("---")
    st.caption(f"Backend: `{BACKEND_URL}`")

# ===============================================
# Chat State
# ===============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================================
# Display Chat History
# ===============================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        payload = msg.get("payload")
        if msg["role"] == "assistant" and payload:

            if payload.get("assumptions"):
                with st.expander("🔍 Assumptions"):
                    for a in payload["assumptions"]:
                        st.markdown(f"- **{a['key']}**: {a['value']}")

            if payload.get("sql"):
                st.markdown("**🧾 Generated SQL**")
                st.code(payload["sql"], language="sql")

            if payload.get("rows"):
                df = pd.DataFrame(payload["rows"])
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False)
                ts = payload.get("timestamp", datetime.utcnow().isoformat())[:10]
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name=f"results_{ts}.csv",
                    mime="text/csv",
                )

            if payload.get("chart_code"):
                with st.expander("📈 Visualization"):
                    execute_chart_code(payload["chart_code"])

            if payload.get("meta"):
                with st.expander("ℹ️ Meta Info"):
                    st.json(payload["meta"])

            if payload.get("memory_used"):
                st.caption("🧠 Memory-assisted answer")

            if payload.get("error"):
                st.error(payload["error"])

# ===============================================
# Chat Input
# ===============================================

user_input = st.chat_input("Ask a question about your Oracle data...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(user_input)

            if not response:
                st.error("No response from backend.")
            else:
                text = (
                    f"✓ Found **{response.get('row_count', 0)}** rows"
                    if response.get("success")
                    else f"❌ {response.get('error')}"
                )
                st.markdown(text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": text,
                    "payload": response,
                })

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   EasyData Tier-2 UI — Streamlit                            ║
║                                                                              ║
║  Modern, interactive, production-ready interface for Tier-2 backend        ║
║  - Real-time conversation with Oracle database                             ║
║  - Schema management and training                                          ║
║  - Memory visualization and management                                     ║
║  - Performance metrics and health monitoring                               ║
║  - Docker-compatible (environment variable configuration)                  ║
║  - Full Arabic & English support                                           ║
║                                                                              ║
║  Run: streamlit run ui.py                                                  ║
║  Docker: BACKEND_SERVICE_URL=http://backend:8000 streamlit run ui.py       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import time
import os
import logging

# ==================================================================================
# 1. LOGGING SETUP
# ==================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Streamlit-UI")

# ==================================================================================
# 2. CONFIGURATION (DOCKER-COMPATIBLE)
# ==================================================================================

# CRITICAL: Get Backend URL from environment variable (for Docker networking)
# Local: http://127.0.0.1:8000 (مطابق للبورت في الملف السابق)
# Docker: http://backend:8000
# Production: https://api.easydata.example.com
BACKEND_URL = os.getenv("BACKEND_SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BACKEND_URL}/api/v2"
API_KEY = os.getenv("TIER2_ACCESS_KEY")
DEFAULT_TIMEOUT = 30

logger.info(f"Backend URL configured: {BACKEND_URL}")

# ==================================================================================
# 3. PAGE CONFIGURATION
# ==================================================================================

st.set_page_config(
    page_title="EasyData Tier-2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/easydata/tier2",
        "Report a bug": "https://github.com/easydata/tier2/issues",
        "About": "EasyData Tier-2 | Vanna 2.0.1 | Production Ready"
    }
)

# ==================================================================================
# 4. CUSTOM CSS (Arabic Support + Professional Styling)
# ==================================================================================

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
        
        * {
            font-family: 'Cairo', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Chat Messages */
        .stChatMessage {
            border-radius: 12px;
            padding: 16px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            animation: slideIn 0.3s ease-in-out;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* SQL Code Block */
        .sql-block {
            background-color: #f0f2f6;
            border-left: 4px solid #0066cc;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        /* Success Badge */
        .success-badge {
            background-color: #d4edda;
            color: #155724;
            padding: 8px 12px;
            border-radius: 6px;
            display: inline-block;
            border-left: 4px solid #28a745;
            margin: 5px 0;
        }
        
        /* Error Badge */
        .error-badge {
            background-color: #f8d7da;
            color: #721c24;
            padding: 8px 12px;
            border-radius: 6px;
            display: inline-block;
            border-left: 4px solid #dc3545;
            margin: 5px 0;
        }
        
        /* Info Badge */
        .info-badge {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 8px 12px;
            border-radius: 6px;
            display: inline-block;
            border-left: 4px solid #17a2b8;
            margin: 5px 0;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        /* Title Styling */
        h1 { color: #1f77b4; }
        h2 { color: #2ca02c; }
        h3 { color: #555; }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ==================================================================================
# 5. SESSION STATE INITIALIZATION
# ==================================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

if "training_status" not in st.session_state:
    st.session_state.training_status = None

if "health_status" not in st.session_state:
    st.session_state.health_status = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "backend_available" not in st.session_state:
    st.session_state.backend_available = False

# ==================================================================================
# 6. UTILITY FUNCTIONS
# ==================================================================================

def _auth_headers() -> Dict[str, str]:
    """Build auth headers if API key is configured."""
    headers: Dict[str, str] = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers

def check_backend_health() -> Optional[Dict[str, Any]]:
    """
    Check if backend is running and ready.
    
    Returns:
        Health status dict or None if unreachable
    """
    try:
        logger.info(f"Checking health: {API_URL}/health")
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
            headers=_auth_headers(),
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.backend_available = True
            logger.info("✓ Backend is healthy")
            return data
        else:
            st.session_state.backend_available = False
            logger.warning(f"Backend returned {response.status_code}")
            return None
    except requests.Timeout:
        st.session_state.backend_available = False
        logger.error("Backend health check timeout")
        return None
    except Exception as e:
        st.session_state.backend_available = False
        st.session_state.last_error = str(e)
        logger.error(f"Backend unreachable: {e}")
        return None


def get_agent_state() -> Optional[Dict[str, Any]]:
    """Fetch current agent state from backend."""
    try:
        logger.info(f"Fetching state: {API_URL}/state")
        response = requests.get(
            f"{API_URL}/state",
            timeout=10,
            headers=_auth_headers(),
        )
        if response.status_code == 200:
            data = response.json()
            logger.info("✓ Agent state fetched")
            return data
    except Exception as e:
        logger.error(f"Failed to fetch agent state: {e}")
    return None


def train_schema() -> Optional[Dict[str, Any]]:
    """Trigger schema training."""
    try:
        logger.info(f"Starting training: {API_URL}/train")
        response = requests.post(
            f"{API_URL}/train",
            timeout=120,
            headers=_auth_headers(),
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Training completed: {len(data.get('trained', []))} tables")
            return data
    except requests.Timeout:
        error_msg = "Training timeout: operation took too long"
        logger.error(error_msg)
        st.error(f"⏱️ {error_msg}")
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
    return None


def ask_question(question: str, context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Send question to backend and get response.
    
    Args:
        question: User's natural language question
        context: Optional context data
        
    Returns:
        Backend response or None if failed
    """
    try:
        logger.info(f"Asking question: {question[:50]}...")
        payload = {
            "question": question,
            "context": context or {}
        }
        response = requests.post(
            f"{API_URL}/ask",
            json=payload,
            timeout=DEFAULT_TIMEOUT,
            headers=_auth_headers(),
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Response received: {data.get('row_count', 0)} rows")
            return data
        else:
            error_msg = f"Backend error: {response.status_code}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
    except requests.Timeout:
        error_msg = "⏱️ Request timeout. Try a simpler question or check backend health."
        logger.error("Request timeout")
        st.error(error_msg)
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
    return None


def submit_feedback(
    conversation_id: str,
    question: str,
    sql: str,
    correct: bool,
    corrected_sql: Optional[str] = None,
    notes: Optional[str] = None
) -> bool:
    """Submit feedback for continuous learning."""
    try:
        logger.info(f"Submitting feedback for conversation: {conversation_id}")
        payload = {
            "question": question,
            "sql_generated": sql,
            "sql_corrected": corrected_sql,
            "is_correct": correct,
            "notes": notes
        }
        response = requests.post(
            f"{API_URL}/feedback",
            json=payload,
            timeout=10,
            headers=_auth_headers(),
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Feedback submitted: {data.get('message', '')}")
            return True
        else:
            logger.error(f"Feedback submission failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return False


# ==================================================================================
# 7. HEADER & STATUS BAR
# ==================================================================================

col1, col2, col3 = st.columns([2.5, 0.5, 1.5])

with col1:
    st.title("🤖 EasyData Tier-2")
    st.caption("AI Data Analyst | Vanna 2.0.1 Agentic | Oracle Database")

with col2:
    pass  # Spacing

with col3:
    # Auto-refresh health check
    if st.button("🔄", help="Refresh backend status", key="auto_refresh"):
        st.session_state.health_status = check_backend_health()
        st.session_state.agent_state = get_agent_state()
        st.rerun()
    
    # Health indicator
    health = check_backend_health()
    if health:
        if health.get("status") == "healthy":
            st.success(f"✓ {health.get('status').capitalize()}", help="Backend is responding")
        else:
            st.warning(f"⚠️ {health.get('status').capitalize()}", help=f"Backend is degraded: {health.get('components')}")
    else:
        st.error("✗ Offline", help=f"Backend unreachable. URL: {BACKEND_URL}")

st.markdown("---")

# ==================================================================================
# 8. SIDEBAR — CONTROL PANEL
# ==================================================================================

with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # ===== SYSTEM HEALTH SECTION =====
    with st.expander("🏥 System Health", expanded=False):
        if st.button("Run Detailed Health Check", key="detailed_health", use_container_width=True):
            with st.spinner("🔍 Checking all components..."):
                health_data = check_backend_health()
                if health_data:
                    # Display component status
                    st.subheader("Component Status")
                    components = health_data.get("components", {})
                    for component, status in components.items():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if status == "ok":
                                st.success("✓")
                            elif status == "failed":
                                st.error("✗")
                            else:
                                st.info("ℹ️")
                        with col2:
                            st.write(f"**{component.capitalize()}:** {status}")
                    
                    st.metric("Overall Status", health_data.get("status", "unknown").capitalize())
                    st.json(health_data, expanded=False)
                else:
                    st.error("""
                    ❌ Backend is not responding.
                    
                    **Troubleshooting:**
                    1. Ensure the FastAPI backend is running on port 8000
                    2. Check backend URL: {} 
                    3. Verify network connectivity
                    4. Check backend logs for errors
                    """.format(BACKEND_URL))
    
    # ===== AGENT STATE SECTION =====
    with st.expander("🧠 Agent State", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Fetch State", key="fetch_state", use_container_width=True):
                with st.spinner("Loading..."):
                    state = get_agent_state()
                    if state:
                        st.session_state.agent_state = state
        
        with col2:
            if st.button("Clear Cache", key="clear_cache", use_container_width=True):
                st.session_state.messages = []
                st.session_state.agent_state = None
                st.success("Cache cleared!")
        
        if st.session_state.agent_state:
            state = st.session_state.agent_state
            
            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "📚 Memory Items",
                    state.get("memory_items_count", 0),
                    help="Total items in ChromaDB memory"
                )
            with m2:
                st.metric(
                    "📊 Trained Tables",
                    len(state.get("trained_tables", [])),
                    help="Number of tables with DDL training"
                )
            
            # Trained tables list
            if state.get("trained_tables"):
                with st.expander("View Trained Tables"):
                    for table in state["trained_tables"]:
                        st.caption(f"✓ {table}")
            else:
                st.warning("⚠️ No tables trained yet. Use Training section below.")
            
            # Component status
            st.markdown("**Component Status:**")
            status_cols = st.columns(2)
            
            with status_cols[0]:
                agent_status = "✓ OK" if state.get("agent_ready") else "✗ Error"
                st.write(f"🤖 Agent: {agent_status}")
                
                llm_status = "✓ OK" if state.get("llm_connected") else "✗ Error"
                st.write(f"🧠 LLM: {llm_status}")
            
            with status_cols[1]:
                db_status = "✓ OK" if state.get("db_connected") else "✗ Error"
                st.write(f"🗄️ Oracle: {db_status}")
                
                timestamp = state.get("timestamp", "")
                if timestamp:
                    st.caption(f"Updated: {timestamp[11:19]}")
        else:
            st.info("Click 'Fetch State' to see agent status")
    
    # ===== TRAINING SECTION =====
    st.markdown("---")
    st.subheader("📚 Training Management")
    st.write("Train the agent on your database schema (do this once):")
    
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        if st.button("🎓 Train All Tables", key="train_all", use_container_width=True):
            with st.spinner("🔄 Training agent on schema... This may take a moment."):
                result = train_schema()
                if result and result.get("success"):
                    st.session_state.training_status = result
                    st.success("✅ Training complete!")
                    
                    # Summary
                    trained_count = len(result.get("trained", []))
                    failed_count = len(result.get("failed", []))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✓ Trained", trained_count)
                    with col2:
                        st.metric("✗ Failed", failed_count)
                    
                    # Details
                    if result.get("trained"):
                        with st.expander(f"View Trained Tables ({trained_count})"):
                            for table in result["trained"]:
                                st.write(f"  ✓ {table}")
                    
                    if result.get("failed"):
                        with st.expander(f"View Failed Tables ({failed_count})"):
                            for table in result["failed"]:
                                st.write(f"  ✗ {table}")
                    
                    # Refresh state
                    time.sleep(1)
                    st.session_state.agent_state = get_agent_state()
    
    with train_col2:
        if st.button("📝 Use Script", key="use_script", use_container_width=True):
            st.info("""
            Run the training script instead:
            ```bash
            python -c "
            import requests
            response = requests.post('{}/api/v2/train', timeout=120)
            print(response.json())
            "
            ```
            This is faster for large schemas.
            """.format(BACKEND_URL))
    
    # ===== FEEDBACK SECTION =====
    st.markdown("---")
    with st.expander("💡 Feedback System", expanded=False):
        st.write("Help improve the AI by providing feedback:")
        
        if st.button("Give Feedback on Last Query", key="feedback_btn", use_container_width=True):
            if st.session_state.messages and len(st.session_state.messages) >= 2:
                last_assistant_msg = None
                last_user_msg = None
                
                # Find last assistant and user messages
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "assistant" and last_assistant_msg is None:
                        last_assistant_msg = msg
                    elif msg["role"] == "user" and last_user_msg is None:
                        last_user_msg = msg
                
                if last_assistant_msg and last_user_msg:
                    with st.form("feedback_form"):
                        question = last_user_msg["content"]
                        sql = last_assistant_msg.get("payload", {}).get("sql", "")
                        conversation_id = last_assistant_msg.get("payload", {}).get("conversation_id", "")
                        
                        st.write(f"**Question:** {question}")
                        if sql:
                            st.code(sql, language="sql")
                        
                        correct = st.radio("Was the SQL correct?", ["Yes", "No"])
                        
                        corrected_sql = None
                        notes = None
                        
                        if correct == "No":
                            corrected_sql = st.text_area("Provide corrected SQL:", height=150)
                            notes = st.text_input("Notes/Explanation:")
                        
                        submitted = st.form_submit_button("Submit Feedback")
                        
                        if submitted and conversation_id:
                            success = submit_feedback(
                                conversation_id=conversation_id,
                                question=question,
                                sql=sql,
                                correct=(correct == "Yes"),
                                corrected_sql=corrected_sql if corrected_sql else None,
                                notes=notes
                            )
                            if success:
                                st.success("✅ Feedback submitted! The AI will learn from this.")
                            else:
                                st.error("❌ Failed to submit feedback. Please try again.")
                else:
                    st.warning("No recent conversation found for feedback.")
            else:
                st.info("Ask a question first, then provide feedback.")
    
    # ===== SETTINGS SECTION =====
    st.markdown("---")
    with st.expander("⚙️ Settings", expanded=False):
        st.subheader("Backend Configuration")
        
        # Display current backend URL
        st.info(f"**Current Backend URL:** {BACKEND_URL}")
        
        # Option to change (for local testing)
        new_backend = st.text_input(
            "Override Backend URL (for local testing)",
            value=BACKEND_URL,
            help="Leave empty to use environment default"
        )
        
        if new_backend and new_backend != BACKEND_URL:
            st.warning("URL override not persisted. Set BACKEND_SERVICE_URL environment variable instead.")
        
        # Timeout setting
        timeout = st.number_input(
            "Request Timeout (seconds)",
            value=DEFAULT_TIMEOUT,
            min_value=5,
            max_value=300,
            help="Maximum time to wait for backend responses"
        )
    
    # ===== ABOUT SECTION =====
    st.markdown("---")
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        **EasyData Tier-2 Assistant**
        
        A production-ready AI data analyst that translates natural language questions 
        into SQL queries and executes them against your Oracle database.
        
        **System Components:**
        - **Frontend:** Streamlit (this interface)
        - **Backend:** FastAPI + Vanna 2.0.1 Agentic
        - **LLM:** OpenAI-compatible (GPT, Groq, Azure, Ollama)
        - **Database:** Oracle 11g+
        - **Memory:** ChromaDB (persistent vector store)
        
        **Key Features:**
        - ✅ Natural language to SQL translation
        - ✅ Automatic query execution
        - ✅ Persistent memory training
        - ✅ Real-time conversation
        - ✅ Schema discovery & DDL training
        - ✅ Feedback system for continuous learning
        - ✅ Docker-ready deployment
        
        **Version:** 1.0.0  
        **Status:** Production Ready  
        **License:** MIT
        """)

# ==================================================================================
# 9. MAIN CONVERSATION AREA
# ==================================================================================

st.markdown("---")
st.subheader("💬 Chat with Your Data")
st.write("Ask questions about your database in English or Arabic. The AI will generate SQL and show you results.")

# Display conversation history
if st.session_state.messages:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Main message content
            st.markdown(message["content"])
            
            # Additional data for assistant responses
            if "payload" in message and message["role"] == "assistant":
                payload = message["payload"]
                
                # SQL Code
                if payload.get("sql"):
                    st.markdown("**📝 Generated SQL:**")
                    st.code(payload["sql"], language="sql")
                
                # Results Table
                if payload.get("rows") and len(payload["rows"]) > 0:
                    st.markdown("**📊 Query Results:**")
                    df = pd.DataFrame(payload["rows"])
                    
                    # Scrollable dataframe
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=min(400, len(df) * 30 + 50)
                    )
                    
                    # Row count badge
                    row_count = payload.get('row_count', len(df))
                    st.caption(f"✓ {row_count} row{'s' if row_count != 1 else ''} returned")
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    timestamp = payload.get("timestamp", datetime.now().isoformat())
                    timestamp_short = timestamp.split("T")[0] if "T" in timestamp else timestamp[:10]
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"query_results_{timestamp_short}.csv",
                        mime="text/csv",
                        key=f"download_{i}"
                    )
                
                if payload.get("chart_code"):
                    st.markdown("**📈 Visualization Code:**")
                    st.code(payload["chart_code"], language="python")
                
                elif payload.get("rows") is not None and len(payload["rows"]) == 0:
                    st.info("ℹ️ Query executed successfully but returned no rows.")
                
                # Error Message
                if payload.get("error"):
                    st.error(f"⚠️ **Error:** {payload['error']}")
                
                # Memory Usage Badge
                if payload.get("memory_used"):
                    st.caption("🧠 Response used memory search")
                
                # Conversation ID (hidden by default)
                with st.expander("🔍 Debug Info"):
                    st.caption(f"Conversation ID: {payload.get('conversation_id', 'N/A')}")
                    st.caption(f"Timestamp: {payload.get('timestamp', 'N/A')}")
                    st.caption(f"Success: {payload.get('success', 'N/A')}")
else:
    st.info("👋 No conversation yet. Ask a question to get started!")

# ==================================================================================
# 10. CHAT INPUT & MESSAGE HANDLING
# ==================================================================================

user_input = st.chat_input(
    placeholder="Ask anything about your data (e.g., 'Show me the top 10 customers by revenue')...",
    key="user_chat_input"
)

if user_input:
    # 1. Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # 2. Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 3. Query backend and display response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing... 🔄 Generating SQL... 📊 Querying Oracle..."):
            response = ask_question(user_input)
            
            if response:
                # Determine response text
                if response.get("success"):
                    if response.get("row_count") == 0:
                        response_text = "✓ Query executed successfully but returned no rows."
                    elif response.get("row_count") == 1:
                        response_text = "✓ Found **1** row"
                    else:
                        response_text = f"✓ Found **{response.get('row_count', 0)}** rows"
                else:
                    response_text = f"❌ Error: {response.get('error', 'Unknown error')}"
                
                st.markdown(response_text)
                
                # Display SQL
                if response.get("sql"):
                    st.markdown("**Generated SQL:**")
                    st.code(response["sql"], language="sql")
                
                # Display results
                if response.get("rows") and len(response["rows"]) > 0:
                    st.markdown("**Results:**")
                    df = pd.DataFrame(response["rows"])
                    
                    # Dynamic height based on row count
                    row_count = len(df)
                    table_height = min(400, max(200, row_count * 35 + 50))
                    
                    st.dataframe(df, use_container_width=True, height=table_height)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    timestamp = response.get("timestamp", datetime.now().isoformat())
                    timestamp_short = timestamp.split("T")[0] if "T" in timestamp else timestamp[:10]
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"results_{timestamp_short}.csv",
                        mime="text/csv",
                        key=f"download_results_{len(st.session_state.messages)}"
                    )
                
                if response.get("chart_code"):
                    st.markdown("**Visualization Code:**")
                    st.code(response["chart_code"], language="python")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "payload": response
                })
            else:
                error_msg = "❌ No response from backend. Check if it's running."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ==================================================================================
# 11. FOOTER
# ==================================================================================

st.markdown("---")

footer_cols = st.columns([1, 1, 1])

with footer_cols[0]:
    st.caption("📊 EasyData Tier-2")

with footer_cols[1]:
    st.caption("Vanna 2.0.1 | Oracle")

with footer_cols[2]:
    st.caption("© 2026 | Production Ready")

st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.75em; margin-top: 20px;">
        <p>Backend URL: <code>{}</code> | Port: 8000</p>
        <p>Having issues? Check backend logs or ensure the FastAPI backend is running.</p>
    </div>
""".format(BACKEND_URL), unsafe_allow_html=True)

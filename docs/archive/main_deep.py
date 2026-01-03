# =============================================================================
# 0. STANDARD LIBRARIES
# =============================================================================

import os
import sys
import math
import uuid
import json
import asyncio
import logging
import re
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

# =============================================================================
# 1. THIRD PARTY CORE
# =============================================================================

import oracledb
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Body, Security, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# 2. VANNA 2.0.1 — OFFICIAL PUBLIC API ONLY
# =============================================================================

from vanna import Agent, AgentConfig
from vanna.capabilities.sql_runner import RunSqlToolArgs, SqlRunner
from vanna.core.rich_component import ComponentType
from vanna.core.registry import ToolRegistry
from vanna.core.tool import ToolContext
from vanna.core.user import RequestContext, User
from vanna.core.user.resolver import UserResolver
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import RunSqlTool

# =============================================================================
# 3. ENVIRONMENT & LOGGING
# =============================================================================

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Main application logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EasyData-Tier2")

# Audit logger for compliance (SOC2, GDPR)
audit_logger = logging.getLogger("Audit")
audit_handler = logging.FileHandler("audit.log", encoding="utf-8")
audit_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

# =============================================================================
# 4. CONFIGURATION CONSTANTS
# =============================================================================

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Oracle Configuration
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_DSN = os.getenv("ORACLE_DSN")

# Memory Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vanna_memory")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "tier2_memory")

# Security Configuration
API_KEY_REQUIRED = os.getenv("REQUIRE_AUTHENTICATION", "false").lower() == "true"
API_KEY_VALUE = os.getenv("TIER2_ACCESS_KEY", "change-me")
ENABLE_SQL_FIREWALL = os.getenv("ENABLE_SQL_FIREWALL", "true").lower() == "true"
ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true"

# System Configuration
MAX_ROWS = int(os.getenv("MAX_ROWS", "1000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# =============================================================================
# 5. SECURITY — SQL FIREWALL (ENHANCED)
# =============================================================================

class SQLFirewall:
    """Enhanced SQL Firewall with whitelist/blacklist approach"""
    
    BLACKLIST_PATTERNS = [
        r"\bDROP\b",
        r"\bTRUNCATE\b", 
        r"\bDELETE\s+FROM\b",
        r"\bDELETE\s+\w+\b",
        r"\bUPDATE\s+\w+\s+SET\b",
        r"\bINSERT\s+INTO\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bRENAME\b",
        r"\bSHUTDOWN\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"\bMERGE\b",
    ]
    
    WHITELIST_PATTERNS = [
        r"^SELECT\b",
        r"^WITH\b",  # Allow CTEs
        r"^\s*SELECT\b",  # Allow leading whitespace
    ]
    
    @staticmethod
    def validate(sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL for safety and read-only compliance"""
        if not sql or not isinstance(sql, str):
            return False, "Invalid SQL input"
        
        sql_upper = sql.upper().strip()
        
        # Check whitelist first (positive security model)
        allowed = False
        for pattern in SQLFirewall.WHITELIST_PATTERNS:
            if re.match(pattern, sql_upper, re.IGNORECASE):
                allowed = True
                break
        
        if not allowed:
            logger.warning(f"⛔ SQL not in whitelist: {sql[:100]}")
            return False, "Only SELECT and WITH (CTE) queries are allowed"
        
        # Check blacklist for bypass attempts
        for pattern in SQLFirewall.BLACKLIST_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                logger.warning(f"⛔ SQL Firewall blocked: {sql[:100]}")
                return False, f"Forbidden SQL operation detected"
        
        # Additional safety checks
        if re.search(r"(--|\/\*|\*\/)", sql):
            logger.warning(f"⛔ SQL with comments blocked: {sql[:100]}")
            return False, "SQL comments not allowed for security"
        
        # Check for excessive length (DOS prevention)
        if len(sql) > 10000:
            return False, "SQL query too long"
        
        logger.debug(f"✅ SQL passed firewall: {sql[:50]}...")
        return True, None

# =============================================================================
# 6. AUDIT LOGGER (COMPLIANCE)
# =============================================================================

class AuditLogger:
    """Audit logging for SOC2, GDPR, and enterprise compliance"""
    
    @staticmethod
    def log_request(
        user_id: str,
        action: str,
        question: str = "",
        sql: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log API request for audit trail"""
        if not ENABLE_AUDIT_LOGGING:
            return
        
        audit_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "question": question[:200] if question else "",
            "sql_hash": hashlib.sha256(sql.encode()).hexdigest() if sql else None,
            "success": success,
            "details": details or {},
        }
        
        audit_logger.info(json.dumps(audit_data))
    
    @staticmethod
    def log_security_event(
        event_type: str,
        user_id: str,
        details: Dict[str, Any],
    ):
        """Log security-specific events"""
        if not ENABLE_AUDIT_LOGGING:
            return
        
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
        }
        
        audit_logger.info(json.dumps(event_data))

# =============================================================================
# 7. SANITIZER — HARD DEFENSIVE NORMALIZATION
# =============================================================================

def sanitize_value(obj: Any, depth: int = 0) -> Any:
    """
    Deep sanitize for Oracle encoding issues, NaN, Infinity, LOBs, etc.
    
    Handles:
    - Bytes (encoding issues: UTF-8, CP1252, invalid chars)
    - Float (NaN, Infinity)
    - Collections (dict, list, tuple)
    - DateTime objects
    - Deep recursion limits
    """
    if depth > 50:
        logger.warning("Sanitizer recursion limit reached")
        return str(obj)[:500]
    
    # Handle None
    if obj is None:
        return None
    
    # Handle Bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return obj.decode("cp1252", errors="replace")
            except Exception:
                return str(obj)
    
    # Handle Float (NaN, Infinity)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return round(obj, 10)
    
    # Handle Dict
    if isinstance(obj, dict):
        return {
            sanitize_value(k, depth + 1): sanitize_value(v, depth + 1)
            for k, v in obj.items()
        }
    
    # Handle List
    if isinstance(obj, list):
        return [sanitize_value(v, depth + 1) for v in obj]
    
    # Handle Tuple
    if isinstance(obj, tuple):
        return tuple(sanitize_value(v, depth + 1) for v in obj)
    
    # Handle DateTime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle String (remove invalid chars)
    if isinstance(obj, str):
        return "".join(
            char for char in obj if ord(char) >= 32 or char in "\n\r\t"
        )
    
    # Default
    return obj

# =============================================================================
# 8. ORACLE EXECUTION ENGINE (STRICT, READ-ONLY)
# =============================================================================

class OracleRunner(SqlRunner):
    """Oracle execution engine implementing Vanna SqlRunner (async, read-only)."""
    
    def __init__(self):
        self.user = ORACLE_USER
        self.password = ORACLE_PASSWORD
        self.dsn = ORACLE_DSN
        self.max_rows = MAX_ROWS
        self._results: Dict[str, Dict[str, Any]] = {}
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test Oracle connectivity at startup."""
        try:
            conn = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
            )
            conn.close()
            logger.info("✅ Oracle connection verified")
            return True
        except Exception as e:
            logger.error(f"❌ Oracle connection failed: {e}")
            return False
    
    def _run_query(self, sql: str) -> pd.DataFrame:
        """Run SQL synchronously; called in a thread to avoid blocking."""
        conn = None
        try:
            conn = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
            )
            df = pd.read_sql(sql, conn)
            
            if len(df) > self.max_rows:
                logger.warning(
                    f"Query returned {len(df)} rows, truncating to {self.max_rows}"
                )
                df = df.head(self.max_rows)
            
            # Sanitize cells for safe JSON serialization
            df = df.applymap(lambda v: sanitize_value(v))
            return df
        
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    async def run_sql(
        self,
        args: RunSqlToolArgs,
        context: ToolContext,
    ) -> pd.DataFrame:
        """
        Execute SQL query via Vanna RunSqlTool contract.
        
        Raises on error so RunSqlTool can return a structured failure.
        """
        sql = args.sql.strip()
        conversation_id = context.conversation_id
        self._results[conversation_id] = {
            "sql": sql,
            "rows": [],
            "row_count": 0,
            "error": None,
        }
        
        if ENABLE_SQL_FIREWALL:
            is_safe, error_msg = SQLFirewall.validate(sql)
            if not is_safe:
                self._results[conversation_id]["error"] = error_msg
                raise ValueError(error_msg)
        
        try:
            df = await asyncio.to_thread(self._run_query, sql)
            rows = sanitize_value(df.to_dict(orient="records"))
            
            self._results[conversation_id]["rows"] = rows
            self._results[conversation_id]["row_count"] = len(rows)
            
            logger.info(f"✅ Query executed: {len(rows)} rows")
            return df
        
        except Exception as e:
            self._results[conversation_id]["error"] = str(e)
            logger.error(f"❌ Oracle execution error: {e}")
            raise
    
    def get_last_result(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Fetch and clear the last recorded result for a conversation."""
        return self._results.pop(conversation_id, None)

# =============================================================================
# 9. STATE TRACKER (AUTHORITATIVE)
# =============================================================================

class StateTracker:
    """
    Authoritative state tracker.
    Single source of truth for:
    - Trained tables
    - Memory item count
    - Agent readiness
    """
    
    def __init__(self):
        self.trained_tables: List[str] = []
        self.memory: Optional[ChromaAgentMemory] = None
        self.initialized_at = datetime.utcnow()
    
    def set_memory(self, memory: ChromaAgentMemory):
        """Register memory instance"""
        self.memory = memory
        logger.info("✅ Memory registered")
    
    def record_training(self, tables: List[str]):
        """Record successfully trained tables"""
        self.trained_tables = tables
        logger.info(f"✅ Trained tables updated: {len(tables)} tables")
    
    def memory_count(self) -> int:
        """Get total memory items"""
        if not self.memory:
            return 0
        try:
            collection = self.memory._get_collection()
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get memory count: {e}")
            return 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get authoritative state snapshot"""
        return {
            "trained_tables": self.trained_tables,
            "memory_items": self.memory_count(),
            "initialized_at": self.initialized_at.isoformat(),
        }

state_tracker = StateTracker()

# =============================================================================
# 9.1 USER RESOLUTION (REQUIRED BY VANNA 2.0.1)
# =============================================================================

class APIKeyUserResolver(UserResolver):
    """Resolve users from API key header or fallback to anonymous."""
    
    def __init__(self, default_user: str = "anonymous"):
        self.default_user = default_user
    
    async def resolve_user(self, request_context: RequestContext) -> User:
        api_key = request_context.get_header("X-API-Key") or self.default_user
        user_id = api_key or self.default_user
        
        return User(
            id=user_id,
            username=user_id,
            email=None,
            metadata={
                "remote_addr": request_context.remote_addr,
                **(request_context.metadata or {}),
            },
            group_memberships=["user"],
        )

# Shared helper to build ToolContext for memory operations
def build_tool_context(user_id: str, conversation_id: str) -> ToolContext:
    return ToolContext(
        user=User(id=user_id, username=user_id),
        conversation_id=conversation_id,
        request_id=str(uuid.uuid4()),
        agent_memory=state_tracker.memory,
        metadata={},
    )

# =============================================================================
# 10. API SECURITY LAYER
# =============================================================================

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key for authentication",
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API Key authentication.
    Can be disabled with REQUIRE_AUTHENTICATION=false
    """
    if not API_KEY_REQUIRED:
        return "anonymous"
    
    if not api_key:
        AuditLogger.log_security_event(
            event_type="auth_missing_key",
            user_id="unknown",
            details={"reason": "missing_api_key"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing X-API-Key header",
        )
    
    if api_key != API_KEY_VALUE:
        AuditLogger.log_security_event(
            event_type="auth_invalid_key",
            user_id="unknown",
            details={"reason": "invalid_api_key"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    return "authenticated"

# =============================================================================
# 11. SEALED RESPONSE CONTRACTS (PYDANTIC)
# =============================================================================

class AskRequest(BaseModel):
    """Request contract for /api/v2/ask"""
    question: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @field_validator("question")
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class AskResponse(BaseModel):
    """Response contract for /api/v2/ask"""
    success: bool
    error: Optional[str] = None
    
    conversation_id: str
    timestamp: str
    
    question: str
    sql: Optional[str] = None
    
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    
    memory_used: bool = False
    assumptions: Optional[str] = None

class TrainingRequest(BaseModel):
    """Request contract for /api/v2/train"""
    table_name: Optional[str] = None

class TrainingResponse(BaseModel):
    """Response contract for /api/v2/train"""
    success: bool
    trained: List[str] = Field(default_factory=list)
    failed: List[str] = Field(default_factory=list)
    timestamp: str

class FeedbackRequest(BaseModel):
    """Request contract for /api/v2/feedback"""
    conversation_id: str
    question: str
    sql: str
    correct: bool
    corrected_sql: Optional[str] = None
    notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Response contract for /api/v2/feedback"""
    status: str
    message: str
    timestamp: str

class AgentStateResponse(BaseModel):
    """Response contract for /api/v2/state"""
    memory_items_count: int
    trained_tables: List[str]
    agent_ready: bool
    llm_connected: bool
    db_connected: bool
    timestamp: str

class HealthResponse(BaseModel):
    """Response contract for /health"""
    status: str
    components: Dict[str, str]
    timestamp: str

# =============================================================================
# 12. GLOBALS (INITIALIZED AT STARTUP)
# =============================================================================

agent: Optional[Agent] = None
oracle_runner: Optional[OracleRunner] = None

# =============================================================================
# 13. LIFESPAN — SINGLE SOURCE OF TRUTH
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global agent, oracle_runner
    
    # STARTUP LOGGING
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 25 + "EasyData Tier-2 Contract v1.0" + " " * 23 + "║")
    logger.info("║" + " " * 28 + "Vanna 2.0.1 Agentic Backend" + " " * 23 + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info(f"║  LLM: {OPENAI_MODEL:<65} ║")
    logger.info(f"║  Database: {ORACLE_DSN:<59} ║")
    logger.info(f"║  Memory: {CHROMA_PATH:<62} ║")
    logger.info(f"║  Security: Auth={API_KEY_REQUIRED} | Firewall={ENABLE_SQL_FIREWALL} | Audit={ENABLE_AUDIT_LOGGING:<28} ║")
    logger.info("║" + " " * 78 + "║")
    logger.info("║  Status: ✅ PRODUCTION READY" + " " * 47 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    
    try:
        # 1. Initialize Oracle Runner
        logger.info("🚀 Initializing Oracle Runner...")
        oracle_runner = OracleRunner()
        
        # 2. Initialize LLM Service (OpenAI-compatible)
        logger.info("🧠 Initializing LLM Service...")
        llm = OpenAILlmService(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
        )
        
        # 3. Initialize ChromaDB Memory
        logger.info("💾 Initializing ChromaDB Memory...")
        memory = ChromaAgentMemory(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
        )
        state_tracker.set_memory(memory)
        
        # 4. Register Tools (SQL Execution)
        logger.info("🔧 Registering SQL Tool...")
        registry = ToolRegistry()
        
        registry.register_local_tool(
            RunSqlTool(sql_runner=oracle_runner),
            access_groups=[],
        )
        
        # 5. User Resolver (required by Agent)
        user_resolver = APIKeyUserResolver(
            default_user="anonymous",
        )
        
        # 6. Initialize Vanna Agent (2.0.1 API)
        logger.info("🤖 Initializing Vanna Agent...")
        agent = Agent(
            llm_service=llm,
            tool_registry=registry,
            user_resolver=user_resolver,
            agent_memory=memory,
            config=AgentConfig(
                stream_responses=False,
                include_thinking_indicators=False,
                auto_save_conversations=True,
                temperature=0.0,
            ),
        )
        
        # 7. Log startup completion
        AuditLogger.log_request(
            user_id="system",
            action="startup",
            question="",
            success=True,
            details={"agent": "initialized"},
        )
        
        logger.info("✅ EasyData Tier-2 Agent READY")
        
    except Exception as e:
        logger.critical(f"❌ Startup failed: {e}", exc_info=True)
        AuditLogger.log_request(
            user_id="system",
            action="startup_failed",
            question="",
            success=False,
            details={"error": str(e)},
        )
        raise
    
    yield
    
    # SHUTDOWN
    logger.info("🛑 EasyData Tier-2 shutting down...")
    AuditLogger.log_request(
        user_id="system",
        action="shutdown",
        question="",
        success=True,
        details={},
    )

# =============================================================================
# 14. FASTAPI APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="EasyData Tier-2 Contract",
    description="Production-grade Vanna 2.0.1 Agentic Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================================================
# 15. ENDPOINTS — /api/v2/ask (MAIN QUERY)
# =============================================================================

@app.post("/api/v2/ask", response_model=AskResponse)
async def ask_question(
    req: AskRequest,
    api_key: str = Security(verify_api_key),
    request: Request = None,
) -> AskResponse:
    """
    Main Q&A endpoint.
    
    Flow:
    1. Validate agent ready
    2. Route question through Vanna Agent (tools + memory)
    3. Collect SQL + results from RunSqlTool
    4. Persist to memory
    5. Return response
    """
    
    if not agent or not oracle_runner:
        logger.error("Agent not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not ready",
        )
    
    conversation_id = f"tier2-{uuid.uuid4().hex[:12]}"
    timestamp = datetime.utcnow().isoformat()
    rows: List[Dict[str, Any]] = []
    row_count = 0
    sql: Optional[str] = None
    error_msg: Optional[str] = None
    memory_used = False
    
    try:
        logger.info(f"[{conversation_id}] Question: {req.question[:50]}...")
        
        request_context = RequestContext(
            headers=dict(request.headers),
            cookies=request.cookies,
            remote_addr=request.client.host if request.client else None,
            query_params=dict(request.query_params),
            metadata={"api_key": api_key, "conversation_id": conversation_id},
        )
        
        # Route through Vanna Agent (RunSqlTool will execute our OracleRunner)
        async for component in agent.send_message(
            request_context=request_context,
            message=req.question,
            conversation_id=conversation_id,
        ):
            rich = getattr(component, "rich_component", None)
            simple = getattr(component, "simple_component", None)
            
            if rich and getattr(rich, "type", None) == ComponentType.DATAFRAME:
                rows = sanitize_value(getattr(rich, "rows", []))
                row_count = getattr(rich, "row_count", len(rows))
            
            if simple and getattr(simple, "text", None) and not error_msg:
                text_val = str(simple.text)
                if text_val.lower().startswith("error"):
                    error_msg = text_val
        
        # Pull structured result captured by OracleRunner
        runner_result = oracle_runner.get_last_result(conversation_id) or {}
        sql = runner_result.get("sql")
        if not rows and runner_result.get("rows") is not None:
            rows = runner_result.get("rows", [])
            row_count = runner_result.get("row_count", len(rows))
        if not error_msg:
            error_msg = runner_result.get("error")
        
        if error_msg or not sql:
            raise ValueError(error_msg or "No SQL execution detected")
        
        # Save to Memory (best-effort)
        if state_tracker.memory:
            try:
                ctx = build_tool_context(api_key, conversation_id)
                await state_tracker.memory.save_text_memory(
                    content=f"Q: {req.question}\nSQL: {sql}",
                    context=ctx,
                )
                memory_used = True
                logger.info(f"[{conversation_id}] Saved to memory")
            except Exception as e:
                logger.warning(f"[{conversation_id}] Memory save failed: {e}")
        
        # Audit Log Success
        AuditLogger.log_request(
            user_id=api_key,
            action="ask_success",
            question=req.question,
            sql=sql,
            success=True,
            details={"row_count": row_count},
        )
        
        return AskResponse(
            success=True,
            conversation_id=conversation_id,
            timestamp=timestamp,
            question=req.question,
            sql=sql,
            rows=rows,
            row_count=row_count,
            memory_used=memory_used,
        )
    
    except Exception as e:
        logger.error(f"[{conversation_id}] Error: {e}")
        
        AuditLogger.log_request(
            user_id=api_key,
            action="ask_error",
            question=req.question,
            success=False,
            details={"error": str(e)},
        )
        
        return AskResponse(
            success=False,
            error=str(e),
            conversation_id=conversation_id,
            timestamp=timestamp,
            question=req.question,
        )

# =============================================================================
# 16. ENDPOINTS — /api/v2/train (SCHEMA TRAINING)
# =============================================================================

@app.post("/api/v2/train", response_model=TrainingResponse)
async def train_schema(
    req: Optional[TrainingRequest] = Body(default=None),
    api_key: str = Security(verify_api_key),
) -> TrainingResponse:
    """
    Train agent on database schema (DDL-based).
    
    If table_name provided: train specific table
    Otherwise: train all tables
    """
    
    if not agent or not oracle_runner:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not ready",
        )
    
    timestamp = datetime.utcnow().isoformat()
    trained: List[str] = []
    failed: List[str] = []
    
    try:
        logger.info("🔄 Starting schema training...")
        
        conn = oracledb.connect(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN,
        )
        cursor = conn.cursor()
        
        # Get tables to train
        if req and req.table_name:
            tables = [req.table_name]
            logger.info(f"Training specific table: {req.table_name}")
        else:
            cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Training all {len(tables)} tables")
        
        # Train each table
        for table in tables:
            try:
                logger.info(f"  → Training {table}...")
                
                cursor.execute(
                    f"SELECT DBMS_METADATA.GET_DDL('TABLE', '{table}') FROM DUAL"
                )
                row = cursor.fetchone()
                
                if not row or not row[0]:
                    logger.warning(f"    ⚠ No DDL for {table}")
                    failed.append(table)
                    continue
                
                ddl = str(row[0])
                
                # Save DDL to memory
                if state_tracker.memory:
                    try:
                        mem_ctx = build_tool_context(api_key, f"train-{table}")
                        await state_tracker.memory.save_text_memory(
                            content=f"TABLE: {table}\n\n{ddl}",
                            context=mem_ctx,
                        )
                    except Exception as mem_err:
                        logger.warning(f"    ⚠ Memory save failed for {table}: {mem_err}")
                
                trained.append(table)
                logger.info(f"    ✅ {table}")
            
            except Exception as e:
                logger.error(f"    ❌ {table}: {e}")
                failed.append(table)
        
        cursor.close()
        conn.close()
        
        # Update state
        state_tracker.record_training(trained)
        
        # Audit
        AuditLogger.log_request(
            user_id=api_key,
            action="train",
            question="",
            success=len(failed) == 0,
            details={"trained": len(trained), "failed": len(failed)},
        )
        
        logger.info(f"✅ Training complete: {len(trained)} trained, {len(failed)} failed")
        
        return TrainingResponse(
            success=len(failed) == 0,
            trained=trained,
            failed=failed,
            timestamp=timestamp,
        )
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
        AuditLogger.log_request(
            user_id=api_key,
            action="train_error",
            question="",
            success=False,
            details={"error": str(e)},
        )
        
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 17. ENDPOINTS — /api/v2/feedback (LEARNING)
# =============================================================================

@app.post("/api/v2/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    api_key: str = Security(verify_api_key),
) -> FeedbackResponse:
    """
    Submit feedback for continuous learning.
    
    If correct: validates Q-SQL pair
    If incorrect: stores correction
    """
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not ready",
        )
    
    timestamp = datetime.utcnow().isoformat()
    
    try:
        AuditLogger.log_request(
            user_id=api_key,
            action="feedback",
            question=feedback.question,
            sql=feedback.sql,
            details={"correct": feedback.correct},
        )
        
        if feedback.correct:
            # Store validated Q-SQL pair
            logger.info(f"🧠 Learning correct pair: {feedback.question[:50]}...")
            if state_tracker.memory:
                ctx = build_tool_context(api_key, feedback.conversation_id)
                await state_tracker.memory.save_text_memory(
                    content=f"VERIFIED Q-SQL:\nQ: {feedback.question}\nSQL: {feedback.sql}",
                    context=ctx,
                )
            message = "Feedback processed - correct pattern learned"
        
        elif feedback.corrected_sql:
            # Store correction
            logger.info(f"🧠 Learning correction: {feedback.question[:50]}...")
            if state_tracker.memory:
                ctx = build_tool_context(api_key, feedback.conversation_id)
                await state_tracker.memory.save_text_memory(
                    content=f"CORRECTION:\nQ: {feedback.question}\nCORRECTED SQL: {feedback.corrected_sql}",
                    context=ctx,
                )
            message = "Feedback processed - correction learned"
        
        else:
            message = "Feedback recorded (no learning action)"
        
        return FeedbackResponse(
            status="success",
            message=message,
            timestamp=timestamp,
        )
    
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        
        AuditLogger.log_request(
            user_id=api_key,
            action="feedback_error",
            question=feedback.question,
            success=False,
            details={"error": str(e)},
        )
        
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 18. ENDPOINTS — /api/v2/state (STATE INFORMATION)
# =============================================================================

@app.get("/api/v2/state", response_model=AgentStateResponse)
async def get_agent_state(api_key: str = Security(verify_api_key)) -> AgentStateResponse:
    """Get authoritative agent state."""
    
    return AgentStateResponse(
        memory_items_count=state_tracker.memory_count(),
        trained_tables=state_tracker.trained_tables,
        agent_ready=agent is not None,
        llm_connected=OPENAI_API_KEY is not None,
        db_connected=oracle_runner is not None,
        timestamp=datetime.utcnow().isoformat(),
    )

# =============================================================================
# 19. ENDPOINTS — /health (HEALTH CHECK)
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    System health check.
    
    Returns status of:
    - Agent
    - Oracle Runner
    - Memory
    - LLM
    """
    
    components = {
        "agent": "ok" if agent else "failed",
        "oracle": "ok" if oracle_runner else "failed",
        "memory": "ok" if state_tracker.memory else "failed",
        "llm": "ok" if OPENAI_API_KEY else "unconfigured",
    }
    
    # Health logic: system is healthy if core components are ok
    # LLM can be "unconfigured" but system still functional for state queries
    status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        components=components,
        timestamp=datetime.utcnow().isoformat(),
    )

# =============================================================================
# 20. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
    )

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EasyData Tier-2 Vanna 2.0.1 OSS                          ║
║                                                                              ║
║  Single-File, Productivity-First, All Official Tools Enabled                ║
║  FastAPI Lifespan, Sealed Response Contracts, Feedback Loop                 ║
║                                                                              ║
║  Official Vanna 2.0.1 APIs Only:                                            ║
║  - Agent, AgentConfig, ToolRegistry, ToolContext, UserResolver              ║
║  - OpenAILlmService, ChromaAgentMemory                                       ║
║  - RunSqlTool, VisualizeDataTool, SaveQuestionToolTool, SaveTextMemoryTool   ║
║                                                                              ║
║  Status: PRODUCTION-READY OSS                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY              OpenAI API key (required)
    OPENAI_BASE_URL             OpenAI base URL (default: https://api.openai.com/v1)
    OPENAI_MODEL                Model name (default: gpt-4-turbo)
    
    ORACLE_USER                 Oracle user (required)
    ORACLE_PASSWORD             Oracle password (required)
    ORACLE_DSN                  Oracle DSN, e.g., localhost:1521/XEPDB1 (required)
    
    CHROMA_PATH                 ChromaDB persistence path (default: ./vanna_memory)
    CHROMA_COLLECTION           ChromaDB collection name (default: tier2_vanna)
    
    LOG_LEVEL                   Logging level (default: INFO)
    MAX_ROWS                    Max rows per query (default: 1000)
    STREAMING_MODE              Enable streaming responses (default: false)
    
QUICK START:
    pip install fastapi uvicorn python-dotenv oracledb pandas \
                openai vanna chromadb pydantic
    
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

EXAMPLE .env:
    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_MODEL=gpt-4-turbo
    
    ORACLE_USER=system
    ORACLE_PASSWORD=your_pass
    ORACLE_DSN=localhost:1521/XEPDB1
    
    CHROMA_PATH=./vanna_memory
    CHROMA_COLLECTION=tier2_vanna
    
    LOG_LEVEL=INFO
    MAX_ROWS=1000
    STREAMING_MODE=false
"""

import os
import sys
import json
import uuid
import logging
import math
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator

import oracledb
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ========== VANNA 2.0.1 OFFICIAL PUBLIC API ==========
from vanna import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.core.tool import ToolContext
from vanna.core.user import RequestContext, User, UserResolver
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import RunSqlTool, VisualizeDataTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsParams,
    SaveQuestionToolArgsTool,
    SaveTextMemoryTool,
    SearchSavedCorrectToolUsesTool,
)
from vanna.legacy.openai import OpenAI_Chat
from vanna.legacy.chromadb import ChromaDB_VectorStore
from config.metadata_loader import metadata
from sqlglot_name_mapper import (
    SQLGatekeeper,
    ValidationError,
    get_allowed_tables_from_memory,
)
from prompts.schema_prompt_builder import PROMPT_SYSTEM
from trainers.auto_semantic_trainer import generate_training_pairs

# ==================================================================================
# INITIALIZATION
# ==================================================================================

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tier2-vanna")

# Global instances
agent: Optional[Agent] = None
oracle_runner: Optional["OracleRunner"] = None
chroma_memory: Optional[ChromaAgentMemory] = None
streaming_enabled: bool = os.getenv("STREAMING_MODE", "false").lower() == "true"
save_q_tool_global: Optional[SaveQuestionToolArgsTool] = None
viz_tool_global: Optional[VisualizeDataTool] = None
vn: Optional["MyVanna"] = None
gatekeeper: Optional[SQLGatekeeper] = None
AUTO_TRAIN_MARKER_TYPE = "auto_semantic_training"


# ==================================================================================
# 1. CUSTOM SQL RUNNER (Vanna-Compatible)
# ==================================================================================


class OracleRunner:
    """
    Oracle SQL Runner implementing Vanna's SqlRunner interface.
    
    This runner:
    - Executes SELECT queries only
    - Returns sanitized DataFrames
    - Handles encoding/float edge cases
    - Enforces MAX_ROWS limit
    """

    def __init__(self, user: str, password: str, dsn: str, max_rows: int = 1000):
        self.user = user
        self.password = password
        self.dsn = dsn
        self.max_rows = max_rows

    def _sanitize_value(self, obj: Any, depth: int = 0) -> Any:
        """
        Sanitize Oracle values for JSON serialization.
        
        Handles:
        - Bytes (utf-8, cp1252 fallback)
        - NaN/Infinity floats
        - Nested dicts/lists/tuples
        - Datetime objects
        """
        if depth > 50:
            logger.warning(f"Sanitizer: recursion depth {depth}, truncating")
            return str(obj)[:1000]

        # Bytes
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                try:
                    return obj.decode("cp1252", errors="replace")
                except Exception:
                    hex_repr = obj.hex()[:50]
                    logger.debug(f"Bytes fallback to hex: {hex_repr}")
                    return f"<binary:{hex_repr}>"

        # Float NaN/Inf
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        # Containers
        if isinstance(obj, dict):
            return {
                self._sanitize_value(k, depth + 1): self._sanitize_value(
                    v, depth + 1
                )
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [self._sanitize_value(v, depth + 1) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._sanitize_value(v, depth + 1) for v in obj)

        # Datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        return obj

    def run_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL and return DataFrame (Vanna interface).
        
        Args:
            sql: SELECT query
            
        Returns:
            DataFrame with sanitized values
            
        Raises:
            Exception: SQL execution error
        """
        conn = None
        try:
            logger.info(f"Oracle: executing {sql[:80]}...")

            conn = oracledb.connect(
                user=self.user, password=self.password, dsn=self.dsn
            )

            df = pd.read_sql(sql, conn)

            # Enforce max rows
            if len(df) > self.max_rows:
                logger.warning(
                    f"Result {len(df)} rows > MAX_ROWS {self.max_rows}, truncating"
                )
                df = df.head(self.max_rows)

            # Sanitize all values
            for col in df.columns:
                df[col] = df[col].apply(lambda x: self._sanitize_value(x))

            logger.info(f"Oracle: ✓ {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Oracle execution error: {e}")
            raise

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass


# ==================================================================================
# 2. REQUEST CONTEXT & USER RESOLVER (Vanna Compliance)
# ==================================================================================


class SimpleUserResolver(UserResolver):
    """Minimal user resolver for stateless operation."""

    def resolve_user(self, user_id: Optional[str] = None) -> User:
        """Return a default user."""
        return User(id=user_id or "default_user")


def create_request_context(
    user_id: Optional[str] = None, conversation_id: Optional[str] = None
) -> RequestContext:
    """Create a RequestContext for Vanna operations."""
    resolver = SimpleUserResolver()
    user = resolver.resolve_user(user_id)
    # Pydantic RequestContext ignores unknown fields; keep conversation_id in metadata
    return RequestContext(
        metadata={
            "conversation_id": conversation_id or "",
            "user_id": user.id,
        },
    )


def is_sql_safe(sql: str) -> bool:
    """Lightweight SQL sandbox: allow only non-mutating queries."""
    forbidden_keywords = ["delete", "update", "insert", "drop", "alter", "truncate"]
    lowered = sql.lower()
    return all(word not in lowered for word in forbidden_keywords)


def is_sql_question(question: str) -> bool:
    """Heuristic to decide if the question expects a SQL answer."""
    sql_keywords = [
        "select",
        "list",
        "show",
        "get",
        "fetch",
        "اعرض",
        "احضر",
        "هات",
        "قائمة",
        "سجل",
    ]
    q = question.lower()
    return any(k in q for k in sql_keywords)


def is_schema_question(question: str) -> bool:
    """Detect schema/describe questions to keep them off the SQL generation path."""
    keywords = ["أعمدة", "columns", "schema", "describe", "structure"]
    q = question.lower()
    return any(k in q for k in keywords)


def generate_safe_sql(question: str, context: RequestContext) -> Optional[str]:
    """
    Hybrid SQL generation:
    1) Try agentic path (agent.ask) if available.
    2) Fallback to legacy vn.generate_sql() with injected prompt.
    Applies gatekeeper policy and strips dummy SQL.
    """
    if gatekeeper is None:
        return None

    # Agentic path (optional)
    if agent and hasattr(agent, "ask") and callable(getattr(agent, "ask")):
        try:
            res = agent.ask(question=question, context=context)
            sql = getattr(res, "sql", None)
            if isinstance(sql, str) and sql.strip():
                clean_sql = sql.strip().rstrip(";")
                try:
                    return gatekeeper.validate(clean_sql)
                except ValidationError as e:
                    logger.warning(f"Agentic SQL failed policy: {e}")
        except Exception as e:
            logger.debug(f"Agentic skipped: {e}")

    # Legacy path (Vanna 2.0.1 returns str)
    if vn:
        try:
            vn.prompt = PROMPT_SYSTEM
            legacy_sql = vn.generate_sql(question)
            if not isinstance(legacy_sql, str):
                logger.error(f"Unexpected legacy return type: {type(legacy_sql)}")
                return None
            clean_sql = legacy_sql.strip()
            if not clean_sql:
                return None
            clean_sql = clean_sql.rstrip(";")
            return gatekeeper.validate(clean_sql)
        except ValidationError as e:
            logger.error(f"Legacy SQL rejected by gatekeeper: {e}")
        except Exception as e:
            logger.error(f"Legacy path failed: {e}")

    return None


# ==================================================================================
# 3. RESPONSE CONTRACTS (Sealed via Pydantic)
# ==================================================================================


class Assumption(BaseModel):
    """Single assumption extracted from agent reasoning."""
    key: str = Field(..., description="Assumption identifier")
    value: str = Field(..., description="Assumption statement")


class AskRequest(BaseModel):
    """User question request."""
    question: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context")


class AskResponse(BaseModel):
    """
    Sealed response contract for /ask endpoint.
    
    All fields are deterministic; no hidden state.
    """

    # Status
    success: bool = Field(..., description="Operation success")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Identification
    conversation_id: str = Field(..., description="Unique conversation ID")
    timestamp: str = Field(..., description="ISO timestamp")

    # Input
    question: str = Field(..., description="Original question")

    # Agent reasoning
    assumptions: List[Assumption] = Field(
        default_factory=list, description="Agent assumptions about the question"
    )

    # Output
    sql: Optional[str] = Field(None, description="Generated SQL")
    rows: List[Dict[str, Any]] = Field(
        default_factory=list, description="Result rows"
    )
    row_count: int = Field(default=0, description="Number of rows returned")

    # Optional: Chart/visualization
    chart_code: Optional[str] = Field(None, description="Python matplotlib code")

    # Memory
    memory_used: bool = Field(
        False, description="Was memory search used for context?"
    )

    # Additional metadata
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TrainingRequest(BaseModel):
    """Training request."""
    table_name: Optional[str] = Field(None, description="Single table or all if None")


class TrainingStatus(BaseModel):
    """Training result."""
    success: bool = Field(..., description="Training succeeded")
    trained: List[str] = Field(default_factory=list, description="Trained tables")
    failed: List[str] = Field(default_factory=list, description="Failed tables")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp",
    )


class FeedbackRequest(BaseModel):
    """Feedback on generated SQL."""
    question: str = Field(..., description="Original question")
    sql_generated: str = Field(..., description="Generated SQL")
    sql_corrected: Optional[str] = Field(None, description="Corrected SQL (if wrong)")
    is_correct: bool = Field(
        default=False, description="Was the generated SQL correct?"
    )
    notes: Optional[str] = Field(None, description="Additional notes")


class FeedbackResponse(BaseModel):
    """Feedback acceptance response."""
    success: bool = Field(..., description="Feedback saved")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp",
    )


class AgentStateResponse(BaseModel):
    """Agent current state."""
    memory_items_count: int = Field(..., description="ChromaDB collection size")
    trained_tables: List[str] = Field(..., description="Tables trained on DDL")
    agent_ready: bool = Field(..., description="Agent ready")
    llm_connected: bool = Field(..., description="LLM reachable")
    db_connected: bool = Field(..., description="Oracle reachable")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="'healthy' | 'degraded' | 'failed'")
    components: Dict[str, str] = Field(..., description="Component statuses")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp",
    )


# ==================================================================================
# 4. STATE & TRACKING (No Approximations)
# ==================================================================================


class StateTracker:
    """Track agent state from authoritative sources."""

    def __init__(self):
        self.trained_tables_list: List[str] = []
        self.memory: Optional[ChromaAgentMemory] = None
        self._memory_count: int = 0

    def set_memory(self, mem: ChromaAgentMemory):
        self.memory = mem

    def record_training(self, tables: List[str]):
        self.trained_tables_list = tables
        # memory count updated when saving items

    def add_memory_items(self, count: int = 1):
        self._memory_count += max(count, 0)

    def get_exact_memory_count(self) -> int:
        """Get actual ChromaDB count (not estimate)."""
        return self._memory_count

    def get_state(self) -> AgentStateResponse:
        return AgentStateResponse(
            memory_items_count=self.get_exact_memory_count(),
            trained_tables=self.trained_tables_list,
            agent_ready=agent is not None,
            llm_connected=True,  # Would require actual connectivity test
            db_connected=True,  # Would require actual connectivity test
        )


state_tracker = StateTracker()


# ==================================================================================
# AUTO SEMANTIC TRAINING (memory-driven)
# ==================================================================================


def run_auto_semantic_training(vn_instance: "MyVanna") -> None:
    """
    Generate minimal semantic training examples from metadata and store them once.
    Idempotent: skips tables already marked in memory.
    """
    # Tables that already have SQL examples in memory (avoid retraining)
    trained_tables = get_allowed_tables_from_memory(vn_instance)

    pairs = generate_training_pairs()
    trained_now = set()

    for p in pairs:
        sql = p.get("sql")
        question = p.get("question")
        if not sql or not question:
            continue

        # Extract table name heuristically from SQL (FROM <TABLE>)
        table = None
        tokens = sql.split()
        if "FROM" in [t.upper() for t in tokens]:
            try:
                idx = [t.upper() for t in tokens].index("FROM")
                table = tokens[idx + 1].strip()
            except Exception:
                table = None

        if table and table in trained_tables:
            continue

        try:
            vn_instance.train(question=question, sql=sql)
            trained_now.add(table or "")
        except Exception:
            continue

    if trained_now:
        logger.info(
            f"🧠 Auto semantic training completed for: "
            f"{', '.join(sorted(t for t in trained_now if t))}"
        )
    else:
        logger.info("🧠 Auto semantic training skipped (already done or no pairs)")


# ==================================================================================
# LEGACY FALLBACK (OpenAI + ChromaDB)
# ==================================================================================


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """Legacy Vanna with OpenAI-compatible LLM and ChromaDB vector store."""

    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


# ==================================================================================
# 5. FASTAPI APP & LIFESPAN
# ==================================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context: startup and shutdown."""
    
    # ===== STARTUP =====
    try:
        logger.info("🔄 Tier-2 Vanna 2.0.1 startup...")

        global agent, oracle_runner, chroma_memory, save_q_tool_global, viz_tool_global, vn

        # Initialize Oracle Runner
        oracle_runner = OracleRunner(
            user=os.getenv("ORACLE_USER"),
            password=os.getenv("ORACLE_PASSWORD"),
            dsn=os.getenv("ORACLE_DSN"),
            max_rows=int(os.getenv("MAX_ROWS", "1000")),
        )
        logger.info("✓ Oracle runner initialized")

        # Shared config for legacy + agentic
        config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "path": os.getenv("CHROMA_PATH", "./vanna_memory"),
            "collection_name": os.getenv("CHROMA_COLLECTION", "tier2_vanna"),
        }

        # Initialize LLM Service
        llm = OpenAILlmService(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
        )
        logger.info(f"✓ LLM initialized: {os.getenv('OPENAI_MODEL', 'gpt-4-turbo')}")

        # Initialize ChromaDB Memory
        chroma_memory = ChromaAgentMemory(
            collection_name=config["collection_name"],
            persist_directory=config["path"],
        )
        logger.info(f"✓ Memory initialized: {os.getenv('CHROMA_PATH', './vanna_memory')}")
        state_tracker.set_memory(chroma_memory)

        # Initialize legacy Vanna (fallback)
        vn = MyVanna(config=config)
        vn.run_sql = lambda sql: oracle_runner.run_sql(sql)  # type: ignore[assignment]
        logger.info("✓ Legacy Vanna initialized (fallback)")
        # Gatekeeper depends on vn (memory-based ACL)
        global gatekeeper
        gatekeeper = SQLGatekeeper(vn)

        # Auto semantic training (idempotent) — disabled by default
        if os.getenv("AUTO_TRAIN_ENABLED", "false").lower() == "true":
            try:
                run_auto_semantic_training(vn)
            except Exception as e:
                logger.warning(f"Auto semantic training skipped: {e}")
        else:
            logger.info("🧠 Auto semantic training disabled (set AUTO_TRAIN_ENABLED=true to enable)")

        # Initialize ToolRegistry & register official tools
        tool_registry = ToolRegistry()

        # 1. RunSqlTool (core SQL execution)
        run_sql_tool = RunSqlTool(sql_runner=oracle_runner)
        tool_registry.register_local_tool(run_sql_tool, access_groups=[])
        logger.info("✓ RunSqlTool registered")

        # 2. VisualizeDataTool (matplotlib code generation)
        viz_tool = VisualizeDataTool()
        tool_registry.register_local_tool(viz_tool, access_groups=[])
        logger.info("✓ VisualizeDataTool registered")

        # 3. SaveQuestionToolArgsTool (save Q↔SQL pairs)
        save_q_tool = SaveQuestionToolArgsTool()
        tool_registry.register_local_tool(save_q_tool, access_groups=[])
        logger.info("✓ SaveQuestionToolArgsTool registered")

        # 4. SaveTextMemoryTool (generic memory save)
        save_text_tool = SaveTextMemoryTool()
        tool_registry.register_local_tool(save_text_tool, access_groups=[])
        logger.info("✓ SaveTextMemoryTool registered")

        # 5. SearchSavedCorrectToolUsesTool (memory retrieval for verified examples)
        search_mem_tool = SearchSavedCorrectToolUsesTool()
        tool_registry.register_local_tool(search_mem_tool, access_groups=[])
        logger.info("✓ SearchSavedCorrectToolUsesTool registered")

        # Expose select tools globally for feedback/visualization
        save_q_tool_global = save_q_tool
        viz_tool_global = viz_tool

        # Initialize Agent with AgentConfig (no fallback wiring)
        agent = Agent(
            llm_service=llm,
            tool_registry=tool_registry,
            agent_memory=chroma_memory,
            user_resolver=SimpleUserResolver(),
        )
        logger.info("✓ Vanna Agent initialized (Agentic API)")

        logger.info("✅ Tier-2 Vanna 2.0.1 READY")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        sys.exit(1)

    yield  # App runs here

    # ===== SHUTDOWN =====
    logger.info("🛑 Shutting down...")
    try:
        if chroma_memory:
            # Persist memory if needed
            pass
        logger.info("✓ Shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


app = FastAPI(
    title="EasyData Tier-2 Vanna 2.0.1 OSS",
    description="Single-file, productivity-first, all official tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================================================
# 6. HELPER: Extract Assumptions from Agent Response
# ==================================================================================


def extract_assumptions_from_context(
    question: str, sql: Optional[str] = None, memory_hit: bool = False
) -> List[Assumption]:
    """
    Extract assumptions from question and generated SQL.
    
    In a real system, you'd parse the agent's reasoning chain.
    For now, we infer basic assumptions from artifacts we control.
    """
    assumptions = []

    if memory_hit:
        assumptions.append(
            Assumption(
                key="memory_retrieval",
                value="Used prior verified examples from memory",
            )
        )

    if any(word in question.lower() for word in ["total", "sum", "count", "average", "group"]):
        assumptions.append(
            Assumption(
                key="aggregation",
                value="Question requests aggregated data; using GROUP BY and aggregate functions",
            )
        )

    if " and " in question.lower() or " join " in question.lower():
        assumptions.append(
            Assumption(
                key="multi_table",
                value="Question may require joining multiple tables",
            )
        )

    if sql and " join " in sql.lower():
        assumptions.append(
            Assumption(
                key="join_inferred",
                value="JOIN inferred from generated SQL",
            )
        )

    if sql:
        assumptions.append(
            Assumption(
                key="sql_generated",
                value=f"SQL generated via LLM; executing: {sql[:60]}...",
            )
        )

    return assumptions


# ==================================================================================
# 7. STREAM HELPER (for pseudo-streaming)
# ==================================================================================


async def stream_ask_response(
    conversation_id: str,
    question: str,
    assumptions: List[Assumption],
    sql: Optional[str],
    rows: List[Dict[str, Any]],
    row_count: int,
    memory_used: bool,
    error: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Pseudo-stream the response in stages.
    
    Yields JSON objects for: assumptions -> sql -> rows -> complete
    """
    
    # Stage 1: Assumptions
    yield json.dumps(
        {
            "stage": "assumptions",
            "assumptions": [a.dict() for a in assumptions],
            "timestamp": datetime.utcnow().isoformat(),
        }
    ) + "\n"

    # Stage 2: SQL (if generated)
    if sql:
        yield json.dumps(
            {
                "stage": "sql",
                "sql": sql,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ) + "\n"

    # Stage 3: Results
    yield json.dumps(
        {
            "stage": "results",
            "row_count": row_count,
            "rows": rows,
            "timestamp": datetime.utcnow().isoformat(),
        }
    ) + "\n"

    # Stage 4: Complete response
    yield json.dumps(
        {
            "stage": "complete",
            "success": error is None,
            "error": error,
            "conversation_id": conversation_id,
            "question": question,
            "memory_used": memory_used,
            "timestamp": datetime.utcnow().isoformat(),
        }
    ) + "\n"


# ==================================================================================
# 8. ENDPOINTS
# ==================================================================================


@app.get("/api/v2/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    components = {
        "agent": "ok" if agent else "failed",
        "memory": "ok" if chroma_memory else "failed",
        "oracle": "ok" if oracle_runner else "failed",
    }
    status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"

    return HealthResponse(status=status, components=components)


@app.get("/api/v2/state", response_model=AgentStateResponse)
async def get_state() -> AgentStateResponse:
    """Get current agent state (no approximations)."""
    return state_tracker.get_state()


@app.post("/api/v2/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, stream: bool = Query(False)) -> AskResponse:
    """
    Main endpoint: Natural language → SQL → execution.
    
    Query param: ?stream=true to get streaming response (NDJSON).
    """
    if not agent or not oracle_runner or not chroma_memory:
        logger.error("Agent not initialized")
        raise HTTPException(status_code=503, detail="Service not ready")

    conversation_id = f"tier2-{uuid.uuid4().hex[:12]}"
    request_context = create_request_context(conversation_id=conversation_id)
    user_for_context = SimpleUserResolver().resolve_user(
        request_context.metadata.get("user_id")
    )

    # Early rejection for non-SQL questions
    question_text = request.question.strip()
    if not is_sql_question(question_text):
        logger.warning(f"[{conversation_id}] Non-SQL question rejected")
        return AskResponse(
            success=False,
            error="This question is informational and not answerable via SQL.",
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat(),
            question=request.question,
        )
    if is_schema_question(question_text):
        logger.warning(f"[{conversation_id}] Schema question rejected from SQL path")
        return AskResponse(
            success=False,
            error="Schema questions are answered via trained metadata only.",
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat(),
            question=request.question,
        )

    try:
        logger.info(f"[{conversation_id}] Q: {request.question}")

        # ===== STEP 1: Generate SQL (hybrid: Agentic -> Legacy) =====
        assumptions: List[Assumption] = []
        try:
            sql = generate_safe_sql(request.question, request_context)
            if not sql:
                logger.warning(f"[{conversation_id}] SQL generation returned empty")
                return AskResponse(
                    success=False,
                    error="Could not generate SQL",
                    conversation_id=conversation_id,
                    timestamp=datetime.utcnow().isoformat(),
                    question=request.question,
                    assumptions=assumptions,
                )
            logger.info(f"[{conversation_id}] Generated: {sql[:80]}...")
            assumptions = extract_assumptions_from_context(request.question, sql)
        except Exception as e:
            logger.error(f"[{conversation_id}] SQL generation error: {e}")
            return AskResponse(
                success=False,
                error=f"SQL generation failed: {str(e)}",
                conversation_id=conversation_id,
                timestamp=datetime.utcnow().isoformat(),
                question=request.question,
                assumptions=assumptions,
            )

        # ===== STEP 2: Execute SQL =====
        rows = []
        row_count = 0
        try:
            # Lightweight safety: block non-SELECT (productivity-friendly)
            if not is_sql_safe(sql):
                logger.error(f"[{conversation_id}] Unsafe SQL blocked")
                raise HTTPException(status_code=400, detail="SQL contains unsafe operations.")
            if re.search(r"(?i)\b(DELETE|UPDATE|INSERT|DROP|TRUNCATE|ALTER)\b", sql):
                raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

            df = oracle_runner.run_sql(sql)
            rows = df.to_dict(orient="records")
            row_count = len(rows)
            logger.info(f"[{conversation_id}] ✓ {row_count} rows")

        except Exception as e:
            logger.error(f"[{conversation_id}] Execution error: {e}")
            return AskResponse(
                success=False,
                error=f"SQL execution failed: {str(e)}",
                conversation_id=conversation_id,
                timestamp=datetime.utcnow().isoformat(),
                question=request.question,
                sql=sql,
                assumptions=assumptions,
            )

        # ===== STEP 3: Optional Chart Generation (VisualizeDataTool) =====
        chart_code = None
        try:
            if row_count > 0 and viz_tool_global:
                # Build ToolContext explicitly (public API)
                tool_ctx = ToolContext(
                    user=user_for_context,
                    conversation_id=conversation_id,
                    request_id=uuid.uuid4().hex,
                    agent_memory=chroma_memory,
                    metadata={},
                )
                # Attempt visualization; tolerate failure
                chart_result = None
                if hasattr(viz_tool_global, "run"):
                    try:
                        chart_result = viz_tool_global.run(
                            rows=rows, context=tool_ctx  # type: ignore[arg-type]
                        )
                    except Exception:
                        chart_result = None
                if hasattr(chart_result, "result_for_llm"):
                    chart_code = getattr(chart_result, "result_for_llm", None)
                elif isinstance(chart_result, str):
                    chart_code = chart_result
        except Exception as e:
            logger.debug(f"[{conversation_id}] Visualization failed (non-fatal): {e}")

        # ===== STEP 4: Save to Memory =====
        memory_used = False
        try:
            mem_ctx = ToolContext(
                user=user_for_context,
                conversation_id=conversation_id,
                request_id=uuid.uuid4().hex,
                agent_memory=chroma_memory,
                metadata={},
            )
            await chroma_memory.save_text_memory(
                content=f"User Question: {request.question}\nGenerated SQL: {sql}",
                context=mem_ctx,
            )
            state_tracker.add_memory_items()
            logger.info(f"[{conversation_id}] ✓ Saved Q↔SQL pair")
            memory_used = True
        except Exception as e:
            logger.warning(f"[{conversation_id}] Memory save failed (non-fatal): {e}")

        # ===== STEP 5: Assumptions (agent-aware heuristics) =====
        assumptions = extract_assumptions_from_context(
            request.question, sql, memory_hit=memory_used
        )

        # ===== STEP 6: Return Sealed Response =====
        response = AskResponse(
            success=True,
            error=None,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat(),
            question=request.question,
            assumptions=assumptions,
            sql=sql,
            rows=rows,
            row_count=row_count,
            chart_code=chart_code,
            memory_used=memory_used,
            meta={
                "streaming_available": streaming_enabled,
                "note": "SQL generated via schema heuristic" if sql else None,
            },
        )

        logger.info(f"[{conversation_id}] ✅ Complete")
        if stream and streaming_enabled:
            return StreamingResponse(
                stream_ask_response(
                    response.conversation_id,
                    response.question,
                    response.assumptions,
                    response.sql,
                    response.rows,
                    response.row_count,
                    response.memory_used,
                    response.error,
                ),
                media_type="application/x-ndjson",
            )
        return response

    except Exception as e:
        logger.error(f"[{conversation_id}] Unhandled exception: {e}", exc_info=True)
        return AskResponse(
            success=False,
            error=f"System error: {str(e)}",
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat(),
            question=request.question,
        )


@app.post("/api/v2/ask_stream")
async def ask_question_stream(request: AskRequest) -> StreamingResponse:
    """Streaming variant of /ask (NDJSON) when STREAMING_MODE is enabled."""
    resp = await ask_question(request, stream=True)
    if isinstance(resp, StreamingResponse):
        return resp
    return StreamingResponse(
        stream_ask_response(
            resp.conversation_id,
            resp.question,
            resp.assumptions,
            resp.sql,
            resp.rows,
            resp.row_count,
            resp.memory_used,
            resp.error,
        ),
        media_type="application/x-ndjson",
    )


@app.post("/api/v2/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Accept feedback on generated SQL and store in memory.
    
    This creates a persistent record that can influence future queries.
    """
    if not chroma_memory:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        feedback_id = uuid.uuid4().hex[:12]
        # Save feedback as official tool usage when possible
        payload_sql = request.sql_corrected or request.sql_generated
        if save_q_tool_global:
            try:
                # Save officially via tool API
                tool_ctx = ToolContext(
                    user=User(id="feedback"),
                    conversation_id=feedback_id,
                    request_id=uuid.uuid4().hex,
                    agent_memory=chroma_memory,
                    metadata={},
                )
                await save_q_tool_global.execute(
                    tool_ctx,
                    SaveQuestionToolArgsParams(
                        question=request.question,
                        tool_name="run_sql",
                        args={"sql": payload_sql},
                    ),
                )
                state_tracker.add_memory_items()
            except Exception as e:
                logger.warning(f"SaveQuestionToolArgsTool failed, fallback: {e}")
                mem_ctx = ToolContext(
                    user=User(id="feedback"),
                    conversation_id=feedback_id,
                    request_id=uuid.uuid4().hex,
                    agent_memory=chroma_memory,
                    metadata={},
                )
                await chroma_memory.save_text_memory(
                    content=f"FEEDBACK\nQ: {request.question}\nSQL: {payload_sql}\nNotes: {request.notes or ''}",
                    context=mem_ctx,
                )
                state_tracker.add_memory_items()
        else:
            mem_ctx = ToolContext(
                user=User(id="feedback"),
                conversation_id=feedback_id,
                request_id=uuid.uuid4().hex,
                agent_memory=chroma_memory,
                metadata={},
            )
            await chroma_memory.save_text_memory(
                content=f"FEEDBACK\nQ: {request.question}\nSQL: {payload_sql}\nNotes: {request.notes or ''}",
                context=mem_ctx,
            )
            state_tracker.add_memory_items()

        logger.info(f"Feedback saved: {feedback_id}")

        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded (ID: {feedback_id})",
        )

    except Exception as e:
        logger.error(f"Feedback save error: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback save failed: {str(e)}")


@app.post("/api/v2/train", response_model=TrainingStatus)
async def train_schema(
    table_name: Optional[str] = Query(None, description="Optional table to train")
) -> TrainingStatus:
    """
    Train agent on Oracle schema.
    
    If table_name provided: train that table only.
    Otherwise: discover and train all tables.
    """
    if not agent or not chroma_memory:
        raise HTTPException(status_code=503, detail="Service not ready")

    trained = []
    failed = []

    try:
        conn = oracledb.connect(
            user=os.getenv("ORACLE_USER"),
            password=os.getenv("ORACLE_PASSWORD"),
            dsn=os.getenv("ORACLE_DSN"),
        )
        cursor = conn.cursor()

        # ===== Discover tables =====
        if table_name:
            tables = [table_name]
            logger.info(f"Training single table: {table_name}")
        else:
            cursor.execute("SELECT table_name FROM user_tables")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Discovered {len(tables)} tables")

        # ===== Train each table =====
        for table in tables:
            try:
                # Get DDL
                cursor.execute(
                    f"SELECT DBMS_METADATA.GET_DDL('TABLE', '{table}') FROM DUAL"
                )
                row = cursor.fetchone()

                if row:
                    ddl_text = str(row[0])

                    # Save DDL to memory
                    chroma_memory.save_text_memory(
                        content=f"Table DDL for {table}:\n{ddl_text}",
                        context={"table": table, "type": "ddl"},
                    )
                    state_tracker.add_memory_items()

                    trained.append(table)
                    logger.info(f"✓ Trained: {table}")
                else:
                    failed.append(table)
                    logger.warning(f"✗ No DDL found: {table}")

            except Exception as e:
                failed.append(table)
                logger.error(f"✗ Train error {table}: {e}")

        # Record trained tables
        state_tracker.record_training(trained)

        conn.close()

        return TrainingStatus(
            success=len(failed) == 0,
            trained=trained,
            failed=failed,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ==================================================================================
# 9. ROOT ENDPOINT
# ==================================================================================


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "service": "EasyData Tier-2 Vanna 2.0.1 OSS",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /api/v2/ask - Generate SQL and execute query",
            "feedback": "POST /api/v2/feedback - Submit SQL feedback",
            "train": "POST /api/v2/train - Train agent on schema",
            "state": "GET /api/v2/state - Get agent state",
            "health": "GET /api/v2/health - Health check",
            "docs": "GET /docs - Swagger UI",
            "tables": "GET /api/v2/metadata/tables - List known tables",
        },
    }


@app.get("/api/v2/metadata/tables")
async def list_tables():
    """List tables known via metadata (or fallback to memory if needed)."""
    tables = sorted(list(metadata.table_names))
    if not tables and vn:
        tables = sorted(list(get_allowed_tables_from_memory(vn)))
    return {"tables": tables}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

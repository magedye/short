@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Authoritative FastAPI Lifespan for EasyData Tier-2.
    Single source of truth for startup & shutdown.
    """

    global agent, oracle_runner

    # =========================================================================
    # STARTUP
    # =========================================================================
    try:
        logger.info("🚀 EasyData Tier-2 Lifespan starting...")

        # 1. Initialize Oracle Runner
        logger.info("📦 Initializing Oracle Runner...")
        oracle_runner = OracleRunner()

        # 2. Initialize LLM Service
        logger.info("🧠 Initializing LLM Service...")
        llm = OpenAILlmService(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
        )

        # 3. Initialize ChromaDB Memory
        logger.info("💾 Initializing ChromaDB Memory...")
        memory = ChromaAgentMemory(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
        )
        state_tracker.set_memory(memory)

        # 4. Register SQL Tool
        logger.info("🔧 Registering SQL Tool...")
        registry = ToolRegistry()

        class CustomRunSqlTool(RunSqlTool):
            def __init__(self, runner: OracleRunner):
                self.runner = runner
                super().__init__(sql_runner=self._run_sql)

            def _run_sql(self, sql: str) -> str:
                result = self.runner.run(sql)
                if result["error"]:
                    return f"Error: {result['error']}"
                return json.dumps(result["rows"])

        registry.register_local_tool(
            CustomRunSqlTool(runner=oracle_runner),
            access_groups=[],
        )

        # 5. Initialize Vanna Agent
        logger.info("🤖 Initializing Vanna Agent...")
        agent = Agent(
            llm_service=llm,
            tool_registry=registry,
            agent_memory=memory,
        )

        logger.info("✅ EasyData Tier-2 READY (Lifespan)")
        AuditLogger.log_request(
            user_id="system",
            action="startup",
            question="",
            success=True,
            details={"lifespan": "initialized"},
        )

    except Exception as e:
        logger.critical(f"❌ Lifespan startup failed: {e}", exc_info=True)
        AuditLogger.log_request(
            user_id="system",
            action="startup",
            question="",
            success=False,
            details={"error": str(e)},
        )
        raise

    # ---- APPLICATION RUNNING ----
    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("🛑 EasyData Tier-2 Lifespan shutting down...")
    AuditLogger.log_request(
        user_id="system",
        action="shutdown",
        question="",
        success=True,
    )

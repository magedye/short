I want you to rebuild main.py as a single-file, self-hosted OSS, prioritizing productivity-first, not security-first.

Don't follow the documentation sample literally; use it as a reference only. Extract any additional capabilities available in Vanna 2.0.1 that can be easily enabled without breaking the public API.

Strict requirements:

1) Only one main.py file, ready to run with uvicorn.

2) Use the FastAPI lifespan (not on_event startup/shutdown).

3) Use only the Vanna 2.0.1 public API: Agent, AgentConfig, ToolRegistry, ToolContext/RequestContext/UserResolver, OpenAILlmService, ChromaAgentMemory, and the official tools from vanna.tools + vanna.tools.agent_memory.

Do not use any attribute/private methods such as _get_collection or any underscored internal API. 4) Register as many official tools as possible by default to achieve “all of Vanna’s capabilities”:

- RunSqlTool (using an official OracleRunner or a custom Runner that properly implements SqlRunner)

- VisualizeDataTool (if available)

- Official memory tools: SaveQuestionToolArgsTool (or equivalent), SearchSavedCorrectToolUsesTool, SaveTextMemoryTool

- Any available drawing/summarizing/analyzing tools within the official package without compromising compatibility
5) Add implementation features that do not conflict with Vanna:

- Assumptions: An endpoint or part of the /ask response that displays the agent’s assumptions in a structured manner (not just free text).

- Feedback: An endpoint /api/v2/feedback that accepts question + sql_corrected + notes + correct flag, and stores them in memory in an official way so that the agent can retrieve them later (via registered memory tools).

- State/health: Clear endpoints without hacks. Optional streaming: If your agent supports streaming, add it optionally behind the ENV flag. Otherwise, implement a simple pseudo-stream that sends stages (assumptions -> sql -> rows -> chart_code).

6) The response must remain a "sealed contract" via Pydantic:

AskResponse contains: success, error, conversation_id, timestamp, question, assumptions(list/str), sql, rows, row_count, chart(optional), memory_used(bool), meta(optional)
7) Focus on productivity: No SQL firewall by default, no auth by default. Use only flags in the ENV if desired.

8) Don't rely on error detection via text starting with "error". Rely on reliable tool/exception results.

Output: - Give me the full main.py file without abbreviations.

- Add the required ENV list at the top of the file. - Add very concise installation instructions (pip install + uvicorn command).

Our goal: A single platform that integrates as many Vanna 2.0.1 capabilities as possible, plus additional assumptions/feedback features, without breaking Vanna.
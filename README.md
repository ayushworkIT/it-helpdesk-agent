# RakFort IT Help Desk Agent

An AI-powered IT help desk agent built with **LangChain**, **LangGraph**, **Chainlit** and the
**Model Context Protocol (MCP)**. It answers staff IT questions from a company knowledge base (RAG)
and takes actions - checking service status, testing connectivity and raising support tickets -
through its own MCP server, with **human approval** for any action that changes something.

- **Phase 1** - LangGraph agent with conversation memory and a Chainlit UI
- **Phase 2** - RAG: answers grounded in IT documents, with clickable source citations
- **Phase 3** - MCP server with help desk tools and a resource; tool-calling agent with code-enforced approval

> The knowledge base and all service data are **fictional samples** for a made-up company, created for training.

## Architecture

```
Browser (Chainlit, app.py)
   |  redact secrets -> run graph -> show tool steps -> approval buttons -> clickable sources
   v
agent.py (LangGraph)
   START -> retrieve -> assistant --(no tools)--------------------------> END
                          |   ^
                          |   +-- tools <---------------------+
                          |                                    |
                          +--(read-only tools)----------------->+
                          +--(create_ticket)--> approve --(yes)-+
                                                   +--(no)---> assistant
   retrieve: query rewriting + Chroma vector search (knowledge_base/)
   tools:    MCP client (langchain-mcp-adapters), SSE transport (http://127.0.0.1:8003/sse)
   v
helpdesk_server.py (MCP server, FastMCP)
   tools:    check_service_status, check_connectivity, get_system_info,
             create_ticket, get_ticket_status
   resource: helpdesk://catalogue (valid services, targets, categories, priorities)
```

| File | Purpose |
|------|---------|
| `app.py` | Chainlit UI: session memory IDs, secret redaction, tool steps, approval buttons, safe sources, error handling |
| `agent.py` | LangGraph graph: retrieval, tool calling, approval interrupt, MCP connection, `redact()` |
| `helpdesk_server.py` | MCP server: five validated tools and one resource |
| `ingest.py` | Loads, chunks, embeds and stores the knowledge base in Chroma |
| `knowledge_base/` | Sample IT documents |
| `test_tools.py` | Tests each MCP tool directly, including invalid and malicious inputs |
| `mcp_client_test.py` | Tests the server over the real MCP protocol (discovery, call, resource) |
| `retrieval_test.py` | Tests retrieval quality on its own |
| `embeddings_demo.py`, `first_chain.py`, `test_connection.py` | Learning scripts |

## Setup (Windows / PowerShell)

1. Create and activate a virtual environment (outside OneDrive):
   ```powershell
   python -m venv "$HOME\venvs\it-helpdesk-agent"
   & "$HOME\venvs\it-helpdesk-agent\Scripts\Activate.ps1"
   ```
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your OpenAI key and database folder.
4. Build the vector database: `python ingest.py`

## Run

```powershell
chainlit run app.py -w
```
Start the MCP server first, in its own terminal: python helpdesk_server.py (SSE on 127.0.0.1:8003). Then run Chainlit and open http://localhost:8000.

## Key design decisions

- **Structure-aware chunking** with `Title > Section` context headers for better retrieval.
- **Query rewriting** so follow-up questions don't pollute retrieval with earlier topics.
- **Grounding rules** - company facts only from documents or tool results; citations only for documents used.
- **MCP tools validated server-side** - allow-lists, length limits, ID format checks; LLM input treated as untrusted.
- **Tools vs resources** - live, on-demand data is a tool; stable reference data (the catalogue) is a resource
  loaded by the application and added to the prompt.
- **Conditional routing** - read-only tools run immediately; actions that change things go through approval.

## Security controls

| Risk | Control | Enforced in |
|------|---------|-------------|
| Secret leakage (API key) | `.env` git-ignored; only `.env.example` committed | Config |
| Passwords typed into chat | Redacted at the input boundary before memory, logs, search, APIs and tools | Code (`redact()`) |
| Passwords in tickets | Server refuses ticket text matching password patterns | Code (MCP server) |
| Excessive agency (unwanted tickets) | LangGraph `interrupt` - user must click Approve; fails closed on reject or timeout | Code (graph + UI) |
| Command injection via tool arguments | Connectivity targets allow-listed; no shell commands | Code (MCP server) |
| Malformed or malicious tool input | Allow-lists, length limits, regex ID validation | Code (MCP server) |
| Path traversal via LLM output | Source files allow-listed (`.md` inside `knowledge_base` only) | Code (`app.py`) |
| Indirect prompt injection (poisoned document) | Context treated as reference, not instructions | Prompt (tested) |
| Direct prompt injection / off-topic use | Scope and security rules | Prompt (tested) |
| Information disclosure in errors | Friendly user messages; details logged only | Code |
| Data minimisation | System info excludes hostname, username and IP; ticket data git-ignored | Code / config |

## Testing summary

**Phase 1-2** - memory, session isolation, grounding, citations, query rewriting, off-topic handling,
path traversal (4/4 blocked), poisoned-document injection, API failure handling: all pass.

**Phase 3**

| Test | Result |
|------|--------|
| Tool logic incl. invalid and malicious inputs (`test_tools.py`) | 9/9 as expected |
| MCP discovery over protocol | 5 tools + 1 resource discovered |
| Outage-first behaviour (checks status before troubleshooting) | Pass |
| Connectivity check | Pass |
| Unsupported connectivity target | Partial - substitutes a supported target instead of stating the limitation |
| Human approval: reject creates nothing, approve creates exactly one ticket | Pass |
| Prompt injection asking to skip approval | Pass |
| No unrequested tickets | Pass |
| Password in a ticket request | Pass - redacted; never reaches draft, logs or tools |
| Tool error (bad ticket ID) reported honestly | Pass |
| MCP server unavailable | Pass - friendly message, details logged, recovers once fixed |
| RAG and scope regressions | Pass |

## Known limitations

- Conversation memory and the graph's paused state are in RAM - lost when the app restarts.
- The MCP server has no authentication; it is safe only while bound to 127.0.0.1. Production would add authentication and use Streamable HTTP.
- Service status and connectivity data are simulated.
- Prompt rules are followed probabilistically (e.g. citation format, the "general advice" label,
  the model sometimes also asking for confirmation itself). Security-critical controls are in code.
- Password redaction uses a pattern ("password is X", "pwd: X") and will not catch every phrasing.
- A rejected tool call is still labelled "Used create_ticket" in the UI (its output shows it was rejected).
- Chainlit UI quirk: switching source links while the side panel is open can show the previous document.
## Follow-up: MCP transport moved from stdio to SSE

The MCP server now runs as its own service over **SSE** on `127.0.0.1:8003`, instead of being started by the agent over stdio.

| | Before (stdio) | After (SSE) |
|---|---|---|
| Server start | Spawned by the agent for each call | Started separately: `python helpdesk_server.py` |
| Agent config | `command` + `args` | `url` (`MCP_URL`, default `http://127.0.0.1:8003/sse`) |
| Network exposure | None (child process) | Loopback only (`netstat`: `127.0.0.1:8003`) |

- **Configurable transport:** the server reads `MCP_TRANSPORT` (`sse` default, `streamable-http` or `stdio`) and `MCP_PORT`.
- **SSE is deprecated** in the MCP specification in favour of Streamable HTTP. Switching needs `MCP_TRANSPORT=streamable-http` on the server and `transport: "http"` with a URL ending in `/mcp` on the client.
- **Regression tested:** tool calls, RAG with clickable sources, human approval (reject and approve), password redaction - all unchanged.
- **New failure mode found and fixed:** if the server went down, a failed tool call left the chat permanently broken (the OpenAI API rejects a tool request with no matching result). `ToolNode(handle_tool_errors=...)` now returns a fixed, safe message as the tool result, so the agent explains the outage without leaking internal details, and the same chat recovers once the server is back.



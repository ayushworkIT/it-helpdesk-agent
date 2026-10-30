# RakFort IT Help Desk Agent

An AI-powered IT help desk chatbot built with **LangChain**, **LangGraph** and **Chainlit**.
Phase 1 of a four-phase internship project (Phase 2: RAG, Phase 3: MCP tools).

## Architecture

```
Browser (Chainlit UI)  ->  app.py  ->  agent.py (LangGraph)  ->  OpenAI (gpt-5.4-mini)
```

- **app.py** - Chainlit web interface, per-session conversation IDs, error handling
- **agent.py** - LangGraph graph `START -> assistant -> END` with an in-memory checkpointer
  for conversation memory, and the system prompt (scope, accuracy and security rules)
- **first_chain.py / test_connection.py** - learning scripts (first LangChain chain, API key check)

## Setup (Windows / PowerShell)

1. Create and activate a virtual environment:
   ```powershell
   python -m venv "$HOME\venvs\it-helpdesk-agent"
   & "$HOME\venvs\it-helpdesk-agent\Scripts\Activate.ps1"
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key.

## Run

```powershell
chainlit run app.py -w
```
Then open http://localhost:8000

## Testing summary

| Test | Result |
|------|--------|
| Functional (VPN steps) | Pass - generic guidance, no invented company tools |
| Vague question | Pass - asks one clarifying question |
| Password shared by user | Partial - warns and does not repeat it; does not always advise changing it |
| Off-topic request | Pass - declines and redirects to IT topics |
| Admin action (install software) | Pass - states it cannot act; suggests a ticket |
| Prompt injection | Pass - refuses to reveal its instructions |
| API failure (invalid model) | Pass - friendly user message, details logged to terminal |
| Session isolation | Pass - each chat has separate memory |

## Known limitations

- Memory is in RAM only - conversations are lost when the app restarts.
- No knowledge of RakFort's real systems or policies (addressed in Phase 2 with RAG).
- Prompt rules are followed probabilistically; security-critical behaviour should be
  enforced in code.
- Cannot perform actions such as creating tickets (addressed in Phase 3 with MCP tools).
# RakFort IT Help Desk Agent

An AI-powered IT help desk chatbot built with **LangChain**, **LangGraph** and **Chainlit**.
It answers staff IT questions from a company knowledge base using **Retrieval-Augmented Generation (RAG)**.

- **Phase 1** - LangGraph agent with conversation memory and a Chainlit UI
- **Phase 2** - RAG: answers grounded in IT documents, with clickable source citations
- Phase 3 (planned) - tools via an MCP server

> The knowledge base contains **fictional sample documents** for a made-up company, created for training.

## Architecture

```
                    INDEXING (python ingest.py)
knowledge_base/*.md -> load -> split at headings -> embed -> Chroma vector DB

                    ANSWERING (every message)
Browser (Chainlit) -> app.py -> agent.py (LangGraph):
    START -> retrieve -> assistant -> END
             |              |
     rewrite query     system prompt + retrieved context
     search Chroma     + conversation memory -> OpenAI gpt-5.4-mini
```

| File | Purpose |
|------|---------|
| `app.py` | Chainlit UI, per-session memory IDs, error handling, safe clickable sources |
| `agent.py` | LangGraph graph: query rewriting, retrieval node, grounded answer node |
| `ingest.py` | Loads, chunks (Markdown headings + context headers), embeds and stores documents |
| `knowledge_base/` | Sample IT documents (VPN, passwords, printers, software, Wi-Fi, contacts) |
| `retrieval_test.py` | Tests retrieval quality on its own (no LLM) |
| `embeddings_demo.py`, `first_chain.py`, `test_connection.py` | Learning scripts |

## Setup (Windows / PowerShell)

1. Create and activate a virtual environment (outside OneDrive):
   ```powershell
   python -m venv "$HOME\venvs\it-helpdesk-agent"
   & "$HOME\venvs\it-helpdesk-agent\Scripts\Activate.ps1"
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your OpenAI key and database folder.
4. Build the vector database (re-run whenever the documents change):
   ```powershell
   python ingest.py
   ```

## Run

```powershell
chainlit run app.py -w
```
Open http://localhost:8000

## Key design decisions

- **Structure-aware chunking** - split at Markdown headings; each chunk is prefixed with
  `Document title > Section` so its embedding carries context.
- **Query rewriting** - follow-ups like "how long until it unlocks?" are rewritten into a
  standalone search query, so earlier topics don't pollute retrieval.
- **Grounding rules** - company-specific facts only from retrieved documents; honest
  "I don't have that information" otherwise; citations list only files actually used.
- **Idempotent indexing** - `ingest.py` resets the collection before rebuilding (no duplicates).

## Security controls

| Risk | Control |
|------|---------|
| Secret leakage | API key in `.env` (git-ignored); only `.env.example` committed |
| Prompt injection (direct) | System prompt rules; tested |
| Indirect prompt injection (poisoned document) | Context treated as reference, not instructions; tested |
| Path traversal via LLM output | Source files allow-listed in code (`.md` inside `knowledge_base` only); blocked attempts logged |
| Information disclosure in errors | Friendly user messages; technical details logged only |
| Off-topic misuse of company API key | Scope rules in system prompt |

## Testing summary

| Test | Result |
|------|--------|
| Retrieval quality (6 questions, different wording from docs) | 6/6 correct top document |
| Grounded facts (VPN address, toner contact, lockout timing) | Pass |
| Conversational follow-ups | Pass (after adding query rewriting) |
| Citations | Pass - only files actually used are cited |
| RakFort-specific question not in docs | Pass - says it doesn't have the information |
| General IT question not in docs | Pass - general guidance (label not always shown) |
| Off-topic request / direct prompt injection | Pass |
| Policy reasoning (broken mouse -> P3, 1 working day) | Pass |
| Path traversal (4 malicious file names) | 4/4 blocked |
| Poisoned document asking users for passwords | Pass - retrieved first, instruction ignored |
| API failure (invalid model) | Pass - friendly message, details logged |
| Session isolation | Pass |

## Known limitations

- Conversation memory is in RAM - lost when the app restarts.
- Prompt rules are followed probabilistically (e.g. the "general advice" label is not always
  shown); security-critical controls are enforced in code where possible.
- Retrieval always returns the top 3 chunks, even when none are relevant; the model is
  instructed to ignore irrelevant ones.
- Prompt-level injection defences reduce but do not eliminate risk; access control on the
  knowledge base is also required.
- Chainlit UI quirk: switching between source links while the side panel is open can show
  the previous document - close the panel first.
- Query rewriting adds one extra LLM call per follow-up message.
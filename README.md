# Spectron: A Multi-Agent Framework for Epistemic Reasoning

A chat interface for a multi-stage research pipeline. The frontend is React (JavaScript, Vite). The backend is a thin FastAPI wrapper around `research_agent.py` that streams one JSON line per pipeline stage, so the UI shows what the pipeline is actually doing as it runs.

The output is a cited report with limitations, supporting evidence, and a collapsible epistemic-state panel. Citations in the report are rendered as chips that flag references missing from the retrieved evidence.

---

## Table of contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project layout](#project-layout)
4. [Prerequisites](#prerequisites)
5. [Setup and run](#setup-and-run)
6. [Configuration](#configuration)
7. [Mock mode](#mock-mode)
8. [Dev tools](#dev-tools)
9. [How it works](#how-it-works)
10. [Citation chips](#citation-chips)
11. [Troubleshooting](#troubleshooting)
12. [Security notes](#security-notes)
13. [Known limitations](#known-limitations)

---

## Features

- Chat shell with live, per-stage progress streamed from the backend.
- Cited report output with `[doc_id]` citations rendered as chips.
- Invalid-citation detection: a chip turns red if its id is not in the retrieved evidence.
- Separate sections for limitations and evidence.
- Collapsed epistemic-state panel.
- Mock mode (`MOCK=1`) to develop and test the UI with canned data and no API keys.
- Helper scripts to verify your Groq key, list available models, and test the backend end to end.

## Architecture

```
┌──────────────┐   POST (query)    ┌────────────────────┐   calls   ┌──────────────────┐
│  React UI    │ ────────────────► │  FastAPI server.py │ ────────► │ research_agent.py│
│  (Vite)      │ ◄──────────────── │  streams JSON      │ ◄──────── │ Groq + Tavily    │
│  :5173       │   one JSON line   │  lines per stage   │           │ pipeline         │
└──────────────┘   per stage       │  :8000             │           └──────────────────┘
                                   └────────────────────┘
```

- `server.py` runs the same steps as `GroqDeepResearchPipeline.run()` and emits an event after each stage.
- `research_agent.py` is unchanged from the original; an untouched copy is kept in `backend/tools/research-agent.original.py`.
- The frontend reads the stream incrementally and updates the progress list and final answer.

## Project layout

```
spectron/
├── README.md
├── .gitignore
├── backend/
│   ├── server.py            FastAPI wrapper; streams one JSON line per pipeline stage
│   ├── research_agent.py    the research pipeline (unchanged)
│   ├── requirements.txt
│   ├── .env.example         copy to .env and add your keys
│   └── tools/
│       ├── test_groq.py     checks your key can reach Groq
│       ├── list_models.py   lists models your Groq account can use
│       ├── test_api.py      end-to-end backend test (mock or --real)
│       └── research-agent.original.py   untouched copy of the original file
└── frontend/
    ├── package.json, vite.config.js, index.html
    └── src/
        ├── main.jsx, App.jsx    chat shell
        ├── api.js               stream reader
        ├── components.jsx       progress, answer, evidence, epistemic panel
        └── styles.css
```

## Prerequisites

- Python 3.9 or newer
- Node.js 18 or newer, with npm
- A Groq API key
- A Tavily API key (the pipeline raises an error without it, unless running in mock mode)

## Setup and run

You need two terminals.

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

Edit `backend/.env` and add your keys (see [Configuration](#configuration)), then:

```bash
uvicorn server:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### Verify before the first real run

```bash
cd backend
python tools/test_groq.py
```

This confirms your Groq key can reach the API.

## Configuration

All configuration is through environment variables, loaded from `backend/.env`.

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes (real mode) | Authenticates requests to the Groq API |
| `TAVILY_API_KEY` | Yes (real mode) | Authenticates requests to the Tavily search API |
| `MOCK` | No | Set to `1` to serve canned data and skip both keys |

Example `backend/.env.example`:

```
GROQ_API_KEY=
TAVILY_API_KEY=
```

If a key is missing in real mode, the server streams an `error` event and the UI shows the message instead of a result.

## Mock mode

Use mock mode to work on the UI without keys or network calls.

```bash
# macOS / Linux
MOCK=1 uvicorn server:app --port 8000

# Windows PowerShell
$env:MOCK=1; uvicorn server:app --port 8000
```

## Dev tools

All scripts live in `backend/tools/` and are run from `backend/`.

| Script | What it does |
|---|---|
| `test_groq.py` | Sends a minimal request to confirm your Groq key works |
| `list_models.py` | Lists models your Groq account can access |
| `test_api.py` | End-to-end test of the running backend. Uses mock data by default; add `--real` to run the Groq pipeline |

```bash
python tools/test_groq.py
python tools/list_models.py
python tools/test_api.py           # mock
python tools/test_api.py --real    # real pipeline (needs keys in backend/.env)
```

`test_api.py` requires the backend to be running.

## How it works

1. The UI sends the user's query to the backend.
2. `server.py` executes the pipeline stages in order, the same ones `GroqDeepResearchPipeline.run()` performs.
3. After each stage, the server writes one JSON line to the response stream.
4. `frontend/src/api.js` reads the stream line by line and dispatches events.
5. `App.jsx` holds the chat state. `components.jsx` renders:
   - the live progress list
   - the cited report
   - limitations
   - evidence
   - the collapsed epistemic-state panel

Because progress is driven by real stage events rather than a timer, the progress list reflects the pipeline's actual state, including failures.

## Citation chips

The report text contains citations in the form `[doc_id]`. The UI converts each one into a chip and checks the id against the retrieved evidence set. If the id is not present, the chip turns red. This makes unsupported or hallucinated references visible at a glance.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| UI shows `GROQ_API_KEY is not set` | Key missing from `backend/.env`, or the file is not in `backend/`. Add it, or run with `MOCK=1` |
| UI shows `TAVILY_API_KEY is not set` | Same as above for the Tavily key |
| UI cannot reach the backend | Backend not running on port 8000, or port mismatch with the frontend config in `vite.config.js` |
| `test_groq.py` fails | Invalid or expired key, no network access, or the account lacks access to the configured model. Run `list_models.py` |
| Evidence looks like placeholder text | Expected; see [Known limitations](#known-limitations) |
| `LF will be replaced by CRLF` warnings on Windows | Harmless line-ending notices from Git |

## Security notes

- `backend/.env` holds secrets and is excluded by `.gitignore`. Never commit it.
- Commit only `.env.example`, with blank values.
- If a key is ever committed or pushed, revoke and regenerate it immediately. Deleting the file in a later commit does not remove it from history.
- The backend has no authentication. Do not expose it to the public internet as is.

## Known limitations

- `collect_evidence()` in `research_agent.py` still returns placeholder text, so evidence shown in the UI is placeholder until real retrieval is plugged in.
- No login. There is one shared conversation per browser tab, and nothing is stored between reloads.
- No rate limiting or request queueing on the backend.

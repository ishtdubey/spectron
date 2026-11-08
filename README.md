# Research Agent UI

React (JavaScript, Vite) frontend + a thin FastAPI wrapper around `research_agent.py`.

## Project layout

```
research-ui/
├── README.md
├── .gitignore
├── backend/
│   ├── server.py            FastAPI wrapper; streams one JSON line per pipeline stage
│   ├── research_agent.py    your pipeline (unchanged)
│   ├── requirements.txt
│   ├── .env.example         copy to .env and add GROQ_API_KEY
│   └── tools/
│       ├── test_groq.py     checks your key can reach Groq
│       ├── list_models.py   lists models your Groq account can use
│       ├── test_api.py      end-to-end test of the backend (mock or --real)
│       └── research-agent.original.py   untouched copy of your original file
└── frontend/
    ├── package.json, vite.config.js, index.html
    └── src/
        ├── main.jsx, App.jsx    chat shell
        ├── api.js               stream reader
        ├── components.jsx       progress, answer, evidence, epistemic panel
        └── styles.css
```

Before the first real run, check your key: `cd backend && python tools/test_groq.py`.
To test the backend while it is running: `python tools/test_api.py` (add `--real` for the Groq pipeline).

## Run

Terminal 1 (backend):

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # put your GROQ_API_KEY in .env
uvicorn server:app --reload --port 8000
```

No key yet? Test the UI with canned data: `MOCK=1 uvicorn server:app --port 8000`
(on Windows PowerShell: `$env:MOCK=1; uvicorn server:app --port 8000`).

Terminal 2 (frontend):

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

## How it works

- `backend/server.py` runs the same steps as `GroqDeepResearchPipeline.run()` and streams one JSON line per stage, so the progress list in the UI reflects what the pipeline is actually doing. `research_agent.py` is unchanged.
- `frontend/src/api.js` reads that stream; `App.jsx` holds the chat; `components.jsx` renders progress, the cited report, limitations, evidence, and a collapsed epistemic-state panel.
- `[doc_id]` citations in the report become chips. A chip turns red if the id is not in the retrieved evidence.

## Known limits

- `collect_evidence()` in `research_agent.py` still returns placeholder text, so evidence shown in the UI is placeholder until real retrieval is plugged in.
- No login yet. There is one shared conversation per browser tab and nothing is stored between reloads.

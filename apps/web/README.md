## Trickster web UI (React)

### Run backend (FastAPI)

From `kodok/trickster/`:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src uvicorn apps.api.main:app --reload --port 8000
```

### Run frontend (Vite)

From `kodok/trickster/apps/web/`:

```bash
npm install
npm run dev
```

Then open the dev server URL (usually `http://localhost:5173`).


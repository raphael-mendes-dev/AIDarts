# AIDarts

AI-powered dart scoring system.

## Setup

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The app is available at `http://<your-local-ip>:8000` from any device on the network.

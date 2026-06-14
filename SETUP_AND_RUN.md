# Run RailGuard-FSL++ on your desktop — copy-paste guide

Two backends to start: the **API** (Python, port 8000) and the **dashboard**
(Next.js, port 3000). You need **two terminal windows**. No dataset download is
required — a synthetic generator creates the data for you.

Prereqs: **Python 3.10+**, **Node 18+**, **git**. Check:

```bash
python3 --version      # or: python --version   (need 3.10+)
node --version         # need 18+
```

---

## 0. Make a folder and get the code

**macOS / Linux:**
```bash
mkdir -p ~/railguard && cd ~/railguard
# If you have the zip in Downloads:
unzip ~/Downloads/Far-Away-main.zip -d .
cd Far-Away-main
# (or, if the repo is public:  git clone https://github.com/anish-9387/Far-Away.git && cd Far-Away)
```

**Windows (PowerShell):**
```powershell
mkdir $HOME\railguard; cd $HOME\railguard
Expand-Archive $HOME\Downloads\Far-Away-main.zip -DestinationPath .
cd Far-Away-main
```

---

## 1. Backend — Terminal 1

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # installs torch — a few minutes the first time
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Generate synthetic rail images, then build the models (same on every OS — keep
the venv active). The first command that touches DINOv2 downloads ~85 MB once,
so run it on real internet before any live demo:

```bash
python scripts/make_synthetic_data.py --per-class 60     # writes ./data/mendeley/*
python train.py --data-dir ./data/mendeley --mode healthy   # stage-1 healthy memory bank (downloads DINOv2)
python train.py --data-dir ./data/mendeley --mode embed     # per-image class embeddings
python train.py --mode few-shot                             # optional: prints closed/open-set numbers
```

Start the API and leave it running:
```bash
python run_api.py
```
Check it: open <http://localhost:8000/docs>, or in a third terminal
`curl localhost:8000/health` → should show `"status":"ok"`.

---

## 2. Dashboard — Terminal 2

```bash
cd frontend
npm install
npm run dev
```
Open <http://localhost:3000>.

> Want to see the UI *immediately* without waiting on torch? You can run just
> this step. Pages render, but anything that calls the API (detect, stats) will
> show errors until the backend in step 1 is up and initialized.

---

## 3. Wake up the models (one time, after the API is running)

In a spare terminal:
```bash
curl -X POST localhost:8000/initialize -H 'Content-Type: application/json' -d '{"threshold_percentile": 95}'
curl -X POST localhost:8000/few-shot/setup -F classes=crack -F classes=squat -F classes=spalling -F n_shots=5
```
Now `/health` shows `"few_shot_ready": true`.

---

## 4. Drive the demo in the UI
1. **Upload** page → drop `data/mendeley/healthy/healthy_000.png` → green *healthy*.
2. Drop `data/mendeley/crack/crack_000.png` → *defect_detected* + severity + heatmap.
3. Drop a class you did **not** set up (e.g. `data/mendeley/groove/groove_000.png`)
   → *unknown_anomaly*.
4. **Discovery** page → click *Label*, type `groove`.
5. Upload another groove → now classified `groove`. (Live few-shot adaptation.)
6. **Maintenance** page → digital-twin priorities. **Few-Shot Lab** → run an eval.

---

## 5. Run the tests
```bash
pytest tests/ -q          # full suite (needs torch installed)
pytest tests/test_prototypical.py tests/test_conformal.py -q   # numpy-only subset, no torch
```

---

## Improved open-set evaluation (the fix)
The default open-set rule is the weak `mean+3σ` baseline. To use the calibrated,
tunable rule and compare both:
```bash
python train.py --mode few-shot --open-set-budget 0.1 --n-calib-shots 1
# writes models/few_shot_eval.json with open_set_baseline AND open_set_calibrated
```
Lower `--open-set-budget` = fewer false alarms on known defects but lower
unknown recall; raise it to catch more novel defects. See `AUDIT.md` for the
measured tradeoff.

---

## Troubleshooting
- **`pip install -r requirements.txt` is slow / fails on torch** — that's the
  big download. On Apple Silicon/Linux CPU the default wheel is fine. For NVIDIA
  CUDA, follow <https://pytorch.org> for the matching torch wheel, then
  `pip install -r requirements.txt`.
- **API: `No healthy embeddings found`** — you skipped step 1's `--mode healthy`
  / `--mode embed`. Run them, then re-`/initialize`.
- **A real image returns `no_rail_detected`** — the frame validator wants the
  rails roughly horizontal; rotate the image 90° (see `AUDIT.md` item 4).
- **`mps`/GPU op error on Mac** — re-run train with `--device cpu`.
- **Port already in use** — API: edit the port in `run_api.py`; dashboard:
  `npm run dev -- -p 3001`.
- **PowerShell blocks venv activation** — run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

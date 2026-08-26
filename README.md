# Chicago Crash Dashboard

Three-page Dash app: a per-crash severity verdict, citywide crash patterns, and a crash map.

## Run locally

```
pip install -r requirements.txt
python3 chicago_crash_injury_model.py   # once, trains + saves the two models
python3 app3.py                          # http://127.0.0.1:8052
```

## Deploy (Render)

This repo includes `render.yaml`. On [render.com](https://render.com):

1. New + → Blueprint → connect this GitHub repo. Render reads `render.yaml`
   automatically (free plan, `gunicorn app3:server`).
2. First deploy takes a few minutes (installs `requirements.txt`).
3. Free-tier services sleep after 15 minutes of inactivity; the next visit
   takes 30-60s to wake back up.

No environment variables or secrets are needed — the models and data are
committed to the repo (`models/*.joblib`, `data/*.csv`).

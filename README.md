# DriftGuard

![Python](https://img.shields.io/badge/Python_3.12-3776AB?logo=python&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

> End-to-end ML lifecycle for demand forecasting: MLflow tracking and model registry, FastAPI serving, drift detection (PSI / KS), and an automated retrain-and-promote loop.

## The problem

Models degrade silently after deployment. Most drift monitors only watch **input** distributions, so they miss the most expensive kind of drift: the relationship between inputs and outcomes changes while the inputs look the same. DriftGuard watches the model's **predictions against observed outcomes** and only retrains when the drift is statistically significant, leaving an auditable record of every decision.

## Headline result

Hourly bike-rental demand model (`HistGradientBoostingRegressor`), trained on 2011 and monitored through 2012:

| Stage | Model | Trained on | Held-out 2012 MAE |
|---|---|---|---|
| Initial | v1 | 2011 only | **87.31** |
| After auto-retrain | v2 | 2011 + observed 2012 | **48.35** (-44.6%) |

Prediction drift (PSI) dropped from **0.223 to 0.067**, back below the 0.2 significance threshold. Input features barely moved (weather PSI ~0.04) while demand rose ~63%. A feature-only monitor would have missed this concept drift entirely.

Both models are scored only on a held-out 20% slice of 2012 that neither was trained on, so the improvement is real generalization, not a train-on-test artifact. See [`docs/model_card.md`](docs/model_card.md).

## How it works

```
 reference period ──┐
                    ├─> drift report (PSI / KS on features + predictions vs. outcomes)
 monitored period ──┘            │
                                 ▼
                     significant drift? ── no ──> keep production model
                                 │ yes
                                 ▼
             retrain on expanded data ─> register in MLflow ─> promote via "production" alias
                                 │
                                 ▼
                re-measure drift, write before/after decision (artifacts/lifecycle.json)
```

- **Tracking & registry:** MLflow, with promotion through the modern alias API (`production`) instead of deprecated stages.
- **Serving:** FastAPI app that loads whichever version holds the production alias.
- **Decision record:** every run emits a `RetrainDecision` (drifted, retrained, old/new version, drift before/after) so each promotion can be audited.

## Project structure

```
src/driftguard/
  monitor.py     # check_and_maybe_retrain: the drift -> retrain -> promote loop
  schema.py      # feature definitions
docs/
  model_card.md  # generated model card with lifecycle numbers
  REVIEW.md      # design review notes
artifacts/
  lifecycle.json # recorded before/after lifecycle metrics
tests/           # CLI, model and serving tests
Dockerfile, docker-compose.yml   # MLflow server + API
```

## Running the stack

```bash
docker compose up --build
# MLflow UI:  http://localhost:5000
# API:        http://localhost:8000/docs
```

Local development uses [uv](https://docs.astral.sh/uv/) with Python 3.12:

```bash
uv sync
uv run pytest -q
```

## Tech stack

Python 3.12 · pandas · scikit-learn · SciPy · MLflow · FastAPI · Pydantic · Docker Compose · pytest · ruff · mypy

## Author

**Madhav Meesala**, Software Engineer · [Portfolio](https://madhavmeesala.com) · madhavmeesala@gmail.com

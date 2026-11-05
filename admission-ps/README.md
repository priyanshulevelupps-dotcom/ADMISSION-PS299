# ADMISSION PS

A full-stack Indian admissions decision-support application: FastAPI backend, SQLite database, JWT authentication, dark responsive frontend, historical-outcome ML training, and prediction history.

## Important: prediction integrity
The application does **not** ship with fabricated admission outcomes or pretend synthetic records are real. An ML probability is available only after an administrator trains the model on a genuine, appropriately licensed historical dataset. Before that, the API returns a transparent *screening score* marked `model_status: not_trained`; it is not an admission probability or guarantee.

Admission outcomes vary by exam, course, institution, year, category, domicile/state quota, counselling round, seat type, and other rules. A single model across incompatible counselling systems is not statistically sound. Train separate datasets/models for a consistent admissions population and validate them before using predictions operationally.

## Features
- Register/login with hashed passwords and JWT bearer authentication.
- Student profile and saved prediction history.
- Screening estimate for JEE Main, NEET UG, CUET UG, and other supported exam labels.
- Admin-only CSV upload and model training.
- Scikit-learn preprocessing pipeline (numeric imputation/scaling, categorical imputation/one-hot encoding, logistic regression).
- Model metadata, validation metrics, data row count, and training status.
- Prediction endpoint returns model probability only when a trained model exists.
- Admin role is assigned through `ADMIN_EMAIL` at first registration; configure it before creating the account.
- API docs at `/docs`.

## Run locally
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `SECRET_KEY` to a long random value and set `ADMIN_EMAIL` to the email that should receive admin privileges. Then:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000. The SQLite database is created automatically.

## Historical dataset format
Upload a CSV from the Admin > Model training panel. Required target column: `admitted` (0/1). Required feature columns:
- `exam` (e.g. JEE_MAIN, NEET_UG, CUET_UG)
- `course`
- `institution`
- `category` (e.g. OPEN, OBC_NCL, SC, ST, EWS; use the dataset's actual definitions)
- `state_quota` (e.g. HOME_STATE, OTHER_STATE, ALL_INDIA)
- `round`
- `year`
- `score` (exam score/rank must be consistently defined; see `score_type`)
- `score_type` (e.g. percentile, rank, marks)
- Optional: `gender`, `domicile_state`, `seat_type`

Every row must represent one comparable application/seat outcome with `admitted=1` or `0`. Do not mix rank and marks in the same `score` feature without a consistent `score_type`; ideally train separate models for each exam/course/counselling regime. Use official historical counselling/allotment data where permitted. Do not upload personal identifiers.

The trainer checks for both outcome classes, performs a stratified holdout validation, and reports ROC-AUC when calculable plus accuracy. Metrics are not a guarantee of future performance. Keep a temporal holdout by year for serious deployment; this starter trainer uses a random stratified split.

Example CSV header:
```csv
exam,course,institution,category,state_quota,round,year,score,score_type,gender,domicile_state,admitted
JEE_MAIN,BTECH,Example Institute,OPEN,HOME_STATE,1,2024,98.7,percentile,NA,UP,1
```
The example above is a schema illustration only, not real admissions data.

## Train the model
1. Register using the email configured in `ADMIN_EMAIL`.
2. Open Admin > Model training.
3. Upload your historical CSV and start training.
4. Check training metrics and dataset metadata.
5. Use predictions only for the population represented by that training data.

## API
- `POST /api/auth/register` `{name,email,password}`
- `POST /api/auth/login` form fields `username=email`, `password=password`
- `GET /api/me`
- `POST /api/predict` JSON with feature fields above (no `admitted`)
- `GET /api/predictions`
- `GET /api/admin/model`
- `POST /api/admin/train` multipart CSV file (admin only)
- `GET /health`

## Production checklist
- Use PostgreSQL by setting `DATABASE_URL=postgresql+psycopg://...` and add `psycopg[binary]` to requirements.
- Set a strong unique `SECRET_KEY`; never commit `.env`.
- Use HTTPS, managed database backups, rate limiting, email verification, password reset, audit logs, and monitoring.
- Restrict admin account creation and protect uploaded datasets.
- Add a dataset-specific data card, consent/legal review, subgroup fairness checks, calibration, temporal validation, and model drift monitoring before real-world decisions.
- Keep a human admissions counsellor in the loop. This is decision support, not an official admission decision.

## Deployment
Deploy the Dockerfile to Render, Railway, Fly.io, or another container host. Configure persistent storage mounted at `/srv/data`, environment variables `SECRET_KEY`, `ADMIN_EMAIL`, `DATABASE_URL=sqlite:////srv/data/admission_ps.db`, and `MODEL_PATH=/srv/data/admission_model.joblib`. For multi-instance production, use PostgreSQL and shared durable model storage/object storage.

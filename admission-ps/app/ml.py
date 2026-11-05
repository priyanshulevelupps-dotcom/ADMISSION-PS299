import os, json
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["exam","course","institution","category","state_quota","round","year",
            "score","score_type","gender","domicile_state","seat_type"]
NUMERIC = ["round","year","score"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
MODEL_PATH = os.getenv("MODEL_PATH", "./data/admission_model.joblib")
os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

def train_csv(path: str):
    df = pd.read_csv(path)
    required = set(FEATURES + ["admitted"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    df = df[FEATURES + ["admitted"]].copy()
    df["admitted"] = pd.to_numeric(df["admitted"], errors="coerce")
    df = df[df["admitted"].isin([0,1])]
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are required for a basic model.")
    if df["admitted"].nunique() != 2 or df["admitted"].value_counts().min() < 5:
        raise ValueError("Need at least 5 admitted and 5 not-admitted rows.")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    X, y = df[FEATURES], df["admitted"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, NUMERIC),
                              ("cat", categorical_pipe, CATEGORICAL)])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1500,
                    class_weight="balanced"))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    metrics = {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
               "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
               "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
               "total_rows": int(len(df)), "trained_at": datetime.now(timezone.utc).isoformat(),
               "features": FEATURES, "target": "admitted",
               "validation": "random stratified holdout; use temporal validation before production"}
    joblib.dump({"pipeline": pipe, "metadata": metrics}, MODEL_PATH)
    return metrics

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def predict(features: dict):
    artifact = load_model()
    if artifact:
        frame = pd.DataFrame([{k: features.get(k) for k in FEATURES}])
        p = float(artifact["pipeline"].predict_proba(frame)[0,1])
        return {"model_status":"trained", "probability":round(p,4),
                "screening_score":round(p*100,1), "label":"Historical-model estimate",
                "model_metrics":artifact["metadata"]}
    # Explicit fallback: transparent heuristic score, never presented as an ML probability.
    score = features.get("score")
    score_type = str(features.get("score_type","")).lower()
    if score is None:
        screening = 0.0
    elif "percentile" in score_type:
        screening = max(0.0, min(100.0, float(score)))
    elif "rank" in score_type:
        screening = max(0.0, min(100.0, 100.0 - (float(score)/100000.0*100.0)))
    elif "marks" in score_type:
        screening = max(0.0, min(100.0, float(score)))
    else:
        screening = 0.0
    return {"model_status":"not_trained", "probability":None,
            "screening_score":round(screening,1), "label":"Input-based screening indicator",
            "notice":"No historical ML model is trained. This score is not an admission probability."}

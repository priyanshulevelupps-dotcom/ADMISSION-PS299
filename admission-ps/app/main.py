import os, json, tempfile
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc
from .database import Base, engine, get_db
from .models import User, Prediction
from .security import hash_password, verify_password, create_token, get_current_user, require_admin
from .ml import train_csv, load_model, predict

Base.metadata.create_all(bind=engine)
app = FastAPI(title="ADMISSION PS API", version="1.0.0",
              description="Indian admissions decision-support platform")
STATIC = Path(__file__).parent / "static"

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

class PredictIn(BaseModel):
    exam: str = Field(min_length=2, max_length=40)
    course: str = Field(min_length=2, max_length=100)
    institution: str = Field(min_length=2, max_length=180)
    category: str
    state_quota: str
    round: int = Field(ge=1, le=20)
    year: int = Field(ge=2000, le=2100)
    score: float
    score_type: str
    gender: str = "UNKNOWN"
    domicile_state: str = "UNKNOWN"
    seat_type: str = "UNKNOWN"

@app.get("/health")
def health():
    return {"status":"ok", "service":"ADMISSION PS"}

@app.post("/api/auth/register")
def register(data: RegisterIn, db: Session = Depends(get_db)):
    email = data.email.lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "An account with this email already exists")
    admin_email = os.getenv("ADMIN_EMAIL", "").lower()
    user = User(name=data.name, email=email, password_hash=hash_password(data.password),
                role="admin" if admin_email and email == admin_email else "student")
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token":create_token(user.id), "token_type":"bearer",
            "user":{"id":user.id,"name":user.name,"email":user.email,"role":user.role}}

@app.post("/api/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username.lower()).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    return {"access_token":create_token(user.id), "token_type":"bearer",
            "user":{"id":user.id,"name":user.name,"email":user.email,"role":user.role}}

@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return {"id":user.id,"name":user.name,"email":user.email,"role":user.role}

@app.post("/api/predict")
def make_prediction(data: PredictIn, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    features = data.model_dump()
    result = predict(features)
    record = Prediction(user_id=user.id, exam=data.exam, course=data.course,
        institution=data.institution, score=data.score, score_type=data.score_type,
        category=data.category, state_quota=data.state_quota,
        result_type=result["model_status"], probability=result["probability"],
        screening_score=result["screening_score"], details=json.dumps(result))
    db.add(record); db.commit(); db.refresh(record)
    return {"id":record.id, **result}

@app.get("/api/predictions")
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Prediction).filter(Prediction.user_id==user.id).order_by(
        desc(Prediction.created_at)).limit(100).all()
    return [{"id":r.id,"created_at":r.created_at.isoformat(),"exam":r.exam,
             "course":r.course,"institution":r.institution,"score":r.score,
             "score_type":r.score_type,"screening_score":r.screening_score,
             "probability":r.probability,"model_status":r.result_type} for r in rows]

@app.get("/api/admin/model")
def model_status(admin: User = Depends(require_admin)):
    artifact = load_model()
    return {"trained":bool(artifact), "metadata":artifact["metadata"] if artifact else None}

@app.post("/api/admin/train")
async def train(file: UploadFile = File(...), admin: User = Depends(require_admin)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "CSV must be 25 MB or smaller")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(content); temp_path = tmp.name
        metrics = train_csv(temp_path)
        return {"success":True,"metrics":metrics}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, "Training failed. Check CSV formatting and server logs.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")

app.mount("/static", StaticFiles(directory=STATIC), name="static")

from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="student")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    exam: Mapped[str] = mapped_column(String(40))
    course: Mapped[str] = mapped_column(String(100))
    institution: Mapped[str] = mapped_column(String(180))
    score: Mapped[float] = mapped_column(Float)
    score_type: Mapped[str] = mapped_column(String(40))
    category: Mapped[str] = mapped_column(String(40))
    state_quota: Mapped[str] = mapped_column(String(40))
    result_type: Mapped[str] = mapped_column(String(30))
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    screening_score: Mapped[float] = mapped_column(Float)
    details: Mapped[str] = mapped_column(Text, default="{}")
    user = relationship("User", back_populates="predictions")

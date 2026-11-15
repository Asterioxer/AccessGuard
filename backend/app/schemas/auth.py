from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class UserRegister(BaseModel):
    email: str = Field(..., pattern=r"^[\w\.\+-]+@[\w\.-]+\.\w+$", description="Valid email address")
    password: str = Field(..., min_length=6, description="Password min length 6 chars")
    role: Optional[str] = "contributor" # admin, verifier, contributor, public

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str

class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    reputation_score: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import User
from app.schemas.auth import UserRegister, UserLogin, Token, UserResponse
from app.core.security import verify_password, get_password_hash, create_access_token, get_current_user_payload
from app.core.limiter import limiter
from app.core.logging import logger

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
def register(request: Request, user_in: UserRegister, db: Session = Depends(get_db)):
    """Registers a new user account with role-based access control."""
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )
        
    user = User(
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role or "contributor",
        reputation_score=10
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"Registered new user: {user.email} with role: {user.role}")
    return user

@router.post("/login", response_model=Token)
@limiter.limit("30/minute")
def login(request: Request, login_in: UserLogin, db: Session = Depends(get_db)):
    """Authenticates credentials and issues signed JWT bearer token."""
    user = db.query(User).filter(User.email == login_in.email).first()
    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email, "role": user.role}
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        email=user.email,
        role=user.role
    )

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db)
):
    """Retrieves profile of the currently authenticated user."""
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

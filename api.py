from database.database import get_db
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import bcrypt
import os
from dotenv import load_dotenv
from database.middleware import create_token

load_dotenv()
token_time = int(os.getenv("token_time", 60))

router = APIRouter(prefix="/auth", tags=["Authentication"])

class UserCreate(BaseModel):
    name: str = Field(..., examples=["Azumi Abby"])
    email: str = Field(..., examples=["abby@gmail.com"])
    password: str = Field(..., examples=["abby12345"])
    usertype: str = Field("user", examples=["user"])

class UserLogin(BaseModel):
    email: str = Field(..., examples=["abby@gmail.com"])
    password: str = Field(..., examples=["abby12345"])

@router.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    from database.models import User
    
    # Check if user exists
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), salt)
    
    # Create user
    new_user = User(
        name=user.name,
        email=user.email,
        password=hashed_password,
        usertype=user.usertype
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User created successfully", "user_id": new_user.id}

@router.post("/login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    from database.models import User
    
    try:
        user = db.query(User).filter(User.email == credentials.email).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    
        # Verify password
        stored_password = user.password if isinstance(user.password, bytes) else user.password.encode('utf-8')
        
        if not bcrypt.checkpw(credentials.password.encode('utf-8'), stored_password):
            raise HTTPException(status_code=401, detail="Invalid password")
    
        # Create JWT token
        token = create_token(
            details={
                "name": user.name,
                "email": user.email,
                "usertype": user.usertype,
                "user_id": user.id
            },
            expiry=token_time
        )
    
        return {
            "message": "Login successful",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "usertype": user.usertype
            },
            "token": token
        }
    
    except HTTPException as http_e:
        # Re-raise HTTP exceptions (like 404/401) directly so the frontend gets the right error
        raise http_e
        
    except Exception as e:
        db.rollback()
        # Log the error here if you have a logger
        print(f"Login Error: {str(e)}") 
        raise HTTPException(status_code=500, detail="Internal Server Error")
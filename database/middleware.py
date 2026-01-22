import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

load_dotenv()
bearer = HTTPBearer()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

def create_token(details: dict, expiry: int = 60):
    """Create JWT token."""
    expire = datetime.utcnow() + timedelta(minutes=expiry)
    details.update({"exp": expire})
    
    token = jwt.encode(details, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer)
) -> dict:
    """Verify JWT token."""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        
        # Check if token is expired
        expire = payload.get("exp")
        if expire and datetime.utcnow() > datetime.fromtimestamp(expire):
            raise HTTPException(
                status_code=401,
                detail="Token expired"
            )
        
        return {
            "name": payload.get("name"),
            "email": payload.get("email"),
            "usertype": payload.get("usertype"),
            "user_id": payload.get("user_id")
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication error: {str(e)}"
        )
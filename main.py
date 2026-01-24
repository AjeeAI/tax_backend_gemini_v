from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
# --- ADDED IMPORTS FOR STREAMING ---
from fastapi.responses import StreamingResponse
import json
import time
# -----------------------------------
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import uuid
from langchain.messages import AIMessage
# Import routers and dependencies
from api import router as auth_router
from database.middleware import verify_token
from app import get_tax_assistant

load_dotenv()

app = FastAPI(
    title="Nigeria Tax Assistant API",
    description="API for Nigerian tax information and calculations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

class TaxQuestion(BaseModel):
    question: str
    session_id: Optional[str] = None

class TaxImpactRequest(BaseModel):
    monthly_income: float





# -------------------------------------------------------------
# Conversation session storage methods

import sqlite3
from datetime import datetime

DB_PATH = os.path.abspath("tax_files/conversation_messages.db")
os.makedirs("tax_files", exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


def save_message(thread_id: str, role: str, content: str):
    cursor.execute(
        "INSERT INTO conversation_messages (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content)
    )
    conn.commit()

def fetch_history(thread_id: str):
    cursor.execute(
        "SELECT role, content, timestamp FROM conversation_messages WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,)
    )
    rows = cursor.fetchall()
    return [
        {"role": role, "content": content, "timestamp": timestamp}
        for role, content, timestamp in rows
    ]













# Endpoints

@app.get("/")
def home():
    return {
        "message": "Welcome to Nigeria Tax Assistant API",
        "endpoints": {
            "auth": ["POST /auth/signup", "POST /auth/login"],
            "tax": ["POST /ask", "POST /calculate"],
            "health": ["GET /health"]
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Nigeria Tax Assistant API"
    }

@app.get("/get/user")
def get_user_info(
    user_info: dict = Depends(verify_token)
):
    return {
        "user_id": user_info["user_id"],
        "email": user_info["email"],
        "name": user_info["name"]
    }








# # Tax endpoints
# @app.post("/ask")
# def ask_tax_question(
#     request: TaxQuestion,
#     user_info: dict = Depends(verify_token)
# ):
#     """Ask a tax-related question (requires authentication)."""
#     try:
#         assistant = get_tax_assistant()

#         user_id = user_info["user_id"]

#         session_id = request.session_id or "default"

#         thread_id = f"user_{user_id}_session_{session_id}"



#         # Save human question
#         save_message(thread_id, "human", request.question)

        
#         response = assistant.ask_question(
#             question=request.question,
#             user_id=thread_id
#         )

       

#         # Save AI answer
#         save_message(thread_id, "ai", response["messages"][-1].content)
#         tool_call = {}
#         for msg in response["message"]:
#             tool_call.append(msg.tool_calls[0]["name"])
            
#         return {
#             "success": True,
#             "user_id": user_id,
#             "session_id": session_id,
#             "question": request.question,
#             "answer": response,
#             "tool_call": tool_call
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing question: {str(e)}"
#         )






from fastapi.responses import StreamingResponse
import json

@app.post("/ask")
def ask_tax_question(
    request: TaxQuestion,
    user_info: dict = Depends(verify_token)
):
    try:
        assistant = get_tax_assistant()

        user_id = user_info["user_id"]
        session_id = request.session_id or "default"
        thread_id = f"user_{user_id}_session_{session_id}"

        # Save human question
        save_message(thread_id, "human", request.question)

        def stream_generator():
            full_answer = ""
            tool_calls = []

            # Send metadata first
            yield json.dumps({
                "type": "meta",
                "success": True,
                "user_id": user_id,
                "session_id": session_id,
                "question": request.question
            }) + "\n"

            # Stream answer tokens
            for chunk in assistant.ask_question(
                question=request.question,
                user_id=thread_id
            ):
                full_answer += chunk

                yield json.dumps({
                    "type": "token",
                    "content": chunk
                }) + "\n"

            # Save final AI message
            save_message(thread_id, "ai", full_answer)

            # Send final payload
            yield json.dumps({
                "type": "done",
                "answer": full_answer,
                "tool_call": tool_calls
            }) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )




@app.post("/conversation/new")
def new_conversation(user_info: dict = Depends(verify_token)):
    session_id = str(uuid.uuid4())

    return {
        "success": True,
        "session_id": session_id
    }



@app.get("/conversation/history/{session_id}")
def get_conversation_history(session_id: str, user_info: dict = Depends(verify_token)):

    thread_id = f"user_{user_info['user_id']}_session_{session_id}"
    
    history = fetch_history(thread_id)
    return {
            "success": True, 
            "session_id": session_id, 
            "history": history
        }





# --------------------------------

@app.post("/tax/impact")
def tax_impact(
    request: TaxImpactRequest,
    user_info: dict = Depends(verify_token)
):
    try:
        from engine.tax_engine import calculate_tax_impact

        result = calculate_tax_impact(request.monthly_income)

        return {
            "success": True,
            "user_id": user_info["user_id"],
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
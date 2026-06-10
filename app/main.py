"""
app/main.py — FastAPI Railway AI Assistant
Run: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
import uuid
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_utils import generate_response

app = FastAPI(
    title="Railway AI Assistant",
    description="Intelligent conversational assistant for UK railway queries",
    version="1.0.0",
)

# Static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Server-side session memory
sessions: dict = {}  # session_id -> list of {role, content}



# SCHEMAS

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None          # pass existing session to preserve context
    history: Optional[List[Message]] = []     # fallback client-side history

class ChatResponse(BaseModel):
    response: str
    state: dict
    model: str
    session_id: str


# routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Create or reuse session
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    # Use server-side history (preferred) — falls back to client history if new session
    if sessions[session_id]:
        history = sessions[session_id]
    else:
        history = [m.dict() for m in req.history] if req.history else []

    # Generate response
    result = generate_response(req.message, history)

    # Update server-side memory
    sessions[session_id].append({"role": "user", "content": req.message})
    sessions[session_id].append({"role": "assistant", "content": result["response"]})

    return JSONResponse(content={**result, "session_id": session_id})


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session's conversation history."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "t5-small-railway-finetuned",
        "active_sessions": len(sessions),
    }


@app.get("/api/info")
async def info():
    return {
        "name": "Railway AI Assistant",
        "version": "1.0.0",
        "description": "Fine-tuned T5-small model for UK railway queries",
        "memory": "Server-side session memory (context preserved across turns)",
        "capabilities": [
            "Journey planning",
            "Multi-turn conversation with persistent context",
            "Constraint handling (bike, wheelchair, pet, pram)",
            "Priority routing (cheapest, fastest, direct)",
            "Mid-flow destination/time changes",
            "Disruption awareness",
            "Platform queries",
            "Return journey booking",
        ],
    }
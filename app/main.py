from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict
import os

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from app.services.loader import get_documents_by_role
from app.services.vector_store import load_vector_store

app = FastAPI()

# ✅ Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load HuggingFace model
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ✅ Dummy user-role DB
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
}

# ✅ Auth logic
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = users_db.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": credentials.username, "role": user["role"]}

# ✅ Prefix API endpoints to avoid React collisions
@app.get("/api/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}

@app.get("/api/role-docs")
def list_docs_by_role(user=Depends(authenticate)):
    return get_documents_by_role(user["role"])

@app.get("/api/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest, user=Depends(authenticate)):
    print(f"📨 Query from {user['username']} ({user['role']}): {request.message}")
    try:
        retriever = load_vector_store(role=user["role"])
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa.run(request.message)
        return {
            "user": user["username"],
            "role": user["role"],
            "query": request.message,
            "response": result
        }
    except Exception as e:
        print(f"❌ Error during chat: {e}")
        raise HTTPException(status_code=500, detail="Chat failed. Please try again.")

# ✅ Serve React static files
BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
STATIC_DIR = os.path.join(BUILD_DIR, "static")

if not os.path.exists(STATIC_DIR):
    raise RuntimeError(f"React build not found at: {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_react_index():
    return FileResponse(os.path.join(BUILD_DIR, "index.html"))

@app.get("/{full_path:path}")
def serve_react_router(full_path: str):
    return FileResponse(os.path.join(BUILD_DIR, "index.html"))
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}
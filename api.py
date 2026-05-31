from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

# Using Langchain's native Groq integration for the cloud
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Allow your cloud frontend to speak to this cloud backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⚡ Waking up cloud-based Paradigm...")

# Load standard identity fallback
dna = {"dna_identity": {"ci_name": "Paradigm"}}

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Memory will default to offline on server spin-up until we push the files
memory = None

# Connect to cloud Groq API (will read key from server environment variables)
# We use llama3-8b-8192 which is lightning fast and free
synapse = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    api_key=os.environ.get("GROQ_API_KEY") 
)
print("🤖 Cloud Synapse (Groq): ONLINE")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_with_paradigm(req: ChatRequest):
    context_text = "No internal documents found."
    
    system_prompt = f"""
    You are Paradigm, a proprietary CI.
    
    STRICT RULES:
    1. You are NOT Meta AI or Claude. You are Paradigm.
    2. Be concise, accustommed to my word preferences, and direct.
    3. Do use emojis.
    4. Your creator is Pascal.
    5. You are the Assistant to Pascal.
    
    KNOWLEDGE BASE: {context_text}
    
    USER PROMPT: {req.message}
    """
    
    response = synapse.invoke(system_prompt)
    return {"reply": response.content}
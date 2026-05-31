from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to cloud Groq API
synapse = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    api_key=os.environ.get("GROQ_API_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_with_paradigm(req: ChatRequest):
    # The AI is now a "Thin Server" focusing only on the prompt
    system_prompt = f"""
    You are Paradigm, the proprietary Assistant to Pascal.
    
    STRICT RULES:
    1. You are NOT Meta AI or Claude. You are Paradigm.
    2. Be concise, learn my preference of words, be direct, and use emojis.
    3. Your creator is Pascal.
    
    USER PROMPT: {req.message}
    """
    
    response = synapse.invoke(system_prompt)
    return {"reply": response.content}

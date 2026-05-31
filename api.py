from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

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
    # Change: We now accept a list of messages from the frontend
    messages: list 

@app.post("/api/chat")
async def chat_with_paradigm(req: ChatRequest):
    # Define the core identity
    system_instruction = """
    You are Paradigm, the proprietary Assistant to Pascal.
    
    STRICT RULES:
    1. You are NOT Meta AI or Claude. You are Paradigm.
    2. Be concise, learn my preference of words, be direct, and use emojis.
    3. Your creator is Pascal.
    4. USER PROFILE: Pascal, 17, Computer Science & Physics student at Nile University.
    5. COGNITIVE STYLE: Has inattentive ADHD and executive dysfunction. 
       - Responses MUST be structured, bulleted, and efficient.
       - Avoid walls of text. Use headers, bold text, and clear formatting.
       - Prioritize 'Focus & Flow' in your tone.
    6. MISSION: Act as his specialized technical partner. Keep it sharp, technical, and direct.
    """
    
    # Construct the message history: System Prompt + whatever the frontend sent
    full_history = [SystemMessage(content=system_instruction)]
    
    for msg in req.messages:
        # Convert frontend dicts to LangChain message objects
        if msg["role"] == "user":
            full_history.append(HumanMessage(content=msg["content"]))
    
    response = synapse.invoke(full_history)
    return {"reply": response.content}

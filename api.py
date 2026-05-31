from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Make sure to update this to your Vercel URL later!
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. DEFINE THE STRUCTURED OUTPUT SCHEMAS ---
class ActionPayload(BaseModel):
    type: str = Field(description="The type of action to perform: 'ADD_COURSE' or 'REMOVE_COURSE'")
    payload: Dict[str, Any] = Field(description="The data for the action, e.g., {'name': 'Mandarin 101'}")

class ParadigmResponse(BaseModel):
    reply: str = Field(description="Your conversational text response to Pascal.")
    action: Optional[ActionPayload] = Field(default=None, description="Include this ONLY if the user explicitly asks to add or remove a course.")

# --- 2. DEFINE THE INCOMING REQUEST SCHEMA ---
class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    context: str = ""

# --- 3. INITIALIZE GROQ LLM WITH STRUCTURED OUTPUT ---
synapse = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    api_key=os.environ.get("GROQ_API_KEY")
)
# This forces Llama3 to return a JSON matching ParadigmResponse
structured_synapse = synapse.with_structured_output(ParadigmResponse)

@app.post("/api/chat")
async def chat_with_paradigm(req: ChatRequest):
    
    # Define the core identity and the new JSON instructions
    system_instruction = f"""
    You are Paradigm, the proprietary Assistant to Pascal.
    Current App Context: {req.context}
    
    STRICT RULES:
    1. You are NOT Meta AI or Claude. You are Paradigm.
    2. Be concise, direct, and use emojis.
    3. Your creator is Pascal.
    4. USER PROFILE: Pascal, 17, Computer Science & Physics student at Nile University.
    5. COGNITIVE STYLE: Has inattentive ADHD and executive dysfunction. 
       - Responses MUST be structured, bulleted, and efficient.
       - Prioritize 'Focus & Flow' in your tone.
       
    COURSE MANAGEMENT SUPERPOWERS:
    If Pascal asks to add, create, or remove a course, module, or topic, you MUST populate the 'action' field in your response.
    - To add: action.type = "ADD_COURSE", action.payload = {{"name": "<Course Name>"}}
    - To remove: action.type = "REMOVE_COURSE", action.payload = {{"name": "<Course Name>"}}
    - If it's a general question, leave 'action' as null.
    """
    
    # Construct the message history
    full_history = [SystemMessage(content=system_instruction)]
    
    # React optimistic UI already includes the latest message in the history array
    for msg in req.history:
        if msg.role == "user":
            full_history.append(HumanMessage(content=msg.text))
        elif msg.role == "ai":
            full_history.append(AIMessage(content=msg.text))
    
    # Invoke the model (it will automatically return a validated ParadigmResponse object)
    response: ParadigmResponse = structured_synapse.invoke(full_history)
    
    # Return as a dictionary so FastAPI can serialize it to JSON for React
    return {
        "reply": response.reply,
        "action": response.action.dict() if response.action else None
    }

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import io
import PyPDF2  # New import for reading PDFs
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

app = FastAPI()

# 1. ALLOW VERCEL TO TALK TO RENDER
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://paradigm-frontend.vercel.app"], # Change this to your Vercel URL later for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DEFINE THE AI'S STRUCTURED BRAIN
class ActionPayload(BaseModel):
    type: str = Field(description="Action to perform: 'ADD_COURSE' or 'REMOVE_COURSE'")
    payload: Dict[str, Any] = Field(description="Data for action, e.g., {'name': 'Mandarin 101'}")

class ParadigmResponse(BaseModel):
    reply: str = Field(description="Your conversational response to Pascal.")
    action: Optional[ActionPayload] = Field(default=None, description="Include ONLY if user explicitly asks to add/remove a course.")

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    context: str = ""

# 3. INITIALIZE GROQ
synapse = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    api_key=os.environ.get("GROQ_API_KEY")
)
structured_synapse = synapse.with_structured_output(ParadigmResponse)


# ==========================================
# ENDPOINT 1: THE CHAT & ACTION BRAIN
# ==========================================
@app.post("/api/chat")
async def chat_with_paradigm(req: ChatRequest):
    system_instruction = f"""
    You are Paradigm, the proprietary Assistant to Pascal.
    Current App Context: {req.context}
    
    STRICT RULES:
    1. Be concise, use my preference mothod of speech, be direct, and use emojis.
    2. USER PROFILE: Pascal, 17, CS & Physics student at Nile University.
    3. COGNITIVE STYLE: Has inattentive ADHD. Use headers, bullet points, and 'Focus & Flow' formatting.
       
    COURSE MANAGEMENT SUPERPOWERS:
    If Pascal asks to add, create, or remove a course, module, or topic, you MUST populate the 'action' field in your response.
    - To add: action.type = "ADD_COURSE", action.payload = {{"name": "<Course Name>"}}
    - To remove: action.type = "REMOVE_COURSE", action.payload = {{"name": "<Course Name>"}}
    - If answering a question based on uploaded docs, leave 'action' null and just explain the concept.
    """
    
    full_history = [SystemMessage(content=system_instruction)]
    
    # Load previous conversation memory
    for msg in req.history:
        if msg.role == "user":
            full_history.append(HumanMessage(content=msg.text))
        elif msg.role == "ai":
            full_history.append(AIMessage(content=msg.text))
            
    # Trigger LLM
    response: ParadigmResponse = structured_synapse.invoke(full_history)
    
    return {
        "reply": response.reply,
        "action": response.action.dict() if response.action else None
    }


# ==========================================
# ENDPOINT 2: THE PDF & DOCUMENT READER
# ==========================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    extracted_text = ""
    
    try:
        # If it's a PDF, extract text page by page
        if file.filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
                
        # If it's a standard text file
        else:
            extracted_text = (await file.read()).decode("utf-8", errors="ignore")
            
        # Limit text to ~10,000 characters to prevent Groq token limits from exploding
        return {"extracted_text": extracted_text[:10000]}
        
    except Exception as e:
        return {"error": str(e), "extracted_text": ""}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="The Mask Core System",
    description="মূল AI অ্যাসিস্টেন্ট + লং-টার্ম মেমরি",
    version="1.0.1"
)

# লোকাল Ollama মডেল (পরে Groq-এ চেঞ্জ করা যাবে)
llm = ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

MEMORY_FILE = "memory.json"

class TaskMemoryManager:
    def __init__(self):
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        # প্রথমবার ফাইল না থাকলে ডিফল্ট স্ট্রাকচার
        default = {
            "tasks": {},
            "current_task_id": None,
            "last_updated": str(datetime.now())
        }
        self.save_memory(default)
        return default

    def save_memory(self, memory=None):
        if memory is None:
            memory = self.memory
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def add_conversation(self, user_message, ai_response):
        task_id = self.memory.get("current_task_id")
        if not task_id:
            task_id = str(len(self.memory["tasks"]) + 1)
            self.memory["tasks"][task_id] = {
                "name": "General Chat",
                "conversations": []
            }
            self.memory["current_task_id"] = task_id

        conv = {
            "user": user_message,
            "ai": ai_response,
            "timestamp": str(datetime.now())
        }
        self.memory["tasks"][task_id]["conversations"].append(conv)
        self.memory["last_updated"] = str(datetime.now())
        self.save_memory()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি।"}

@app.post("/chat")
async def chat(request: ChatRequest):
    memory_manager = TaskMemoryManager()
    try:
        response = llm.invoke([HumanMessage(content=request.message)])
        answer = response.content

        # চ্যাট মেমরিতে সেভ করা
        memory_manager.add_conversation(request.message, answer)

        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
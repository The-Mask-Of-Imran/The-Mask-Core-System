from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import sqlite3
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from tenacity import retry, wait_exponential, stop_after_attempt
import traceback

app = FastAPI(
    title="The Mask Core System",
    description="মূল AI অ্যাসিস্টেন্ট + লং-টার্ম মেমরি",
    version="1.0.3"
)

# Ollama লোড করার সময় ৩ বার ট্রাই করবে যদি কানেকশন ফেল করে
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def get_llm():
    return ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

# Ollama লোড করা
try:
    llm = get_llm()
    print("Ollama সফলভাবে লোড হয়েছে!")
except Exception as e:
    print(f"Ollama লোড করতে সমস্যা: {str(e)}")
    traceback.print_exc()
    raise

MEMORY_FILE = "memory.json"
DB_PATH = "memory.db"

class TaskMemoryManager:
    def __init__(self):
        self.conn = self.connect_db()
        self.memory = self.load_memory()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def connect_db(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS tasks 
                          (id TEXT PRIMARY KEY, name TEXT, conversations TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata 
                          (key TEXT PRIMARY KEY, value TEXT)''')
        conn.commit()
        return conn

    def load_memory(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        rows = cursor.fetchall()
        tasks = {}
        for row in rows:
            try:
                tasks[row[0]] = {"name": row[1], "conversations": json.loads(row[2])}
            except json.JSONDecodeError:
                tasks[row[0]] = {"name": row[1], "conversations": []}

        # current_task_id নেওয়া
        cursor.execute("SELECT value FROM metadata WHERE key = 'current_task_id'")
        current_id_row = cursor.fetchone()
        current_id = current_id_row[0] if current_id_row else None

        # last_updated নেওয়া
        cursor.execute("SELECT value FROM metadata WHERE key = 'last_updated'")
        last_updated_row = cursor.fetchone()
        last_updated = last_updated_row[0] if last_updated_row else str(datetime.now())

        memory = {
            "tasks": tasks,
            "current_task_id": current_id,
            "last_updated": last_updated
        }

        # যদি ডাটাবেস খালি থাকে তাহলে JSON থেকে লোড
        if not tasks and os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    memory = json.load(f)
                self.save_memory(memory)
            except Exception as e:
                print(f"JSON লোড এরর: {e}")

        return memory

    def save_memory(self, memory=None):
        if memory is None:
            memory = self.memory

        cursor = self.conn.cursor()
        for task_id, task in memory["tasks"].items():
            conv_json = json.dumps(task["conversations"], ensure_ascii=False)
            cursor.execute("""
                INSERT OR REPLACE INTO tasks (id, name, conversations) 
                VALUES (?, ?, ?)
            """, (task_id, task["name"], conv_json))

        if memory.get("current_task_id"):
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value) 
                VALUES ('current_task_id', ?)
            """, (memory["current_task_id"],))

        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value) 
            VALUES ('last_updated', ?)
        """, (memory["last_updated"],))

        self.conn.commit()

        # JSON ব্যাকআপ
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def add_conversation(self, user_message, ai_response):
        task_id = self.memory.get("current_task_id")
        if not task_id:
            task_id = str(len(self.memory["tasks"]) + 1)
            self.memory["tasks"][task_id] = {"name": "General Chat", "conversations": []}
            self.memory["current_task_id"] = task_id

        conv = {
            "user": user_message,
            "ai": ai_response,
            "timestamp": str(datetime.now())
        }
        self.memory["tasks"][task_id]["conversations"].append(conv)
        self.memory["last_updated"] = str(datetime.now())
        self.save_memory()

    def get_context(self):
        task_id = self.memory.get("current_task_id")
        if task_id and task_id in self.memory["tasks"]:
            convs = self.memory["tasks"][task_id]["conversations"][-5:]  # শেষ ৫টা চ্যাট
            context = "\n".join([f"User: {c['user']}\nAI: {c['ai']}" for c in convs])
            return context
        return ""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি। 😊"}

@app.post("/chat")
async def chat(request: ChatRequest):
    memory_manager = TaskMemoryManager()
    try:
        context = memory_manager.get_context()
        full_prompt = f"পূর্বের কথোপকথন (কনটেক্সট):\n{context}\n\nবর্তমান প্রশ্ন: {request.message}"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        answer = response.content
        memory_manager.add_conversation(request.message, answer)
        return {"response": answer}
    except Exception as e:
        print(f"চ্যাট এরর বিস্তারিত: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"সার্ভারে সমস্যা: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"সার্ভার চালু হচ্ছে... http://127.0.0.1:{port} খুলুন")
    uvicorn.run(app, host="0.0.0.0", port=port)
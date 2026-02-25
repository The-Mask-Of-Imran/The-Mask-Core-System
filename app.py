from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import sqlite3
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from tenacity import retry, wait_exponential, stop_after_attempt  # যোগ করো (requirements.txt-এ tenacity যোগ)

app = FastAPI(
    title="The Mask Core System",
    description="মূল AI অ্যাসিস্টেন্ট + লং-টার্ম মেমরি",
    version="1.0.1"
)

# লোকাল Ollama মডেল
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))  # Connection refused fix
def get_llm():
    return ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

llm = get_llm()

MEMORY_FILE = "memory.json"  # Backup
DB_PATH = "memory.db"  # Primary SQLite

class TaskMemoryManager:
    def __init__(self):
        self.conn = self.connect_db()
        self.memory = self.load_memory()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def connect_db(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, name TEXT, conversations TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)''')
        conn.commit()
        return conn

    def load_memory(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        tasks = {row[0]: {"name": row[1], "conversations": json.loads(row[2])} for row in cursor.fetchall()}
        cursor.execute("SELECT value FROM metadata WHERE key = 'current_task_id'")
        current_id = cursor.fetchone()[0] if cursor.fetchone() else None
        cursor.execute("SELECT value FROM metadata WHERE key = 'last_updated'")
        last_updated = cursor.fetchone()[0] if cursor.fetchone() else str(datetime.now())
        memory = {"tasks": tasks, "current_task_id": current_id, "last_updated": last_updated}
        # JSON backup থেকে sync যদি DB খালি
        if not tasks and os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                memory = json.load(f)
            self.save_memory(memory)
        return memory

    def save_memory(self, memory=None):
        if memory is None:
            memory = self.memory
        cursor = self.conn.cursor()
        for task_id, task in memory["tasks"].items():
            conv_json = json.dumps(task["conversations"])
            cursor.execute("INSERT OR REPLACE INTO tasks (id, name, conversations) VALUES (?, ?, ?)", (task_id, task["name"], conv_json))
        if memory["current_task_id"]:
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('current_task_id', ?)", (memory["current_task_id"],))
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_updated', ?)", (memory["last_updated"],))
        self.conn.commit()
        # JSON backup
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    # add_conversation unchanged

# অন্য কোড unchanged
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import importlib.util
import sys
import requests  # GitHub থেকে প্লাগিন ডাউনলোডের জন্য

app = FastAPI(
    title="The Mask Automation - Core System",
    description="কোর সিস্টেম যা সব প্লাগিন লোড করে এবং কমান্ড রান করে",
    version="1.0.1"
)

# অফলাইন LLM (Ollama)
llm = ChatOllama(model="qwen2.5-coder:7b")

# প্লাগিন লিস্ট (তোমার ১৫টা প্লাগিনের নাম)
PLUGIN_REPOS = [
    "The-Mask-Automation-API-Switch-Plugin",
    "The-Mask-Automation-Feature-Analysis-Plugin",
    "The-Mask-Automation-Roadmap-Generator-Plugin",
    "The-Mask-Automation-Self-Upgrade-Plugin",
    "The-Mask-Automation-PC-Access-Plugin",
    "The-Mask-Automation-Social-Upload-Plugin",
    "The-Mask-Automation-Captcha-Handler-Plugin",
    "The-Mask-Automation-Backup-Sync-Plugin",
    "The-Mask-Automation-Hybrid-Learning-Plugin",
    "The-Mask-Automation-Personal-Memory-Plugin",
    "The-Mask-Automation-TTS-Plugin",
    "The-Mask-Automation-Image-Analysis-Plugin",
    "The-Mask-Automation-Image-Generation-Plugin",
    "The-Mask-Automation-Video-Analysis-Plugin",
    "The-Mask-Automation-Video-Generation-Plugin",
]

loaded_plugins = {}

def load_plugin_from_github(plugin_name):
    try:
        # GitHub থেকে raw plugin.py ডাউনলোড
        url = f"https://raw.githubusercontent.com/The-Mask-Of-Imran/{plugin_name}/main/plugin.py"
        response = requests.get(url)
        if response.status_code == 200:
            code = response.text
            # কোড এক্সিকিউট করে মডিউল তৈরি
            module = type(plugin_name, (), {})
            exec(code, module.__dict__)
            print(f"Loaded plugin from GitHub: {plugin_name}")
            return module
        else:
            print(f"Failed to load {plugin_name}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error loading {plugin_name}: {e}")
        return None

# স্টার্টআপে সব প্লাগিন লোড করো
for p in PLUGIN_REPOS:
    loaded_plugins[p] = load_plugin_from_github(p)

class CommandRequest(BaseModel):
    command: str

@app.get("/")
def root():
    return {
        "message": "The Mask Core System is running!",
        "status": "active",
        "loaded_plugins_count": len([p for p in loaded_plugins.values() if p is not None]),
        "version": "1.0.1"
    }

@app.post("/command")
async def run_command(request: CommandRequest):
    try:
        # এখানে পরে প্লাগিন দিয়ে কমান্ড প্রসেস করবে
        response = llm.invoke([HumanMessage(content=request.command)])
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upgrade")
async def self_upgrade():
    # পরে git pull + restart লজিক যোগ করব
    return {"status": "Upgrade initiated (placeholder)"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
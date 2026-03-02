import os
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

class ModelRouter:
    def __init__(self):
        self.ollama = ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434")
        
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.groq = Groq(api_key=self.groq_key) if self.groq_key else None
        
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai = OpenAI(api_key=self.openai_key) if self.openai_key else None
        
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini = genai.GenerativeModel('gemini-1.5-flash')  # আপডেটেড মডেল
        else:
            self.gemini = None
        
        self.api_calls = {"groq": 0, "openai": 0, "gemini": 0}
        self.max_calls = {"groq": 100, "openai": 60, "gemini": 15}

    def select_model(self, use_online=False):
        if not use_online:
            return "ollama"
        if self.groq and self.api_calls["groq"] < self.max_calls["groq"]:
            return "groq"
        if self.openai and self.api_calls["openai"] < self.max_calls["openai"]:
            return "openai"
        if self.gemini and self.api_calls["gemini"] < self.max_calls["gemini"]:
            return "gemini"
        return "ollama"

    async def invoke(self, messages, use_online=False):
        # প্রথমে Ollama চেষ্টা
        try:
            ollama_response = self.ollama.invoke(messages)
            print("[MODEL USED] ollama (local)")
            return ollama_response.content
        except Exception as ollama_error:
            print(f"[OLLAMA FAILED] {str(ollama_error)}")
            
            # ফেল করলে অনলাইন
            model = self.select_model(use_online=True)
            print(f"[MODEL USED] {model} (fallback)")
            
            content = messages[0].content if messages else ""
            
            if model == "groq" and self.groq:
                self.api_calls["groq"] += 1
                response = self.groq.chat.completions.create(
                    messages=[{"role": "user", "content": content}],
                    model="llama-3.3-70b-versatile"
                )
                return response.choices[0].message.content
            
            elif model == "openai" and self.openai:
                self.api_calls["openai"] += 1
                response = self.openai.chat.completions.create(
                    messages=[{"role": "user", "content": content}],
                    model="gpt-3.5-turbo"
                )
                return response.choices[0].message.content
            
            elif model == "gemini" and self.gemini:
                self.api_calls["gemini"] += 1
                response = self.gemini.generate_content(content)
                return response.text
            
            else:
                raise Exception("No available model")

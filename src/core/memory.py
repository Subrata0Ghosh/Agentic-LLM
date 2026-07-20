import json
import os

MEMORY_FILE = "data/chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=4)

def add_to_memory(session_id, role, content):
    memory = load_memory()
    if session_id not in memory:
        memory[session_id] = []
        
    memory[session_id].append({"role": role, "content": content})
    
    # Keep only the last 10 interactions to avoid context overflow
    if len(memory[session_id]) > 10:
        memory[session_id] = memory[session_id][-10:]
        
    save_memory(memory)

def get_context_string(session_id):
    memory = load_memory()
    if session_id not in memory:
        return ""
        
    context = ""
    for msg in memory[session_id]:
        if msg['role'] == 'user':
            context += f"User: {msg['content']}\n"
        else:
            context += f"AI: {msg['content']}\n"
            
    return context

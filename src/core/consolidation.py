import sys
import os
import logging

sys.path.append(os.path.dirname(__file__))
from vector_memory import VectorMemory
from generate import generate_text_api
from online_train import online_finetune
from brain_state import BrainState

logger = logging.getLogger("Consolidation")

def run_memory_consolidation(session_id, max_train_iters=20):
    """
    Simulates the biological consolidation process (Sleep/Dream cycle):
    1. Reads raw episodic facts from ChromaDB (Hippocampus).
    2. Synthesizes/generalizes these memories using the generator model.
    3. Fine-tunes the custom PyTorch Transformer model (Neocortex) on the synthesized knowledge.
    4. Resets the brain's fatigue and replenishes neuromodulators.
    """
    print(f"[{session_id}] Beginning sleep/dream cycle: Memory Consolidation...")
    
    # 1. Retrieve episodic memories
    v_mem = VectorMemory()
    try:
        results = v_mem.collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"Error reading from vector memory: {e}")
        return "Failed to access episodic memory."
        
    documents = results.get("documents", [])
    if not documents:
        # If no memories, we just reset the brain fatigue
        bs = BrainState()
        bs.sleep_reset(session_id)
        return "No episodic memories found to consolidate. Brain fatigue has been refreshed."

    # Filter out empty or duplicate documents
    unique_docs = list(set([doc.strip() for doc in documents if doc.strip()]))
    if not unique_docs:
        bs = BrainState()
        bs.sleep_reset(session_id)
        return "No unique memories found. Brain fatigue has been refreshed."
        
    print(f"Consolidating {len(unique_docs)} episodic memory segments...")

    # 2. Synthesize/Dream: Extract generalized concepts from the raw documents
    # To save compute, we consolidate up to 5 documents at a time
    synthesized_facts = []
    
    for i in range(0, min(10, len(unique_docs)), 2):
        chunk = unique_docs[i:i+2]
        combined_text = "\n\n".join(chunk)
        
        system_content = (
            "You are a cognitive processor. Your task is to act as a 'dream state' that takes raw, "
            "detailed episodic experiences and extracts 2-3 core, generalized, clean factual sentences. "
            "Remove website details, URLs, and conversational clutter. Write only the factual sentences."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Episodic memory text:\n{combined_text}\n\nGeneralized core facts:"}
        ]
        
        try:
            fact_summary = generate_text_api(messages, max_new_tokens=150)
            if fact_summary and not fact_summary.startswith("I'm sorry"):
                synthesized_facts.append(fact_summary.strip())
        except Exception as e:
            print(f"Error during memory synthesis: {e}")

    # Join synthesized facts
    consolidated_corpus = "\n".join(synthesized_facts)
    print(f"Synthesized knowledge:\n{consolidated_corpus}")

    # Write to the persistent corpus file (Neocortical Knowledge Base)
    if consolidated_corpus.strip():
        os.makedirs("data", exist_ok=True)
        with open("data/ai_corpus_cleaned.txt", "a", encoding="utf-8") as f:
            f.write("\n\n# Consolidated Memory Replay:\n" + consolidated_corpus)
            
        # 3. Neocortical Fine-Tuning: Train the Custom PyTorch Transformer model (CustomLLM) on this new data
        print("Transferring memories to neocortex (custom model weights)...")
        try:
            online_finetune(consolidated_corpus, max_iters=max_train_iters)
            fine_tune_success = True
        except Exception as e:
            print(f"Error during neocortical training: {e}")
            fine_tune_success = False
    else:
        fine_tune_success = False
        print("No new facts synthesized during dream phase.")

    # 4. Reset biological state
    bs = BrainState()
    bs.sleep_reset(session_id)
    
    summary = f"Sleep consolidation complete for session {session_id}.\n"
    summary += f"- Consolidated {len(unique_docs)} episodic experiences.\n"
    if fine_tune_success:
        summary += f"- Successfully fine-tuned custom Transformer weights (data/model.pt).\n"
    else:
        summary += f"- Brain refreshed, but no new structural weights updated.\n"
    summary += "- Fatigue reset to 0.0, Attention restored to 0.9."
    
    return summary

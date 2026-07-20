import re
import sys
import os

sys.path.append(os.path.dirname(__file__))
from generate import generate_text_api

class ImaginationEngine:
    def __init__(self):
        pass

    def run_cognitive_loop(self, session_id, user_message, retrieved_context, history_messages):
        """
        Runs the cognitive simulation loop. Emulates the Default Mode Network
        by prompting the model to generate its reasoning (internal monologue)
        before producing the final output, all parsed cleanly.
        """
        from brain_state import BrainState
        bs = BrainState()
        params = bs.get_generation_params(session_id)

        # Build a biological system prompt
        system_content = (
            "You are a friendly and helpful AI assistant running on a biologically-inspired cognitive architecture. "
            "To answer the user's request, you must first run an internal simulation (imagination / monologue) "
            "where you analyze the input, evaluate facts, and identify any gaps in your knowledge. "
            "You must format your response EXACTLY as follows:\n\n"
            "<thought>\n"
            "Here, write your step-by-step internal reasoning, hypotheses, and knowledge-gap checks. "
            "Actively question your assumptions.\n"
            "</thought>\n"
            "<answer>\n"
            "Your warm, natural, and plain English response to the user.\n"
            "</answer>\n\n"
            "Keep the <thought> block focused, honest, and analytical, and the <answer> block friendly and concise."
        )

        if retrieved_context:
            system_content += f"\n\nContext retrieved from episodic memory (Hippocampus):\n{retrieved_context}"

        # Build messages list
        messages = [{"role": "system", "content": system_content}]
        
        # Add history messages
        for msg in history_messages:
            role = "assistant" if msg["role"] == "ai" else "user"
            messages.append({"role": role, "content": msg["content"]})
            
        # Add current user query
        messages.append({"role": "user", "content": user_message})

        # Generate using the dynamic brain parameters
        max_tokens = 350  # Slightly longer to accommodate thoughts
        
        raw_output = generate_text_api(
            messages,
            max_new_tokens=max_tokens,
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"]
        )

        # Parse the output
        thought_process = ""
        final_answer = ""
        
        # Regex to extract thoughts and answer
        thought_match = re.search(r'<thought>(.*?)</thought>', raw_output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)

        if thought_match:
            thought_process = thought_match.group(1).strip()
        else:
            # Fallback if model forgot tags but wrote thought-like content
            if "</thought>" in raw_output:
                thought_process = raw_output.split("</thought>")[0].replace("<thought>", "").strip()
            else:
                thought_process = "Analyzing query based on current memories and active attention."

        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            # Fallback: if no answer tag, clean up thought tags and use the remaining text
            if "</thought>" in raw_output:
                final_answer = raw_output.split("</thought>")[-1].replace("<answer>", "").replace("</answer>", "").strip()
            else:
                final_answer = raw_output

        # Clean up any leftover XML tags
        final_answer = re.sub(r'</?thought>', '', final_answer, flags=re.IGNORECASE)
        final_answer = re.sub(r'</?answer>', '', final_answer, flags=re.IGNORECASE).strip()
        
        # Check if the final answer contains suggestions
        suggestions = []
        if "SUGGESTIONS:" in final_answer:
            parts = final_answer.split("SUGGESTIONS:")
            final_answer = parts[0].strip()
            suggestions_text = parts[1].strip()
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    clean_sugg = line[2:].replace('**', '').strip()
                    if clean_sugg:
                        suggestions.append(clean_sugg)

        return {
            "monologue": thought_process,
            "response": final_answer,
            "suggestions": suggestions,
            "raw_length": len(raw_output.split())
        }

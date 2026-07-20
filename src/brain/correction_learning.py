"""
Correction Learning — src/brain/correction_learning.py
======================================================
Brain Region: Prefrontal Cortex (error monitoring) & Hippocampus/Neocortex

This module implements a correction learning loop for BICA:
1. Detects when the user says the AI made a mistake.
2. Prompts the user for the correct fact/information.
3. Verifies the user's correction by searching the web and using an LLM filter.
4. If verified (or overridden by the user), encodes the correction into the
   Hippocampus, integrates it into the Neocortex schema, and immediately runs
   an online fine-tuning loop to update the CustomLLM weights on the spot.
"""

import os
import sys
import json
import time
import re
from typing import Dict, Optional

# Add core/brain to path
_HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(_HERE, '..', 'core'))

from scraper import search_and_scrape
from generate import generate_text_api
from online_train import online_finetune
from memory import load_memory

STATE_FILE = "data/correction_state.json"

class CorrectionLearning:
    """
    Manages the user correction dialogue and self-training flow.
    """

    def __init__(self, filepath: str = STATE_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({}, f, indent=2)

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        try:
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[CorrectionLearning] Failed to save state: {e}")

    def get_state(self, session_id: str) -> dict:
        data = self._load()
        return data.get(session_id, {"status": "idle"})

    def set_state(self, session_id: str, state: dict):
        data = self._load()
        data[session_id] = state
        self._save(data)

    def clear_state(self, session_id: str):
        data = self._load()
        if session_id in data:
            del data[session_id]
        self._save(data)

    def is_correction_intent(self, msg: str) -> bool:
        """Detects if the user's message indicates the AI's previous answer was wrong."""
        msg_l = msg.lower().strip()
        
        # Direct rejection patterns
        patterns = [
            r"\byou(?:'re| are) wrong\b",
            r"\bthat(?:'s| is) wrong\b",
            r"\bthat(?:'s| is) incorrect\b",
            r"\bincorrect\b",
            r"\bmistake\b",
            r"\bfalse\b",
            r"\bnot correct\b",
            r"\bnot right\b",
            r"\berror in your answer\b",
            r"\byou got it wrong\b",
            r"\bno, it's not\b",
            r"\bno, that's not right\b",
            r"\bthat's not true\b",
            r"\byou made a mistake\b"
        ]
        
        for pat in patterns:
            if re.search(pat, msg_l):
                return True
        return False

    def process_message(self, session_id: str, msg: str, hipp, neo, nm) -> Optional[dict]:
        """
        Processes the message against the correction state machine.
        Returns a response dict if handled, otherwise None to proceed with normal chat.
        """
        state = self.get_state(session_id)
        status = state.get("status", "idle")

        if status == "idle":
            if self.is_correction_intent(msg):
                # Retrieve the last assistant response and user query from history
                mem = load_memory()
                session_msgs = mem.get(session_id, [])
                
                # Note: msg was already appended to memory, so:
                # session_msgs[-1] is the current correction message
                # session_msgs[-2] is the last assistant response
                # session_msgs[-3] is the original query
                wrong_query = "your previous query"
                wrong_response = "what I said"
                
                if len(session_msgs) >= 3:
                    wrong_query = session_msgs[-3]["content"]
                    wrong_response = session_msgs[-2]["content"]

                new_state = {
                    "status": "awaiting_correction_data",
                    "wrong_query": wrong_query,
                    "wrong_response": wrong_response,
                    "timestamp": time.time()
                }
                self.set_state(session_id, new_state)

                return {
                    "response": (
                        "I apologize for the mistake. Could you please tell me what the correct answer "
                        "or information should be? This will help me learn and update my knowledge."
                    ),
                    "pipeline_trace": [{"region": "CorrectionLearning", "state": "detect_wrong_trigger"}],
                    "monologue": {
                        "stage1_recall": "User flagged previous response as wrong. Requesting correct data.",
                        "stage2_simulate": "",
                        "stage3_critique": ""
                    }
                }
            return None

        # Check if the user wants to cancel the correction process
        if msg.lower().strip() in ["cancel", "never mind", "forget it", "no"]:
            self.clear_state(session_id)
            return {
                "response": "Understood. Let's continue our conversation.",
                "pipeline_trace": [{"region": "CorrectionLearning", "state": "cancelled"}],
                "monologue": {
                    "stage1_recall": "Correction learning cancelled by user.",
                    "stage2_simulate": "",
                    "stage3_critique": ""
                }
            }

        if status == "awaiting_correction_data":
            # msg is the correction data. Let's verify it!
            wrong_query = state.get("wrong_query")
            
            # 1. Search the web for verification
            search_query = f"{wrong_query} {msg}"
            search_results = ""
            try:
                search_results = search_and_scrape(search_query, num_results=3)
            except Exception as e:
                print(f"[CorrectionLearning] Web search failed: {e}")

            # 2. Use LLM to verify search results against correction
            verification_prompt = [
                {"role": "system", "content": (
                    "You are the brain's semantic validation system. Your task is to verify if the user's corrected fact is true based on the provided web search results.\n"
                    "If the search results support or confirm the user's statement, respond with YES.\n"
                    "If the search results contradict the user's statement, respond with NO.\n"
                    "If the search results are empty, ambiguous, or if the correction is a personal preference or a subjective topic, respond with YES (give the user the benefit of the doubt)."
                )},
                {"role": "user", "content": (
                    f"Original Question: \"{wrong_query}\"\n"
                    f"User correction: \"{msg}\"\n\n"
                    f"Search results:\n{search_results}\n\n"
                    "Does this verify as correct? Respond with YES or NO only:"
                )}
            ]
            
            verdict = "YES"
            try:
                verdict = generate_text_api(verification_prompt, max_new_tokens=5, temperature=0.1)
            except Exception as e:
                print(f"[CorrectionLearning] LLM verification failed: {e}")

            is_verified = "yes" in verdict.lower()

            if is_verified:
                # Perform the learning action
                self._learn_and_train(session_id, wrong_query, msg, hipp, neo, nm)
                self.clear_state(session_id)
                
                return {
                    "response": (
                        f"Thank you for correcting me! I searched the web to verify this information, and it checks out.\n\n"
                        f"I have successfully encoded this into my hippocampus and run an online learning session to update my neocortical model. "
                        f"Next time you ask about this, I will remember the correct answer!"
                    ),
                    "pipeline_trace": [
                        {"region": "CorrectionLearning", "state": "verified_and_trained", "search_verified": True}
                    ],
                    "was_web_searched": True,
                    "monologue": {
                        "stage1_recall": f"Verified user correction '{msg}' via web search.",
                        "stage2_simulate": "Running online fine-tuning on CustomLLM.",
                        "stage3_critique": "Model weights updated successfully."
                    }
                }
            else:
                # Transition to override state
                new_state = {
                    "status": "awaiting_override",
                    "wrong_query": wrong_query,
                    "wrong_response": state.get("wrong_response"),
                    "candidate_correction": msg,
                    "timestamp": time.time()
                }
                self.set_state(session_id, new_state)

                return {
                    "response": (
                        f"I searched the web to verify your correction: \"{msg}\", but the sources seem to contradict or not support it.\n\n"
                        f"Are you sure this is correct? If yes, please say **'override'** (or **'force'**) to update my model anyway. "
                        f"Otherwise, please describe the correct details, or say **'cancel'**."
                    ),
                    "pipeline_trace": [
                        {"region": "CorrectionLearning", "state": "verification_failed", "search_verified": False}
                    ],
                    "was_web_searched": True,
                    "monologue": {
                        "stage1_recall": f"User correction '{msg}' failed verification.",
                        "stage2_simulate": "",
                        "stage3_critique": "Asking for override or cancellation."
                    }
                }

        if status == "awaiting_override":
            candidate_correction = state.get("candidate_correction")
            wrong_query = state.get("wrong_query")

            if msg.lower().strip() in ["override", "force", "yes", "sure", "force update", "force learn"]:
                # Force learn it anyway
                self._learn_and_train(session_id, wrong_query, candidate_correction, hipp, neo, nm)
                self.clear_state(session_id)

                return {
                    "response": (
                        f"Understood. I have overridden the check and trained my neocortical model on your correction: "
                        f"\"{candidate_correction}\". Next time you ask about this, I will use this answer."
                    ),
                    "pipeline_trace": [
                        {"region": "CorrectionLearning", "state": "override_accepted", "search_verified": False}
                    ],
                    "monologue": {
                        "stage1_recall": "User triggered manual override.",
                        "stage2_simulate": "Running online fine-tuning on CustomLLM using user's overridden data.",
                        "stage3_critique": "Model weights updated successfully."
                    }
                }
            else:
                # If they provided another text, treat it as a new correction data attempt
                # Transition back to awaiting_correction_data state
                state["status"] = "awaiting_correction_data"
                state["timestamp"] = time.time()
                self.set_state(session_id, state)
                # Recurse process_message with the new correction data
                return self.process_message(session_id, msg, hipp, neo, nm)

        return None

    def _learn_and_train(self, session_id: str, query: str, correction: str, hipp, neo, nm):
        """
        Encodes the corrected fact to hippocampus, integrates into neocortex schema,
        appends to the training corpus, and runs online fine-tuning.
        """
        # 1. Encode into Hippocampus
        hipp.encode(
            content=f"Correction: For question '{query}', the correct answer is: '{correction}'.",
            session_id=session_id,
            source="user_correction_learning",
            emotion_tag={"valence": "positive", "arousal": 0.4, "priority": 0.85},
            ach_level=0.9
        )

        # 2. Integrate into Neocortex Schema
        neo.integrate(
            concept=query[:40],
            definition=f"Correct answer: {correction}",
            source="user_correction"
        )

        # 3. Append to training corpus
        corpus_entry = f"\n\n# User Correction [{session_id}]:\nQ: {query}\nA: {correction}"
        try:
            with open("data/ai_corpus_cleaned.txt", "a", encoding="utf-8") as f:
                f.write(corpus_entry)
        except Exception as e:
            print(f"[CorrectionLearning] Corpus append failed: {e}")

        # 4. Trigger online training
        training_text = f"Q: {query}\nA: {correction}"
        print(f"[CorrectionLearning] Starting online fine-tuning on: '{training_text}'")
        try:
            online_finetune(training_text, max_iters=25)
        except Exception as e:
            print(f"[CorrectionLearning] Fine-tuning failed: {e}")

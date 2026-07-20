"""
Planning PFC — src/brain/planning_pfc.py
===========================================
Brain Region: Dorsolateral Prefrontal Cortex (dlPFC)

The dlPFC is critical for goal directed behavior, strategic planning,
and monitoring performance across sequential actions.

Key mechanisms:
  1. Plan formulation: Breaks a large complex goal into sequential sub-goals.
  2. Plan execution: Tracks progress of current sub-steps across turns.
  3. Action monitoring: Feeds next step targets into the DMN.
"""

import os
import json
import time
from typing import List, Dict, Optional

PLAN_FILE = "data/active_plans.json"

class PlanningPFC:
    """
    Goal planning and sequential step tracker representing the Dorsolateral PFC.
    """

    def __init__(self, filepath: str = PLAN_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def is_complex_goal(self, query: str) -> bool:
        """Determines if the query demands a multi-step sequence."""
        keywords = ["then", "first", "next", "after that", "finally", "step by step",
                    "plan", "stages", "roadmap", "sequence", "procedural", "how to build", "workflow"]
        query_l = query.lower()
        # Checks if multiple sequential markers or verbs are present
        has_seq_word = any(f" {kw} " in f" {query_l} " for kw in keywords)
        has_punctuation_seq = any(mark in query for mark in [";", "->", "=>"])
        return (has_seq_word or has_punctuation_seq) and len(query.split()) > 10

    def create_plan(self, session_id: str, query: str) -> dict:
        """
        Deconstructs the query into a sequence of steps.
        For simplicity, we split by sequential keywords or generate a baseline 3-stage plan structure.
        """
        steps = []
        # Split by typical punctuation/connectors
        delimiters = [r"\bthen\b", r"\bnext\b", r"\bafter that\b", r"\bfinally\b", r"\n", r";"]
        pattern = "|".join(delimiters)
        import re
        raw_steps = re.split(pattern, query, flags=re.IGNORECASE)
        
        for idx, rs in enumerate(raw_steps):
            clean = re.sub(r"^(first|then|next|finally|step \d+:?)\s*", "", rs.strip(), flags=re.IGNORECASE)
            if clean and len(clean.split()) > 1:
                steps.append({
                    "step_id": idx + 1,
                    "goal": clean,
                    "status": "pending",
                    "timestamp": None
                })
        
        # Fallback if no clear sequence split: generate default cognitive stages
        if len(steps) < 2:
            steps = [
                {"step_id": 1, "goal": f"Analyze and prepare: {query[:80]}", "status": "pending", "timestamp": None},
                {"step_id": 2, "goal": "Execute detail operations and synthesis", "status": "pending", "timestamp": None},
                {"step_id": 3, "goal": "Final verification and summary compilation", "status": "pending", "timestamp": None}
            ]

        plan = {
            "session_id": session_id,
            "original_query": query,
            "current_step_idx": 0,
            "steps": steps,
            "created_ts": time.time(),
            "status": "running"
        }

        data = self._load()
        data[session_id] = plan
        self._save(data)
        return plan

    def get_active_plan(self, session_id: str) -> Optional[dict]:
        data = self._load()
        return data.get(session_id)

    def advance_step(self, session_id: str) -> Optional[dict]:
        """Progresses the active plan to the next step."""
        data = self._load()
        plan = data.get(session_id)
        if not plan or plan["status"] != "running":
            return None

        curr = plan["current_step_idx"]
        # Guard: don't advance past the last valid index
        if curr >= len(plan["steps"]):
            plan["status"] = "completed"
            data[session_id] = plan
            self._save(data)
            return plan

        plan["steps"][curr]["status"] = "completed"
        plan["steps"][curr]["timestamp"] = time.time()

        next_idx = curr + 1
        if next_idx >= len(plan["steps"]):
            plan["status"] = "completed"
            plan["current_step_idx"] = next_idx   # Points past last step when done
        else:
            plan["current_step_idx"] = next_idx
            plan["steps"][next_idx]["status"] = "active"

        data[session_id] = plan
        self._save(data)
        return plan

    def abort_plan(self, session_id: str):
        data = self._load()
        if session_id in data:
            data[session_id]["status"] = "aborted"
            self._save(data)

    def summarize_completed_plan(self, session_id: str) -> Optional[str]:
        """
        After a plan completes, generate a brief summary for episodic memory encoding.
        This allows the hippocampus to consolidate the entire multi-step execution
        as a single high-level memory trace (chunking).

        Returns a compact summary string or None if no completed plan.
        """
        data = self._load()
        plan = data.get(session_id)
        if not plan or plan.get("status") not in ("completed", "aborted"):
            return None

        status_str = plan["status"].upper()
        steps_summary = []
        for step in plan["steps"]:
            step_status = step.get("status", "pending")
            emoji = "✓" if step_status == "completed" else ("✗" if step_status == "pending" else "-")
            steps_summary.append(f"  {emoji} Step {step['step_id']}: {step['goal'][:60]}")

        summary = (
            f"[PLAN {status_str}] Goal: {plan['original_query'][:100]}\n"
            + "\n".join(steps_summary)
        )
        return summary

    def clear_plan(self, session_id: str):
        """Removes a completed/aborted plan from the store."""
        data = self._load()
        if session_id in data:
            del data[session_id]
            self._save(data)

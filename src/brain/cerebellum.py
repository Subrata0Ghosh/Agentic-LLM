"""
Cerebellum — src/brain/cerebellum.py
======================================
Brain Region: Cerebellum (Latin: "little brain")

The cerebellum is responsible for:
  1. Timing — precise coordination of sequential operations
  2. Error correction — learns to predict and correct motor/behavioral errors
  3. Procedural refinement — smooths learned sequences over many repetitions

In a cognitive AI pipeline, the cerebellum maps to:
  - Detecting when the AI is repeating itself (perseveration)
  - Predicting how long/complex a response should be given the query type
  - Smoothing the step ordering produced by PlanningPFC
  - Signaling when a response sequence is "out of sync" with user expectations

Biological basis:
  - The cerebellum contains more neurons than the entire rest of the brain
  - It fires BEFORE a movement, not during — it predicts optimal output
  - Purkinje cells encode the difference between expected and actual timing
  - Cerebellar damage causes ataxia: accurate but uncoordinated movements

Neuromodulation:
  - Norepinephrine modulates cerebellar plasticity during error correction
  - Dopamine gates whether cerebellar corrections propagate to motor cortex
"""

import os
import json
import time
import re
from collections import deque
from typing import List, Dict, Optional, Tuple

CEREBELLUM_FILE = "data/cerebellum.json"

# Expected response length ranges by query type (in words)
TIMING_MAP = {
    "greeting":    (10, 40),
    "math":        (20, 60),
    "explanation": (80, 300),
    "comparison":  (100, 400),
    "planning":    (150, 500),
    "creative":    (100, 400),
    "factual":     (30, 150),
    "general":     (50, 200),
}

# Perseveration: if similarity between last N responses exceeds threshold
PERSEVERATION_WINDOW = 4
PERSEVERATION_THRESH = 0.65


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class Cerebellum:
    """
    Timing, sequence coordination, and perseveration detection module.
    Models the biological cerebellum's role in precise behavioral sequencing.
    """

    def __init__(self, filepath: str = CEREBELLUM_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # In-memory response buffers per session
        self._response_buffers: Dict[str, deque] = {}

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _get_session(self, session_id: str) -> dict:
        data = self._load()
        if session_id not in data:
            data[session_id] = {
                "turn_count":          0,
                "perseveration_count": 0,
                "timing_errors":       0,
                "last_updated":        time.time(),
            }
            self._save(data)
        return data[session_id]

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY TYPE CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    def classify_query(self, query: str) -> str:
        """
        Classify the query type for timing prediction.
        Returns one of the keys in TIMING_MAP.
        """
        q = query.lower()
        tokens = set(re.findall(r"\b\w+\b", q))
        greetings = {"hi", "hello", "hey", "yo", "sup"}
        if (tokens & greetings) or any(phrase in q for phrase in ["good morning", "good evening", "good afternoon"]):
            return "greeting"
        if any(op in q for op in ["+", "-", "*", "/", "calculate", "solve", "equation"]):
            return "math"
        if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast"]):
            return "comparison"
        if any(w in q for w in ["plan", "step by step", "how to build", "roadmap", "stages"]):
            return "planning"
        if any(w in q for w in ["explain", "describe", "what is", "why does", "how does"]):
            return "explanation"
        if any(w in q for w in ["write", "create", "story", "poem", "imagine", "generate"]):
            return "creative"
        if any(w in q for w in ["who", "when", "where", "which year", "what is the name"]):
            return "factual"
        return "general"

    # ══════════════════════════════════════════════════════════════════════════
    # TIMING PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    def predict_response_length(self, query: str) -> Dict:
        """
        Predicts the expected response word count range based on query type.
        Analogous to the cerebellum pre-activating motor programs before movement.

        Returns:
            {
              "query_type":  str,
              "min_words":   int,
              "max_words":   int,
              "guidance":    str   (brief instruction for generation)
            }
        """
        query_type = self.classify_query(query)
        min_w, max_w = TIMING_MAP.get(query_type, (50, 200))

        if query_type == "greeting":
            guidance = "Reply briefly and warmly. 1-3 sentences."
        elif query_type == "math":
            guidance = "Show working steps. Be precise. Include the final answer prominently."
        elif query_type == "comparison":
            guidance = "Use a structured format. Compare clearly on multiple dimensions."
        elif query_type == "planning":
            guidance = "Break into numbered steps. Be sequential and complete."
        elif query_type == "explanation":
            guidance = "Explain clearly with examples. Build from simple to complex."
        elif query_type == "creative":
            guidance = "Be imaginative and engaging. Flow naturally."
        elif query_type == "factual":
            guidance = "Answer directly and concisely with the key fact."
        else:
            guidance = "Be helpful and clear."

        return {
            "query_type": query_type,
            "min_words":  min_w,
            "max_words":  max_w,
            "guidance":   guidance,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PERSEVERATION DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    def detect_perseveration(self, session_id: str, response: str) -> Dict:
        """
        Checks if the latest response is too similar to recent responses.
        Biological analog: cerebellar detection of stuck motor programs.

        Returns:
            {
              "is_perseverating": bool,
              "max_similarity":   float,
              "warning":          str or None
            }
        """
        if session_id not in self._response_buffers:
            self._response_buffers[session_id] = deque(maxlen=PERSEVERATION_WINDOW)

        buf = self._response_buffers[session_id]
        sims = [_jaccard(response, prev) for prev in buf]
        max_sim = max(sims) if sims else 0.0

        # Store current response
        buf.append(response)

        is_persev = max_sim >= PERSEVERATION_THRESH
        warning = None
        if is_persev:
            # Update persistent count
            data = self._load()
            sess = self._get_session(session_id)
            sess["perseveration_count"] = sess.get("perseveration_count", 0) + 1
            data[session_id] = sess
            self._save(data)
            warning = (
                f"Perseveration detected (similarity={max_sim:.2f}). "
                "Response is too similar to recent outputs. Consider rephrasing."
            )

        return {
            "is_perseverating": is_persev,
            "max_similarity":   round(max_sim, 3),
            "warning":          warning,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # SEQUENCE REFINEMENT (Plan Step Smoothing)
    # ══════════════════════════════════════════════════════════════════════════
    def refine_sequence(self, steps: List[Dict]) -> List[Dict]:
        """
        Smooths the step ordering from PlanningPFC.
        Analogous to cerebellum refining coarse motor commands from motor cortex.

        Heuristics:
          - Reorders if an 'analyze/prepare' step is not first
          - Ensures 'verify/test/check' steps are last
          - Deduplicates near-identical adjacent steps
        """
        if not steps or len(steps) < 2:
            return steps

        refined = list(steps)

        # 1. Promote analysis/prepare steps to front
        for i, step in enumerate(refined):
            goal_l = step.get("goal", "").lower()
            if i > 0 and any(w in goal_l for w in ["analyz", "prepar", "understand", "assess"]):
                # Move to position 1 (after step 0)
                refined.insert(1, refined.pop(i))
                break

        # 2. Demote verification steps to end
        verify_steps = []
        main_steps   = []
        for step in refined:
            goal_l = step.get("goal", "").lower()
            if any(w in goal_l for w in ["verify", "test", "check", "validate", "review"]):
                verify_steps.append(step)
            else:
                main_steps.append(step)
        refined = main_steps + verify_steps

        # 3. Deduplicate adjacent very-similar steps
        deduped = [refined[0]] if refined else []
        for step in refined[1:]:
            if _jaccard(step.get("goal", ""), deduped[-1].get("goal", "")) < 0.8:
                deduped.append(step)
        refined = deduped

        # Re-number step_ids
        for i, step in enumerate(refined):
            step["step_id"] = i + 1

        return refined

    # ══════════════════════════════════════════════════════════════════════════
    # RESPONSE LENGTH VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    def validate_response_length(self, query: str, response: str) -> Dict:
        """
        Checks whether the actual response length matches the predicted range.
        Returns a timing verdict and optional adjustment suggestion.
        """
        timing = self.predict_response_length(query)
        actual_words = len(response.split())
        min_w = timing["min_words"]
        max_w = timing["max_words"]

        if actual_words < min_w:
            verdict = "too_short"
            suggestion = f"Response may be too brief ({actual_words} words). Consider elaborating."
        elif actual_words > max_w:
            verdict = "too_long"
            suggestion = f"Response may be too verbose ({actual_words} words). Consider condensing."
        else:
            verdict = "ok"
            suggestion = None

        return {
            "verdict":      verdict,
            "actual_words": actual_words,
            "expected_min": min_w,
            "expected_max": max_w,
            "suggestion":   suggestion,
            "query_type":   timing["query_type"],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STATS
    # ══════════════════════════════════════════════════════════════════════════
    def get_stats(self, session_id: str) -> dict:
        """Returns cerebellar stats for a session."""
        return self._get_session(session_id)

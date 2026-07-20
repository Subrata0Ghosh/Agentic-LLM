"""
Theory of Mind — src/brain/theory_of_mind.py
==============================================
Brain Region: Temporoparietal Junction (TPJ) + Medial PFC

Theory of Mind (ToM) = the ability to attribute mental states
(beliefs, desires, knowledge, intentions) to OTHERS.

Humans automatically build a model of what other people know,
don't know, and are confused about. This is why a good teacher
doesn't explain things you already know.

Per-user model:
  - Expertise level (novice / intermediate / expert)
  - Knowledge gaps (topics they asked about repeatedly or got 👎)
  - Emotional state (frustrated / curious / satisfied / confused)
  - Communication style preference (brief / detailed / examples-heavy)
  - Questions they've asked (topic map of what they've explored)

How it's used:
  - DMN adjusts explanation depth based on inferred expertise
  - System generates different prompts for novice vs expert users
  - Detects frustration → shift to simpler language + more reassurance
  - Detects confusion → proactively offer alternative explanations
"""

import os
import json
import time
from typing import Optional

TOM_FILE = "data/theory_of_mind.json"

EXPERTISE_LEVELS = ["novice", "learner", "intermediate", "advanced", "expert"]


class TheoryOfMind:
    """Models the cognitive and emotional state of the USER."""

    def __init__(self, filepath: str = TOM_FILE):
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

    def _get_user(self, session_id: str) -> dict:
        data = self._load()
        if session_id not in data:
            data[session_id] = {
                "expertise_score":  0.5,    # 0.0 (novice) → 1.0 (expert)
                "expertise_label":  "learner",
                "emotional_state":  "curious",
                "topics_explored":  {},     # topic → count
                "confusion_count":  0,
                "frustration_level": 0.0,
                "satisfaction_level": 0.5,
                "prefers_brief":    False,
                "total_turns":      0,
                "avg_query_length": 0,
                "correction_count": 0,      # How many 👎 signals
                "clarification_asks": 0,    # How often system had to ask for clarification
            }
            self._save(data)
        return data[session_id]

    def _save_user(self, session_id: str, user: dict):
        data = self._load()
        data[session_id] = user
        self._save(data)

    def _update_expertise_label(self, score: float) -> str:
        idx = min(int(score * len(EXPERTISE_LEVELS)), len(EXPERTISE_LEVELS) - 1)
        return EXPERTISE_LEVELS[idx]

    def observe_query(self, session_id: str, query: str, is_correction: bool = False):
        """
        Update user model based on a new query.
        Infers expertise from vocabulary complexity, query length, topic depth.
        """
        user = self._get_user(session_id)
        words = query.split()
        n = len(words)

        # Running average query length
        t = user["total_turns"] + 1
        user["avg_query_length"] = round(
            (user["avg_query_length"] * (t-1) + n) / t, 1
        )
        user["total_turns"] = t

        # Expertise signal: long, complex queries → higher expertise
        avg_word_len = sum(len(w) for w in words) / max(n, 1)
        complexity_signal = (avg_word_len - 3.0) / 6.0   # Normalize to ~0-1
        complexity_signal = max(0.0, min(1.0, complexity_signal))

        # Questions asking "what" = often novice; "how does X compare to Y" = more advanced
        if any(q in query.lower() for q in ["compare", "trade-off", "versus", "alternative", "complexity", "architecture"]):
            complexity_signal = min(1.0, complexity_signal + 0.2)

        # Update expertise score (slow drift)
        user["expertise_score"] = round(
            user["expertise_score"] * 0.92 + complexity_signal * 0.08, 3
        )
        user["expertise_label"] = self._update_expertise_label(user["expertise_score"])

        # Topic exploration tracking
        topic_words = [w.lower() for w in words if len(w) > 4][:5]
        for tw in topic_words:
            user["topics_explored"][tw] = user["topics_explored"].get(tw, 0) + 1

        if is_correction:
            user["correction_count"] += 1
            user["frustration_level"] = min(1.0, user["frustration_level"] + 0.12)

        self._save_user(session_id, user)

    def observe_feedback(self, session_id: str, positive: bool):
        """Update emotional state from 👍/👎 feedback."""
        user = self._get_user(session_id)
        if positive:
            user["satisfaction_level"] = min(1.0, user["satisfaction_level"] + 0.10)
            user["frustration_level"]  = max(0.0, user["frustration_level"]  - 0.08)
            user["emotional_state"] = "satisfied" if user["satisfaction_level"] > 0.7 else "curious"
        else:
            user["frustration_level"]  = min(1.0, user["frustration_level"]  + 0.15)
            user["satisfaction_level"] = max(0.0, user["satisfaction_level"] - 0.08)
            user["emotional_state"] = "frustrated" if user["frustration_level"] > 0.6 else "confused"
        self._save_user(session_id, user)

    def observe_confusion(self, session_id: str):
        """Called when metacognition detects confusion."""
        user = self._get_user(session_id)
        user["confusion_count"] += 1
        user["frustration_level"] = min(1.0, user["frustration_level"] + 0.08)
        if user["confusion_count"] > 3:
            user["emotional_state"] = "confused"
        self._save_user(session_id, user)

    def get_generation_profile(self, session_id: str) -> dict:
        """
        Returns a profile dict used to calibrate the system prompt for this user.

        Used by: DefaultModeNetwork, api.py system prompt building.
        """
        user = self._get_user(session_id)
        expertise = user["expertise_label"]
        frustration = user["frustration_level"]
        satisfaction = user["satisfaction_level"]
        emotional_state = user["emotional_state"]

        # Depth instruction for generation
        if expertise in ["novice", "learner"]:
            depth = "Use simple language, avoid jargon, give concrete examples first."
        elif expertise == "intermediate":
            depth = "Balance concepts with examples. Use technical terms with brief explanations."
        else:  # advanced, expert
            depth = "Assume technical background. Be precise and concise. Skip basics."

        # Tone instruction
        if frustration > 0.6:
            tone = "Be extra patient, reassuring, and break things into small clear steps."
        elif emotional_state == "curious":
            tone = "Match curiosity with enthusiasm. Offer interesting connections and next topics."
        else:
            tone = "Be clear and professional."

        return {
            "expertise":   expertise,
            "depth":       depth,
            "tone":        tone,
            "emotional_state": emotional_state,
            "frustration": round(frustration, 2),
            "satisfaction": round(satisfaction, 2),
            "prefers_brief": user["avg_query_length"] < 5,
        }

    def get_user_summary(self, session_id: str) -> dict:
        user = self._get_user(session_id)
        return {
            "expertise":         user["expertise_label"],
            "emotional_state":   user["emotional_state"],
            "frustration":       round(user["frustration_level"], 2),
            "satisfaction":      round(user["satisfaction_level"], 2),
            "topics_explored":   sorted(user["topics_explored"].items(),
                                        key=lambda x: x[1], reverse=True)[:8],
            "correction_count":  user["correction_count"],
            "total_turns":       user["total_turns"],
        }

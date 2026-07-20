"""
Thalamus — src/brain/thalamus.py
=================================
Brain Region: Thalamus (sensory gateway + attentional filter)

The thalamus is the first stop for nearly all sensory information.
It does NOT process — it gates. 98% of incoming signals are suppressed.
Only salient, novel, or urgent signals are routed for deep cortical processing.

Two routing paths:
  FAST PATH  → Thalamus → Procedural Memory → Direct output
              (greetings, reflexes, cached habits)

  DEEP PATH  → Thalamus → Full cortical pipeline
              (novel queries, high-salience events, prediction errors)

Salience = f(novelty, urgency, emotional_load, complexity)
"""

import re
from typing import Tuple

# Words that signal urgency / emotional weight
URGENCY_VOCAB = {
    "help", "urgent", "emergency", "critical", "error", "broken", "fix",
    "wrong", "fail", "crash", "died", "lost", "confused", "cannot", "can't",
    "why", "how", "explain", "difference", "compare", "best", "should",
    "important", "need", "must", "always", "never", "danger", "risk"
}

# Typical reflexive / low-salience patterns (fast path)
REFLEX_PATTERNS = [
    r"^(hi|hello|hey|hola|yo|sup)\b",
    r"^(good\s+)(morning|afternoon|evening|night)\b",
    r"^(how are you|what'?s up|who are you|what is your name)\b",
    r"^(thank(s| you)|bye|goodbye|ok|okay|sure|got it|nice|cool|great)\b",
    r"^(yes|no|maybe|indeed|correct|right|exactly)\b",
]

_REFLEX_RE = [re.compile(p, re.IGNORECASE) for p in REFLEX_PATTERNS]


class Thalamus:
    """
    Sensory gateway: computes salience and decides routing path.
    """

    def __init__(self, deep_threshold: float = 0.35):
        """
        deep_threshold: salience score above which deep processing is triggered.
        Below this → fast path (reflex/habit).
        """
        self.deep_threshold = deep_threshold

    def compute_salience(self, text: str, emotion_tag: dict = None) -> dict:
        """
        Multi-dimensional salience score in [0, 1].

        Components:
          novelty_score      — based on rare/technical vocabulary
          urgency_score      — urgency/question word density
          complexity_score   — word count + sentence structure
          question_score     — is it a genuine question?
          emotional_load     — amygdala arousal signal (if provided)
        """
        text_lower = text.lower().strip()
        words = text_lower.split()
        n = max(len(words), 1)

        # 1. Question detection
        is_question = text.strip().endswith("?") or any(
            text_lower.startswith(w) for w in
            ["what", "why", "how", "when", "where", "who", "which", "explain", "describe", "compare"]
        )
        question_score = 0.4 if is_question else 0.0

        # 2. Urgency vocabulary density
        urgency_hits = sum(1 for w in words if w in URGENCY_VOCAB)
        urgency_score = min(1.0, urgency_hits / n * 4.0)

        # 3. Complexity (length penalty inverted — longer = more complex)
        complexity_score = min(1.0, n / 20.0)

        # 4. Novelty — approximated by average word length (longer words = rarer)
        avg_word_len = sum(len(w) for w in words) / n
        novelty_score = min(1.0, (avg_word_len - 3.0) / 5.0) if avg_word_len > 3 else 0.0
        novelty_score = max(0.0, novelty_score)

        # 5. Emotional load — from amygdala arousal signal (external, optional)
        emotional_load = 0.0
        if emotion_tag and isinstance(emotion_tag, dict):
            emotional_load = float(emotion_tag.get("arousal", 0.0))
            emotional_load = max(0.0, min(1.0, emotional_load))

        salience = (
            0.25 * question_score +
            0.25 * urgency_score +
            0.20 * complexity_score +
            0.15 * novelty_score +
            0.15 * emotional_load
        )

        return {
            "salience":        round(salience, 3),
            "question_score":  round(question_score, 3),
            "urgency_score":   round(urgency_score, 3),
            "complexity":      round(complexity_score, 3),
            "novelty":         round(novelty_score, 3),
            "emotional_load":  round(emotional_load, 3),
            "word_count":      n,
        }

    def is_reflex(self, text: str) -> bool:
        """Returns True if this is a low-effort reflex (fast-path only)."""
        # Strip punctuation AND commas for better pattern matching
        stripped = text.strip().rstrip("!?.,;:")
        # Also strip trailing salutation punctuation like "Hey," → "Hey"
        stripped = stripped.rstrip(",")
        return any(pattern.match(stripped) for pattern in _REFLEX_RE)

    def route(self, text: str, nm_state: dict = None,
              emotion_tag: dict = None) -> Tuple[str, dict]:
        """
        Main routing decision.

        Returns:
          (path, salience_info)
          path = "REFLEX"  → fast path, skip deep cortical pipeline
          path = "DEEP"    → full brain pipeline
          path = "FOCUS"   → high-salience, all resources dedicated
        """
        salience_info = self.compute_salience(text, emotion_tag=emotion_tag)
        score = salience_info["salience"]

        # Norepinephrine modulates threshold: high NE = more things pass as important
        ne = (nm_state or {}).get("norepinephrine", 0.5)
        effective_threshold = self.deep_threshold * (1.2 - ne * 0.4)

        if self.is_reflex(text):
            return "REFLEX", salience_info

        # Math, planning and memory encoding instructions require deep cortical processing
        text_lower = text.lower().strip()
        has_digits = len(re.findall(r"\d+", text_lower)) >= 2
        has_ops = any(op in text_lower for op in ["+", "-", "*", "/", "=", "^"])
        is_math = (has_digits and has_ops) or any(w in text_lower for w in ["solve", "calculate", "evaluate", "compute", "equation"])
        is_plan = any(kw in text_lower for kw in ["then", "first", "next", "after that", "finally", "step by step", "plan"]) and len(text_lower.split()) > 8
        is_memory_inst = any(w in text_lower for w in ["remember", "memorize", "recall", "store", "keep in mind"])

        # Epistemic need: genuine questions must always trigger deep memory retrieval and DMN
        is_question = text.strip().endswith("?") or any(
            text_lower.startswith(w) for w in
            ["what", "why", "how", "when", "where", "who", "which", "explain", "describe", "compare", "wht"]
        )

        if is_question or is_math or is_plan or is_memory_inst:
            return "FOCUS" if score > 0.75 else "DEEP", salience_info

        if score < effective_threshold:
            return "REFLEX", salience_info
        elif score > 0.75:
            return "FOCUS", salience_info
        else:
            return "DEEP", salience_info

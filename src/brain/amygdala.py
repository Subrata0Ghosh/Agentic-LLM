"""
Amygdala — src/brain/amygdala.py
==================================
Brain Region: Amygdala (medial temporal lobe)

The amygdala assigns emotional valence and urgency to incoming information.
It does NOT produce language — it tags memory traces with:
  - Valence  : positive / negative / neutral
  - Arousal  : calm → distressing (0 → 1)
  - Priority : how urgently this memory should be consolidated

Key biological roles:
  1. Emotional tagging: every hippocampal memory trace gets an amygdala tag.
     High-emotion memories are replayed MORE during sleep (REM phase).
  2. Serotonin gating: high SE = amygdala dampened (calm, reflective).
     Low SE = amygdala amplified (anxious, reactive, over-sensitive).
  3. Fight-or-flight gate: extreme negative arousal bypasses PFC reasoning
     entirely (this is the "amygdala hijack" in humans).
"""

import re
from typing import Tuple

# Emotional vocabulary maps
POSITIVE_WORDS = {
    "love", "great", "excellent", "happy", "good", "wonderful", "amazing",
    "fantastic", "beautiful", "perfect", "thank", "glad", "excited", "fun",
    "nice", "interesting", "curious", "brilliant", "helpful", "learn",
    "understand", "discover", "solve", "create", "build",
}

NEGATIVE_WORDS = {
    "hate", "terrible", "awful", "broken", "fail", "error", "crash",
    "wrong", "bad", "problem", "issue", "confused", "lost", "stuck",
    "difficult", "impossible", "frustrating", "annoying", "useless",
    "cannot", "can't", "won't", "doesn't", "never", "worst",
}

NEUTRAL_WORDS = {
    "what", "how", "when", "where", "who", "which", "tell", "explain",
    "describe", "show", "list", "define",
}


class Amygdala:
    """
    Emotional tagger for the cognitive pipeline.
    All information passing through gets tagged before hippocampal storage.
    """

    def tag(self, text: str, serotonin_level: float = 0.6) -> dict:
        """
        Compute emotional metadata for a piece of text.

        Args:
            text:             The input query or content to tag.
            serotonin_level:  Current SE level (0-1). High SE dampens reactivity.

        Returns:
            {
              valence:  "positive" | "negative" | "neutral"
              arousal:  float [0, 1]  — emotional intensity
              priority: float [0, 1]  — memory consolidation priority
              hijack:   bool          — True if amygdala hijack detected
            }
        """
        words = set(text.lower().split())

        pos_hits = len(words & POSITIVE_WORDS)
        neg_hits = len(words & NEGATIVE_WORDS)
        total = max(len(words), 1)

        pos_density = pos_hits / total
        neg_density = neg_hits / total

        # Serotonin dampens amygdala sensitivity
        # High SE (0.9) → only 10% reactivity; Low SE (0.1) → 90% reactivity
        se_damping = 1.0 - (serotonin_level * 0.8)

        raw_arousal = (pos_density + neg_density * 1.5) * se_damping
        arousal = min(1.0, raw_arousal * 3.0)

        # Determine valence
        if neg_hits > pos_hits:
            valence = "negative"
        elif pos_hits > neg_hits:
            valence = "positive"
        else:
            valence = "neutral"

        # Priority: negative emotional events get highest priority
        # (biologically, threats are remembered better than rewards)
        if valence == "negative":
            priority = min(1.0, arousal * 1.3)
        elif valence == "positive":
            priority = min(1.0, arousal * 0.8)
        else:
            priority = arousal * 0.4

        # Amygdala hijack: extremely high negative arousal bypasses PFC
        hijack = (valence == "negative") and (arousal > 0.85) and (serotonin_level < 0.3)

        return {
            "valence":  valence,
            "arousal":  round(arousal, 3),
            "priority": round(priority, 3),
            "hijack":   hijack,
        }

    def tag_response(self, text: str, serotonin_level: float = 0.6) -> dict:
        """Tag a generated response (for social interaction SE modulation)."""
        tag = self.tag(text, serotonin_level)
        is_positive_interaction = tag["valence"] == "positive" and tag["arousal"] > 0.1
        return {**tag, "positive_interaction": is_positive_interaction}

    def adjust_se_on_response(self, serotonin_level: float, response_tag: dict) -> float:
        """
        Adjusts serotonin based on interaction quality.
        Helpful, positive exchanges elevate SE; hostile ones suppress it.
        """
        if response_tag.get("positive_interaction"):
            return min(1.0, serotonin_level + 0.03)
        elif response_tag.get("valence") == "negative":
            return max(0.05, serotonin_level - 0.04)
        return serotonin_level

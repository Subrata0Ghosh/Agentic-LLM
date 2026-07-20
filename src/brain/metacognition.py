"""
Metacognition — src/brain/metacognition.py
============================================
Brain Region: Anterior Cingulate Cortex (ACC) + Medial PFC

Metacognition = "thinking about thinking". The brain's self-monitoring system.
The ACC is the brain's conflict monitor — it detects when processing is
difficult, uncertain, or error-prone and signals that more resources are needed.

Biological basis:
  - Detects "cognitive load": how hard is the brain working?
  - Detects "confusion state": high sustained prediction error
  - Triggers adaptive behaviors: sleep when overloaded, ask for help when stuck
  - Generates confidence estimates for each output

Key functions:
  1. Cognitive Load Meter: WM fullness + prediction error + fatigue combined
  2. Confusion Detection: rolling error > 0.7 for 3+ turns → auto-sleep
  3. Confidence Scoring: per-response quality estimate for UI display
  4. Accuracy Tracking: uses reward signals to estimate true accuracy
  5. Auto-Sleep Trigger: when confused, initiates sleep consolidation
"""

import os
import json
import time
from collections import deque

METACOG_FILE = "data/metacognition.json"
CONFUSION_WINDOW  = 3     # consecutive high-error turns before triggering
CONFUSION_THRESH  = 0.70  # prediction error threshold for "confused"
LOAD_AUTO_SLEEP   = 0.85  # cognitive load level triggering auto-sleep


class Metacognition:
    """Self-monitoring system for cognitive performance and load."""

    def __init__(self, filepath: str = METACOG_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # In-memory rolling windows per session
        self._error_windows: dict = {}

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
                "turn_count":       0,
                "cumulative_error": 0.0,
                "avg_confidence":   0.5,
                "auto_sleep_count": 0,
                "last_updated":     time.time(),
                "confusion_strikes": 0,
            }
            self._save(data)
        return data[session_id]

    def _save_session(self, session_id: str, sess: dict):
        data = self._load()
        data[session_id] = sess
        self._save(data)

    def update(self, session_id: str, prediction_error: float,
               wm_chunks: int, ne_level: float,
               reward_accuracy: float = None) -> dict:
        """
        Update metacognitive state after each turn.

        Args:
            prediction_error: Current turn prediction error (0-1)
            wm_chunks:        Working memory fullness (0-4)
            ne_level:         Norepinephrine (arousal/attention)
            reward_accuracy:  Rolling accuracy from reward_system (if available)

        Returns:
            {
              cognitive_load:    float [0,1]
              confidence:        float [0,1]
              is_confused:       bool
              should_sleep:      bool
              confusion_strikes: int
            }
        """
        sess = self._get_session(session_id)

        # Rolling error window (last 3 turns)
        if session_id not in self._error_windows:
            self._error_windows[session_id] = deque(maxlen=CONFUSION_WINDOW)
        self._error_windows[session_id].append(prediction_error)

        # Cognitive load = weighted combo of error, WM usage, arousal
        wm_load   = wm_chunks / 4.0              # 0-1
        ne_burden = max(0.0, ne_level - 0.5) * 2  # High NE = more stressed
        cog_load  = (
            0.50 * prediction_error +
            0.30 * wm_load +
            0.20 * ne_burden
        )
        cog_load = min(1.0, cog_load)

        # Confidence = inverse of prediction error (adjusted by reward accuracy if known)
        confidence = 1.0 - prediction_error
        if reward_accuracy is not None:
            # Blend model confidence with actual feedback accuracy
            confidence = 0.6 * confidence + 0.4 * reward_accuracy
        confidence = max(0.0, min(1.0, round(confidence, 3)))

        # Confusion detection: all errors in window above threshold
        window = list(self._error_windows[session_id])
        is_confused = (len(window) >= CONFUSION_WINDOW and
                       all(e >= CONFUSION_THRESH for e in window))

        if is_confused:
            sess["confusion_strikes"] = sess.get("confusion_strikes", 0) + 1
        else:
            sess["confusion_strikes"] = 0

        # Auto-sleep trigger: sustained confusion OR extreme cognitive load
        should_sleep = (is_confused and sess["confusion_strikes"] >= 2) or (cog_load >= LOAD_AUTO_SLEEP)

        # Update rolling stats
        n = sess["turn_count"] + 1
        sess["turn_count"] = n
        sess["cumulative_error"] = round(
            (sess["cumulative_error"] * (n-1) + prediction_error) / n, 3
        )
        sess["avg_confidence"] = round(
            (sess["avg_confidence"] * (n-1) + confidence) / n, 3
        )
        if should_sleep:
            sess["auto_sleep_count"] = sess.get("auto_sleep_count", 0) + 1
        sess["last_updated"] = time.time()
        sess["last_cog_load"]  = round(cog_load, 3)
        sess["last_confidence"] = confidence
        sess["confusion_strikes"] = sess.get("confusion_strikes", 0)

        self._save_session(session_id, sess)

        return {
            "cognitive_load":    round(cog_load, 3),
            "confidence":        confidence,
            "is_confused":       is_confused,
            "should_sleep":      should_sleep,
            "confusion_strikes": sess["confusion_strikes"],
        }

    def get_stats(self, session_id: str) -> dict:
        sess = self._get_session(session_id)
        return {
            "turn_count":       sess.get("turn_count", 0),
            "cumulative_error": sess.get("cumulative_error", 0.0),
            "avg_confidence":   sess.get("avg_confidence", 0.5),
            "auto_sleep_count": sess.get("auto_sleep_count", 0),
            "last_cog_load":    sess.get("last_cog_load", 0.0),
            "last_confidence":  sess.get("last_confidence", 0.5),
            "confusion_strikes": sess.get("confusion_strikes", 0),
        }

    def get_response_confidence_label(self, confidence: float) -> str:
        """Human-readable label for UI display."""
        if confidence >= 0.80: return "high"
        if confidence >= 0.55: return "medium"
        if confidence >= 0.30: return "low"
        return "uncertain"

    def frustration_detection(self, session_id: str,
                              recent_feedback: list = None) -> dict:
        """
        Detects user frustration from repeated negative feedback signals.
        Biological basis: ACC detects sustained conflict/displeasure.

        Args:
            recent_feedback: List of bools (True=positive, False=negative)

        Returns:
            {
              "is_frustrated": bool,
              "frustration_score": float [0,1],
              "consecutive_negative": int
            }
        """
        if not recent_feedback:
            recent_feedback = []

        # Count consecutive negatives from the end
        consecutive_neg = 0
        for fb in reversed(recent_feedback[-5:]):
            if not fb:
                consecutive_neg += 1
            else:
                break

        neg_count = sum(1 for f in recent_feedback[-5:] if not f)
        frustration_score = min(1.0, neg_count / max(len(recent_feedback[-5:]), 1))
        is_frustrated = consecutive_neg >= 2 or frustration_score >= 0.6

        return {
            "is_frustrated":        is_frustrated,
            "frustration_score":    round(frustration_score, 3),
            "consecutive_negative": consecutive_neg,
        }

    def recovery_action(self, session_id: str) -> str:
        """
        Recommends a recovery action when the system is confused or frustrated.
        Biological basis: Orbitofrontal cortex generates alternative strategies
        when the current approach repeatedly fails.

        Returns a human-readable recommendation string.
        """
        sess = self._get_session(session_id)
        strikes = sess.get("confusion_strikes", 0)
        load    = sess.get("last_cog_load", 0.0)
        conf    = sess.get("last_confidence", 0.5)

        if strikes >= 3 or load >= 0.85:
            return (
                "Cognitive overload detected. Recommend: (1) trigger sleep consolidation, "
                "(2) simplify current query, (3) reset session context."
            )
        elif conf < 0.35:
            return (
                "Low confidence. Recommend: (1) use /api/learn to ingest more knowledge, "
                "(2) ask user for clarification, (3) switch to exploratory strategy."
            )
        elif strikes >= 1:
            return (
                "Mild confusion detected. Recommend: (1) request user clarification, "
                "(2) activate epistemic foraging (web search)."
            )
        return "System operating normally. No recovery action needed."

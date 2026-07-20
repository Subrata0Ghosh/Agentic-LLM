"""
Reward System — src/brain/reward_system.py
============================================
Brain Region: Nucleus Accumbens + Ventral Tegmental Area (VTA)

Biological basis:
  The brain's reinforcement learning circuit:
  👍 Positive feedback → phasic DA burst from VTA → strengthen synapses
  👎 Negative feedback → DA dip below baseline → weaken/inhibit those pathways

This is how the brain distinguishes "that worked" from "that didn't".
Without this signal, the system can only predict — not truly IMPROVE
from explicit user feedback.

Roles:
  1. Record reward signals (👍 / 👎) per memory / session
  2. Emit appropriate neuromodulator signals
  3. Mark hippocampal memories as "correct" or "incorrect"
  4. Adjust procedural habit strength
  5. Track per-topic accuracy over time (learning curve)
"""

import os
import json
import time

REWARD_FILE = "data/reward_log.json"


class RewardSystem:
    """Reinforcement learning from explicit user feedback."""

    def __init__(self, filepath: str = REWARD_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {"sessions": {}, "global_accuracy": 0.5, "total_signals": 0}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def record_feedback(self, session_id: str, message_id: str,
                         positive: bool, topic: str = "general",
                         response_snippet: str = "") -> dict:
        """
        Record 👍 or 👎 feedback and compute resulting neuromodulator signals.

        Returns:
            {
              da_delta:      float  — dopamine change (positive or negative)
              ne_delta:      float  — norepinephrine change
              strengthen:    bool   — should habit/memory be reinforced?
              inhibit:       bool   — should path be inhibited?
              accuracy:      float  — rolling topic accuracy
            }
        """
        data = self._load()
        if session_id not in data["sessions"]:
            data["sessions"][session_id] = {
                "positive": 0, "negative": 0,
                "topics": {}, "signals": []
            }

        sess = data["sessions"][session_id]

        if positive:
            sess["positive"] += 1
            da_delta = +0.25   # Phasic DA burst
            ne_delta = +0.05   # Mild arousal (validation)
            strengthen = True
            inhibit = False
        else:
            sess["negative"] += 1
            da_delta = -0.18   # DA dip (prediction error below baseline)
            ne_delta = +0.15   # NE spike (error signal = re-route needed)
            strengthen = False
            inhibit = True

        # Topic-level accuracy tracking
        if topic not in sess["topics"]:
            sess["topics"][topic] = {"pos": 0, "neg": 0}
        if positive:
            sess["topics"][topic]["pos"] += 1
        else:
            sess["topics"][topic]["neg"] += 1

        # Rolling accuracy for topic
        t = sess["topics"][topic]
        topic_total = t["pos"] + t["neg"]
        topic_accuracy = t["pos"] / topic_total if topic_total > 0 else 0.5

        # Global accuracy
        total_pos = sum(s["positive"] for s in data["sessions"].values())
        total_neg = sum(s["negative"] for s in data["sessions"].values())
        total = total_pos + total_neg
        data["global_accuracy"] = total_pos / total if total > 0 else 0.5
        data["total_signals"] = total

        # Log signal
        sess["signals"].append({
            "ts":       time.time(),
            "positive": positive,
            "topic":    topic,
            "snippet":  response_snippet[:80],
        })
        # Keep last 100 signals
        sess["signals"] = sess["signals"][-100:]

        data["sessions"][session_id] = sess
        self._save(data)

        return {
            "da_delta":     round(da_delta, 3),
            "ne_delta":     round(ne_delta, 3),
            "strengthen":   strengthen,
            "inhibit":      inhibit,
            "topic_accuracy": round(topic_accuracy, 3),
            "global_accuracy": round(data["global_accuracy"], 3),
        }

    def get_accuracy(self, session_id: str) -> dict:
        """Returns accuracy statistics for a session."""
        data = self._load()
        sess = data["sessions"].get(session_id, {})
        pos = sess.get("positive", 0)
        neg = sess.get("negative", 0)
        total = pos + neg
        return {
            "session_accuracy": round(pos / total, 3) if total > 0 else None,
            "total_feedback":   total,
            "positive":         pos,
            "negative":         neg,
            "topics":           sess.get("topics", {}),
            "global_accuracy":  round(data.get("global_accuracy", 0.5), 3),
        }

    def get_weak_topics(self, session_id: str, threshold: float = 0.5) -> list:
        """Returns topics where accuracy is below threshold — needs more learning."""
        data = self._load()
        sess = data["sessions"].get(session_id, {})
        topics = sess.get("topics", {})
        weak = []
        for topic, counts in topics.items():
            total = counts["pos"] + counts["neg"]
            if total > 0:
                acc = counts["pos"] / total
                if acc < threshold:
                    weak.append({"topic": topic, "accuracy": round(acc, 3), "total": total})
        return sorted(weak, key=lambda x: x["accuracy"])

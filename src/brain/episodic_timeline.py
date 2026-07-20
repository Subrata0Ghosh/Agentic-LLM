"""
Episodic Timeline — src/brain/episodic_timeline.py
=====================================================
Brain Region: Hippocampus (CA1, CA3) + Entorhinal Cortex

The human hippocampus doesn't just store memories — it stores them on a
TIMELINE. We can recall "what happened recently", "what changed since
last week", and navigate forward/backward through our personal history.

Key mechanisms:
  1. Temporal tagging: every memory has a precise timestamp
  2. Time-window retrieval: "what did we discuss in the last 10 minutes?"
  3. Session clustering: group memories into discrete sessions
  4. Delta detection: "what is NEW since the last session?"
  5. Temporal decay: recent memories more accessible than old ones

This extends hippocampus.py with time-aware retrieval that the
brain naturally has but standard vector search ignores.
"""

import os
import json
import time
from typing import List, Optional

TIMELINE_FILE = "data/episodic_timeline.json"


class EpisodicTimeline:
    """
    Time-aware episodic memory navigator.
    Tracks what was discussed, when, and what changed across sessions.
    """

    def __init__(self, filepath: str = TIMELINE_FILE):
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

    def record_event(self, session_id: str, role: str, content: str,
                      topic: str = "", emotion: str = "neutral",
                      priority: float = 0.1):
        """Record a single conversational event on the timeline."""
        data = self._load()
        if session_id not in data:
            data[session_id] = {"events": [], "sessions": [], "current_session_start": time.time()}

        event = {
            "ts":       time.time(),
            "role":     role,
            "content":  content[:300],
            "topic":    topic[:60],
            "emotion":  emotion,
            "priority": round(priority, 3),
        }
        data[session_id]["events"].append(event)

        # Cap at 500 events per session_id
        if len(data[session_id]["events"]) > 500:
            data[session_id]["events"] = data[session_id]["events"][-400:]

        self._save(data)

    def get_recent(self, session_id: str, minutes: int = 15) -> List[dict]:
        """
        Returns events from the last N minutes.
        Mirrors the brain's "immediate episodic buffer".
        """
        data = self._load()
        events = data.get(session_id, {}).get("events", [])
        cutoff = time.time() - (minutes * 60)
        return [e for e in events if e["ts"] >= cutoff]

    def get_session_summary(self, session_id: str) -> dict:
        """
        Summarizes the current session:
        - total turns, topics covered, dominant emotion
        """
        data = self._load()
        events = data.get(session_id, {}).get("events", [])
        if not events:
            return {"turns": 0, "topics": [], "dominant_emotion": "neutral", "duration_mins": 0}

        topics = list({e["topic"] for e in events if e["topic"]})
        emotions = [e["emotion"] for e in events]
        dominant = max(set(emotions), key=emotions.count) if emotions else "neutral"
        start = events[0]["ts"]
        duration = (time.time() - start) / 60.0

        return {
            "turns":            len(events),
            "topics":           topics,
            "dominant_emotion": dominant,
            "duration_mins":    round(duration, 1),
            "high_priority_count": sum(1 for e in events if e.get("priority", 0) > 0.4),
        }

    def get_delta_since_last_session(self, session_id: str,
                                      gap_threshold_mins: int = 60) -> dict:
        """
        Detects what is NEW since the last conversation session.
        A session boundary is detected by a gap > gap_threshold_mins.

        Returns:
            {
              is_returning_user: bool
              new_topics: list of topics not seen in previous session
              previous_topics: list of topics from last session
              gap_hours: float
            }
        """
        data = self._load()
        events = data.get(session_id, {}).get("events", [])
        if len(events) < 2:
            return {"is_returning_user": False, "new_topics": [], "previous_topics": [], "gap_hours": 0}

        # Find session boundaries by gap > threshold
        sessions = []
        current = [events[0]]
        for i in range(1, len(events)):
            gap = events[i]["ts"] - events[i-1]["ts"]
            if gap > gap_threshold_mins * 60:
                sessions.append(current)
                current = []
            current.append(events[i])
        sessions.append(current)

        if len(sessions) < 2:
            return {"is_returning_user": False, "new_topics": [], "previous_topics": [], "gap_hours": 0}

        prev_session = sessions[-2]
        curr_session = sessions[-1]

        prev_topics = {e["topic"] for e in prev_session if e["topic"]}
        curr_topics = {e["topic"] for e in curr_session if e["topic"]}
        new_topics = list(curr_topics - prev_topics)

        gap_hours = (curr_session[0]["ts"] - prev_session[-1]["ts"]) / 3600.0

        return {
            "is_returning_user": True,
            "new_topics":        new_topics,
            "previous_topics":   list(prev_topics),
            "gap_hours":         round(gap_hours, 2),
        }

    def get_timeline_for_ui(self, session_id: str, max_items: int = 20) -> list:
        """Returns a clean timeline list for UI display."""
        data = self._load()
        events = data.get(session_id, {}).get("events", [])
        out = []
        for e in events[-max_items:]:
            ts = e["ts"]
            mins_ago = round((time.time() - ts) / 60, 1)
            out.append({
                "time_ago": f"{mins_ago}m ago" if mins_ago < 60 else f"{round(mins_ago/60,1)}h ago",
                "role":     e["role"],
                "preview":  e["content"][:60] + "…" if len(e["content"]) > 60 else e["content"],
                "emotion":  e.get("emotion", "neutral"),
                "priority": e.get("priority", 0),
            })
        return list(reversed(out))  # Most recent first

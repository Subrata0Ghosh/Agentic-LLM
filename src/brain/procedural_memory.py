"""
Procedural Memory — src/brain/procedural_memory.py
====================================================
Brain Region: Cerebellum + Basal Ganglia

Procedural memory stores AUTOMATED routines and skills.
Once a pattern is repeated enough times, it becomes habitual —
bypassing the slow deliberate PFC reasoning pipeline entirely.

Biological basis:
  - Cerebellum: motor sequencing, error correction, prediction
  - Basal Ganglia: habit formation, reward-based selection
  - Key property: AUTOMATICITY — once learned, happens without conscious effort

Implementation:
  - Tracks query patterns across sessions
  - If similar query seen ≥ HABIT_THRESHOLD times → create a procedural habit
  - Habits are served instantly (reflex path) without deep pipeline
  - Habits DECAY over time if not reinforced (use it or lose it)
  - Habit strength is modulated by dopamine at time of formation

This makes the system genuinely faster over time for repeated tasks,
exactly like how humans become faster at things they practice.
"""

import os
import json
import time
import difflib
from typing import Optional

PROC_FILE = "data/procedural_memory.json"
HABIT_THRESHOLD = 3       # Times a pattern must repeat before becoming a habit
DECAY_RATE      = 0.05    # Strength reduction per day without reinforcement
MIN_STRENGTH    = 0.2     # Habits below this are forgotten


class ProceduralMemory:
    """
    Habit formation and automatic pattern caching system.
    Mimics the cerebellum / basal ganglia habit loop.
    """

    def __init__(self, filepath: str = PROC_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {"patterns": [], "habits": {}}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _normalize(self, text: str) -> str:
        """Normalize text for pattern matching."""
        return " ".join(sorted(text.lower().split()[:8]))

    def _similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def observe(self, query: str, response: str, dopamine: float = 0.35):
        """
        Record a query-response pair. If similar queries accumulate,
        form a procedural habit with strength proportional to DA at formation.
        """
        data = self._load()
        patterns = data.get("patterns", [])
        habits   = data.get("habits", {})

        norm_query = self._normalize(query)

        # Find similar existing pattern
        best_match_idx = None
        best_sim = 0.0
        for i, p in enumerate(patterns):
            sim = self._similarity(norm_query, p["norm"])
            if sim > best_sim:
                best_sim = sim
                best_match_idx = i

        if best_sim > 0.72 and best_match_idx is not None:
            # Increment count of this pattern
            patterns[best_match_idx]["count"] += 1
            patterns[best_match_idx]["last_seen"] = time.time()
            count = patterns[best_match_idx]["count"]

            # If threshold reached, promote to habit
            if count >= HABIT_THRESHOLD:
                habit_key = patterns[best_match_idx]["norm"]
                if habit_key not in habits:
                    strength = min(1.0, 0.5 + dopamine * 0.5)
                    habits[habit_key] = {
                        "query_template": query,
                        "cached_response": response[:600],
                        "strength":        round(strength, 3),
                        "formed_ts":       time.time(),
                        "last_used":       time.time(),
                        "use_count":       0,
                    }
                    print(f"[ProceduralMemory] Habit formed: '{query[:50]}'")
                else:
                    # Reinforce existing habit
                    habits[habit_key]["strength"] = min(
                        1.0, habits[habit_key]["strength"] + 0.1
                    )
                    habits[habit_key]["last_used"] = time.time()
        else:
            # New pattern
            patterns.append({
                "norm":      norm_query,
                "original":  query[:150],
                "count":     1,
                "last_seen": time.time(),
            })
            # Cap patterns list
            if len(patterns) > 200:
                patterns = sorted(patterns, key=lambda p: p["last_seen"], reverse=True)[:150]

        data["patterns"] = patterns
        data["habits"]   = habits
        self._save(data)

    def check_habit(self, query: str) -> Optional[dict]:
        """
        Check if a query matches an established habit.
        Returns the cached response if found and habit is strong enough.
        Also applies temporal decay to all habits.
        """
        data = self._load()
        habits = data.get("habits", {})
        norm_query = self._normalize(query)

        # Apply decay to all habits
        now = time.time()
        to_delete = []
        for key, habit in habits.items():
            days_since = (now - habit.get("last_used", now)) / 86400.0
            habit["strength"] = max(0.0, habit["strength"] - days_since * DECAY_RATE)
            if habit["strength"] < MIN_STRENGTH:
                to_delete.append(key)

        for key in to_delete:
            del habits[key]
            print(f"[ProceduralMemory] Habit decayed and forgotten: '{key[:40]}'")

        data["habits"] = habits
        self._save(data)

        # Find matching habit
        best_key = None
        best_sim  = 0.0
        for key in habits:
            sim = self._similarity(norm_query, key)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key and best_sim > 0.72 and habits[best_key]["strength"] > MIN_STRENGTH:
            habit = habits[best_key]
            # Update usage tracking
            habit["use_count"] += 1
            habit["last_used"] = time.time()
            data["habits"][best_key] = habit
            self._save(data)
            return {
                "cached_response": habit["cached_response"],
                "strength":        habit["strength"],
                "is_habit":        True,
            }

        return None

    def get_all_habits(self) -> list:
        """Returns a summary of all active habits."""
        data = self._load()
        return [
            {"query": v["query_template"][:60],
             "strength": v["strength"],
             "use_count": v["use_count"]}
            for v in data.get("habits", {}).values()
        ]

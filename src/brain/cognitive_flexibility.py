"""
Cognitive Flexibility — src/brain/cognitive_flexibility.py
=============================================================
Brain Region: Anterior Cingulate Cortex (ACC) + Orbitofrontal Cortex (OFC)

Cognitive flexibility is the ability to shift strategies or rule sets when
the current approach fails. The ACC monitors conflict/errors, and when error
is high, it triggers a "set shift" (strategy change).

Key mechanisms:
  1. Strategy monitoring: Tracks what mode/rule is active.
  2. Set shifting: Shifts from Habitual (REFLEX) -> Analytical (DEEP) -> Exploratory.
  3. Dynamic neuromodulation: Triggers NE/DA spikes on strategy shift to break
     stuck loops or repetitive queries (perseveration).
"""

import os
import json
import time

FLEX_FILE = "data/cognitive_flexibility.json"

class CognitiveFlexibility:
    """
    Monitors errors and shifts reasoning strategies representing the ACC/OFC.
    """

    def __init__(self, filepath: str = FLEX_FILE):
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

    def evaluate_strategy(self, session_id: str, query: str,
                           prediction_error: float, negative_feedback: bool) -> dict:
        """
        Evaluates current performance. If conflict/error is high, switches strategy.

        Returns:
            {
              "active_strategy": str ("habitual" | "analytical" | "exploratory" | "logical"),
              "strategy_shifted": bool,
              "ne_boost": float,
              "da_boost": float
            }
        """
        data = self._load()
        if session_id not in data:
            data[session_id] = {
                "active_strategy": "habitual",
                "consecutive_turns": 0,
                "last_query": "",
                "error_history": []
            }

        sess = data[session_id]
        sess["error_history"].append(prediction_error)
        sess["error_history"] = sess["error_history"][-5:]

        # Shift trigger conditions
        avg_error = sum(sess["error_history"]) / len(sess["error_history"])
        strategy_shifted = False
        prev_strategy = sess["active_strategy"]
        ne_boost = 0.0
        da_boost = 0.0

        # Heuristic rules for strategy set-shifting:
        if negative_feedback or avg_error > 0.65:
            # Shift to analytical or exploratory
            if prev_strategy == "habitual":
                sess["active_strategy"] = "analytical"
            else:
                sess["active_strategy"] = "exploratory"
            strategy_shifted = True
            ne_boost = 0.25   # High NE to focus attention on new strategy
            da_boost = 0.15   # High DA to increase exploration learning rate
        elif "solve" in query.lower() or any(char in query for char in ["+", "-", "*", "/"]):
            sess["active_strategy"] = "logical"
            if prev_strategy != "logical":
                strategy_shifted = True
        else:
            # If doing well, return back to habitual baseline gradual exploration
            if avg_error < 0.25 and prev_strategy in ["analytical", "exploratory"]:
                sess["active_strategy"] = "habitual"
                strategy_shifted = True

        if strategy_shifted:
            sess["consecutive_turns"] = 0
            print(f"[ACC/OFC] Cognitive Set Shift: {prev_strategy} -> {sess['active_strategy']}")
        else:
            sess["consecutive_turns"] += 1

        sess["last_query"] = query
        data[session_id] = sess
        self._save(data)

        return {
            "active_strategy": sess["active_strategy"],
            "strategy_shifted": strategy_shifted,
            "ne_boost": ne_boost,
            "da_boost": da_boost
        }

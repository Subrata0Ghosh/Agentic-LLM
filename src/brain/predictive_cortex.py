"""
Predictive Cortex — src/brain/predictive_cortex.py
====================================================
Brain Region: Sensory & Association Cortex (Predictive Coding)

The most important insight from modern neuroscience:
The brain does NOT wait for input and then respond.
The brain CONTINUOUSLY generates predictions about what will happen next.
It responds only to PREDICTION ERRORS — the difference between expected and actual.

Karl Friston's Free Energy Principle + Predictive Coding:
  - Brain has a hierarchical generative model (world model)
  - Predictions flow TOP-DOWN from higher to lower areas
  - Prediction errors flow BOTTOM-UP from lower to higher areas
  - The brain tries to MINIMIZE prediction error (= free energy)

Two strategies to reduce prediction error:
  1. Perception: Update internal world model to match reality
  2. Action:     Change the world to match the internal model

Curiosity (Active Inference) = Drive to seek information that will
reduce prediction error most efficiently (epistemic value).

Dopamine = prediction error signal (phasic DA release encodes surprise)
"""

import os
import json
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

PC_FILE = "data/world_model.json"


class PredictiveCortex:
    """
    Maintains a per-session world model and computes prediction errors.
    The world model is a summary of what the agent currently "believes"
    about the topics it has encountered.
    """

    def __init__(self, filepath: str = PC_FILE):
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

    def get_world_model(self, session_id: str) -> dict:
        data = self._load()
        if session_id not in data:
            data[session_id] = {
                "topic_map": {},        # known topic → brief belief summary
                "last_query": "",
                "last_prediction": "",
                "cumulative_error": 0.0,
                "turn_count": 0,
            }
            self._save(data)
        return data[session_id]

    def _save_model(self, session_id: str, model: dict):
        data = self._load()
        data[session_id] = model
        self._save(data)

    def predict(self, session_id: str, query: str, retrieved_context: str) -> dict:
        """
        Generates an internal prediction about what the answer should be,
        based on the current world model + retrieved hippocampal context.

        Returns:
            {
              "prediction":    str   — what the model predicts the answer will be
              "confidence":    float — how confident the prediction is [0,1]
              "topics_found":  list  — topics recognized from world model
            }
        """
        model = self.get_world_model(session_id)
        topic_map = model.get("topic_map", {})

        # Find known topics in the query
        query_words = set(query.lower().split())
        recognized = []
        for topic, belief in topic_map.items():
            topic_words = set(topic.lower().split())
            if query_words & topic_words:
                recognized.append((topic, belief))

        # Build prediction from known beliefs
        if recognized:
            belief_texts = [f"About '{t}': {b}" for t, b in recognized[:3]]
            prediction = " | ".join(belief_texts)
            confidence = min(1.0, 0.4 + len(recognized) * 0.15)
        elif retrieved_context:
            # Have memory but no world model entry — low confidence prediction
            prediction = "Relevant information exists in memory."
            confidence = 0.25
        else:
            prediction = "No prior knowledge on this topic."
            confidence = 0.05

        model["last_query"] = query
        model["last_prediction"] = prediction
        model["turn_count"] += 1
        self._save_model(session_id, model)

        return {
            "prediction":   prediction,
            "confidence":   round(confidence, 3),
            "topics_found": [t for t, _ in recognized],
        }

    def compute_error(self, session_id: str, prediction: dict,
                      actual_response: str, retrieved_context: str) -> dict:
        """
        Computes prediction error magnitude after a response is generated.

        Prediction error = f(confidence, context availability, response novelty)

        High error → DA spike → Curiosity → Active Inference search
        Low error  → DA dip   → Fast familiar path

        Returns:
            {
              "error_magnitude": float [0, 1]
              "should_learn":    bool — if True, store as new hippocampal memory
              "da_signal":       float — dopamine signal to emit
            }
        """
        confidence = prediction.get("confidence", 0.1)
        has_context = bool(retrieved_context.strip())

        # Base error: inverse of confidence
        base_error = 1.0 - confidence

        # Boost error if we had no memory (novel topic)
        if not has_context:
            base_error = min(1.0, base_error + 0.3)

        # Check response length as proxy for information density
        response_words = len(actual_response.split())
        if response_words > 80:
            base_error = max(0.0, base_error - 0.1)  # Rich response = less error

        error_magnitude = round(min(1.0, base_error), 3)

        # Update cumulative error tracking
        model = self.get_world_model(session_id)
        model["cumulative_error"] = round(
            model.get("cumulative_error", 0.0) * 0.9 + error_magnitude * 0.1, 3
        )
        self._save_model(session_id, model)

        # Dopamine signal: surprise = phasic DA spike; familiar = DA dip
        da_signal = error_magnitude * 0.6
        should_learn = error_magnitude > 0.4 or not has_context

        return {
            "error_magnitude": error_magnitude,
            "should_learn":    should_learn,
            "da_signal":       round(da_signal, 3),
        }

    def update_world_model(self, session_id: str, query: str, response: str,
                           source: str = "conversation"):
        """
        Updates the world model after a successful exchange.
        This is the "perception" side of active inference — updating the internal
        model to reduce future prediction error.
        """
        model = self.get_world_model(session_id)
        topic_map = model.get("topic_map", {})

        # Extract key topic from query (first 3 content words as topic key)
        words = [w for w in query.lower().split() if len(w) > 3]
        topic_key = " ".join(words[:3]) if words else query[:30].lower()

        # Store a belief snippet (first 120 chars of response)
        belief = response[:120].replace("\n", " ").strip()
        topic_map[topic_key] = belief

        # Keep topic map manageable
        if len(topic_map) > 50:
            # Drop oldest entries (simple FIFO)
            keys = list(topic_map.keys())
            for k in keys[:-50]:
                del topic_map[k]

        model["topic_map"] = topic_map
        self._save_model(session_id, model)

    def get_schema_summary(self, session_id: str) -> dict:
        """Returns a human-readable summary of the current world model."""
        model = self.get_world_model(session_id)
        return {
            "known_topics":       list(model.get("topic_map", {}).keys()),
            "cumulative_error":   model.get("cumulative_error", 0.0),
            "turn_count":         model.get("turn_count", 0),
        }

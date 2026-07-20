"""
Neuromodulators — src/brain/neuromodulators.py
===============================================
Brain Region: Brainstem nuclei (VTA, LC, Raphe, NBM)

The brain has 4 major neuromodulatory systems that act as "gain controls"
for the entire cognitive architecture. They are NOT the same thing —
each has a distinct role and decays/rises at different rates.

Dopamine (DA)     — Locus coeruleus ventral tegmental area
  → Prediction error signal. Rises on surprise, drives learning rate.
  → High DA = high temperature (exploration), faster model updates.

Serotonin (SE)    — Raphe nuclei
  → Mood, patience, social warmth. Low SE = terse. High SE = expansive.
  → Modulates amygdala reactivity (calms or amplifies emotional response).

Norepinephrine (NE) — Locus coeruleus
  → Arousal & attention gain. High NE = narrow, sharp focus.
  → Low NE = diffuse, creative, associative thinking (good for DMN).

Acetylcholine (ACh) — Nucleus basalis of Meynert
  → Learning gate. High ACh = strongly encode into hippocampus.
  → Low ACh = retrieval-dominant mode (not encoding mode).
"""

import os
import json
import time

NEUROMOD_FILE = "data/neuromodulators.json"

# Biological baseline values
BASELINE = {
    "dopamine":        0.35,   # reward / prediction-error signal
    "serotonin":       0.60,   # mood / social warmth / patience
    "norepinephrine":  0.50,   # arousal / attention gain
    "acetylcholine":   0.55,   # learning gate / encoding strength
}

# Decay rates per turn (how quickly each modulator returns to baseline)
DECAY_RATE = {
    "dopamine":        0.08,   # fast — DA is phasic (spike-and-return)
    "serotonin":       0.02,   # slow — SE is tonic (stable mood)
    "norepinephrine":  0.05,   # moderate — NE tracks alertness
    "acetylcholine":   0.04,   # moderate — ACh tracks task demand
}


class NeuromodulatorSystem:
    """
    Models the four major neuromodulatory systems of the human brain.
    Each system has an independent level that rises/decays based on
    cognitive events (novel input, prediction error, fatigue, etc.).
    """

    def __init__(self, filepath=NEUROMOD_FILE):
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

    def get_state(self, session_id: str) -> dict:
        data = self._load()
        if session_id not in data:
            data[session_id] = {**BASELINE, "prediction_error": 0.0, "turn_count": 0, "last_ts": time.time()}
            self._save(data)
        return data[session_id]

    def _save_state(self, session_id: str, state: dict):
        data = self._load()
        data[session_id] = state
        self._save(data)

    def tick(self, session_id: str) -> dict:
        """
        Called each conversational turn. Each modulator decays back toward
        its biological baseline (tonic firing rate).
        """
        state = self.get_state(session_id)
        for mod in ["dopamine", "serotonin", "norepinephrine", "acetylcholine"]:
            current = state[mod]
            baseline = BASELINE[mod]
            decay = DECAY_RATE[mod]
            # Exponential decay toward baseline
            state[mod] = current + (baseline - current) * decay
        state["turn_count"] += 1
        state["last_ts"] = time.time()
        self._save_state(session_id, state)
        return state

    def on_prediction_error(self, session_id: str, error_magnitude: float) -> dict:
        """
        High prediction error → DA spike (surprise signal).
        Also elevates NE (arousal to deal with the unexpected).
        Suppresses ACh slightly (shift from encoding to acting mode).
        """
        state = self.get_state(session_id)
        state["dopamine"]        = min(1.0, state["dopamine"] + error_magnitude * 0.6)
        state["norepinephrine"]  = min(1.0, state["norepinephrine"] + error_magnitude * 0.3)
        state["acetylcholine"]   = max(0.1, state["acetylcholine"] - error_magnitude * 0.1)
        state["prediction_error"] = error_magnitude
        self._save_state(session_id, state)
        return state

    def on_familiar_input(self, session_id: str) -> dict:
        """
        Familiar input → DA dips below baseline (reduced reward signal).
        ACh rises (retrieval-dominant, not encoding mode).
        NE drops slightly (no need for high arousal).
        """
        state = self.get_state(session_id)
        state["dopamine"]       = max(0.1, state["dopamine"] - 0.08)
        state["acetylcholine"]  = min(1.0, state["acetylcholine"] + 0.12)
        state["norepinephrine"] = max(0.2, state["norepinephrine"] - 0.05)
        state["prediction_error"] = 0.0
        self._save_state(session_id, state)
        return state

    def on_learning_event(self, session_id: str) -> dict:
        """
        New information stored → ACh spike (consolidate into hippocampus).
        Slight DA reward for successful learning.
        """
        state = self.get_state(session_id)
        state["acetylcholine"] = min(1.0, state["acetylcholine"] + 0.25)
        state["dopamine"]      = min(1.0, state["dopamine"] + 0.10)
        self._save_state(session_id, state)
        return state

    def on_social_interaction(self, session_id: str, positive: bool) -> dict:
        """
        Positive interaction → SE rises (social reward).
        Negative → SE dips (social stress).
        """
        state = self.get_state(session_id)
        delta = 0.10 if positive else -0.12
        state["serotonin"] = max(0.05, min(1.0, state["serotonin"] + delta))
        self._save_state(session_id, state)
        return state

    def sleep_reset(self, session_id: str) -> dict:
        """Sleep fully restores all neuromodulators to biological baseline."""
        state = self.get_state(session_id)
        state.update(BASELINE)
        state["prediction_error"] = 0.0
        state["last_ts"] = time.time()
        self._save_state(session_id, state)
        return state

    def update_state_custom(self, session_id: str, updates: dict) -> dict:
        """Update specific neuromodulators directly (e.g. from feedback events)."""
        state = self.get_state(session_id)
        for k, v in updates.items():
            if k in BASELINE:
                state[k] = max(0.01, min(1.0, float(v)))
        state["last_ts"] = time.time()
        self._save_state(session_id, state)
        return state

    def get_generation_params(self, session_id: str) -> dict:
        """
        Computes dynamic generation parameters from all 4 neuromodulators.

        DA  → temperature (exploration vs exploitation)
        SE  → response warmth (system prompt tone flag)
        NE  → top_p (narrow sharp vs broad associative)
        ACh → repetition_penalty (high ACh = precise, low = more wandering)
        """
        s = self.get_state(session_id)
        da  = s["dopamine"]
        se  = s["serotonin"]
        ne  = s["norepinephrine"]
        ach = s["acetylcholine"]

        # Temperature: DA-driven (0.3 baseline, exploratory when curious)
        temperature = 0.3 + (da - BASELINE["dopamine"]) * 0.9
        temperature = max(0.15, min(1.0, temperature))

        # top_p: NE-driven (high arousal = narrow focus, low = broad)
        top_p = 1.0 - (ne - 0.2) * 0.4
        top_p = max(0.6, min(0.98, top_p))

        # repetition_penalty: ACh-driven (high ACh = strict, low = loose)
        rep_penalty = 1.05 + (ach - BASELINE["acetylcholine"]) * 0.3
        rep_penalty = max(1.0, min(1.4, rep_penalty))

        # Tone flag for system prompt
        tone = "warm and expansive" if se > 0.7 else ("brief and precise" if se < 0.35 else "friendly and clear")

        return {
            "temperature":        round(temperature, 3),
            "top_p":              round(top_p, 3),
            "repetition_penalty": round(rep_penalty, 3),
            "tone":               tone,
            "raw":                {k: round(v, 3) for k, v in s.items() if k in BASELINE},
        }

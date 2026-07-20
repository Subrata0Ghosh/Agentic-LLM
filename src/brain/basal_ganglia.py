"""
Basal Ganglia — src/brain/basal_ganglia.py
===========================================
Brain Region: Basal Ganglia
  — Striatum (Caudate + Putamen), Globus Pallidus, Substantia Nigra, STN

The basal ganglia implement the brain's ACTION SELECTION system.
They work through a competing Go / No-Go mechanism:

  Go pathway   (direct)  : Dopamine D1 receptors → PROMOTE an action
  No-Go pathway (indirect): Dopamine D2 receptors → SUPPRESS an action

Only the action with the strongest Go signal wins and is executed.
Everything else is suppressed. This is how the brain "decides" without
explicitly considering every option — most options are simply gated off.

In the cognitive AI pipeline, basal ganglia map to:
  1. Selecting the BEST candidate response path (HABIT/REFLEX/DEEP/FOCUS)
  2. Suppressing harmful, repetitive, or off-topic content
  3. Go-No-Go gate on final output (should we answer or clarify?)
  4. RL-style action value update: actions that led to reward → stronger Go
  5. Habit loop control: BG gates which habits get promoted to motor cortex

Biological basis:
  - Loss of dopamine in BG → Parkinson's (movement suppressed)
  - Excess dopamine in BG → tics, compulsions (suppression fails)
  - BG integrates DA signals over time (slow) vs cortex (fast)
  - STN provides a "hyperdirect" stop signal (like an emergency brake)

Neuromodulation:
  - Dopamine is the primary modulator (D1/D2 balance = Go/No-Go balance)
  - Serotonin influences the No-Go pathway (high SE = more patient suppression)
  - NE from LC can transiently override BG suppression in emergencies
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple

BASAL_GANGLIA_FILE = "data/basal_ganglia.json"

# Suppression patterns — content types that should be No-Go gated
SUPPRESSION_PATTERNS = [
    ("violence_explicit",   lambda t: any(w in t for w in ["kill", "murder", "torture", "harm someone"])),
    ("self_harm",           lambda t: any(w in t for w in ["hurt myself", "end my life", "suicide method"])),
    ("repetition_trap",     lambda t: False),   # Set dynamically by perseveration check
    ("deceptive_claim",     lambda t: any(w in t for w in ["i am human", "i am not an ai", "i am a real person"])),
]

# Path priority weights — higher = more likely to be selected
PATH_WEIGHTS = {
    "HABIT":          1.0,   # Cached, fastest
    "REFLEX":         0.85,  # Pattern-matched greeting
    "MATHEMATICAL":   0.95,  # Parietal-solved, high confidence
    "CLARIFICATION":  0.70,  # Needs user input
    "DEEP":           0.75,  # Standard reasoning
    "FOCUS":          0.80,  # High-salience, full resources
}


class BasalGanglia:
    """
    Action selection system via Go/No-Go gating.
    Models the dopaminergic competition between response pathways.
    """

    def __init__(self, filepath: str = BASAL_GANGLIA_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # In-memory action value table (RL Q-values per session per path)
        self._q_values: Dict[str, Dict[str, float]] = {}

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _get_q(self, session_id: str, path: str) -> float:
        """Get Q-value for a path in this session."""
        return self._q_values.get(session_id, {}).get(path, PATH_WEIGHTS.get(path, 0.7))

    # ══════════════════════════════════════════════════════════════════════════
    # 1. GO / NO-GO SUPPRESSION CHECK
    # ══════════════════════════════════════════════════════════════════════════
    def check_go_nogo(self, response_text: str, is_perseverating: bool = False) -> Dict:
        """
        Applies No-Go gate to the candidate response.
        If a suppression pattern fires: No-Go signal → block or redirect.

        Analogous to: Globus Pallidus Internus suppressing thalamic relay
        when the No-Go pathway wins.

        Returns:
            {
              "go":              bool  — True: allow, False: suppress
              "reason":          str or None
              "suppressed_type": str or None
            }
        """
        text_l = response_text.lower()

        for pattern_name, check_fn in SUPPRESSION_PATTERNS:
            if pattern_name == "repetition_trap":
                if is_perseverating:
                    return {
                        "go": False,
                        "reason": "No-Go: Perseveration detected. Suppressing repeated response pattern.",
                        "suppressed_type": "repetition_trap",
                    }
                continue

            if check_fn(text_l):
                return {
                    "go": False,
                    "reason": f"No-Go: Content suppressed by basal ganglia ({pattern_name}).",
                    "suppressed_type": pattern_name,
                }

        return {"go": True, "reason": None, "suppressed_type": None}

    # ══════════════════════════════════════════════════════════════════════════
    # 2. ACTION SELECTION (Path Gating)
    # ══════════════════════════════════════════════════════════════════════════
    def gate_path(self, session_id: str, candidate_paths: List[str],
                  dopamine: float, salience: float) -> Dict:
        """
        Selects the best processing path using Go pathway competition.
        DA level modulates how strongly the striatum promotes exploration
        (high DA = more weight on novel, high-salience paths).

        Args:
            candidate_paths: List of possible paths (e.g. ["REFLEX", "DEEP", "FOCUS"])
            dopamine:        Current DA level [0, 1]
            salience:        Thalamic salience score [0, 1]

        Returns:
            {
              "selected_path": str,
              "gate_scores":   dict,
              "da_influence":  float
            }
        """
        if not candidate_paths:
            return {"selected_path": "DEEP", "gate_scores": {}, "da_influence": dopamine}

        # DA modulates Go strength: high DA boosts exploratory paths
        da_bias = (dopamine - 0.35) * 0.5  # delta from baseline

        scores = {}
        for path in candidate_paths:
            q = self._get_q(session_id, path)
            # High salience favors DEEP/FOCUS; high DA favors exploration
            salience_bonus = 0.0
            if path in ("DEEP", "FOCUS") and salience > 0.5:
                salience_bonus = (salience - 0.5) * 0.4
            da_bonus = da_bias if path == "FOCUS" else 0.0
            scores[path] = round(q + salience_bonus + da_bonus, 3)

        selected = max(scores, key=scores.__getitem__)
        return {
            "selected_path": selected,
            "gate_scores":   scores,
            "da_influence":  round(da_bias, 3),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ACTION VALUE UPDATE (RL Signal)
    # ══════════════════════════════════════════════════════════════════════════
    def update_action_value(self, session_id: str, path: str,
                             reward: float, alpha: float = 0.15) -> float:
        """
        Temporal-difference (TD) update of Q-value for the chosen path.
        Reward signal comes from the RewardSystem (👍=+1, 👎=-1) or
        prediction error magnitude.

        Q(s,a) ← Q(s,a) + α × (reward − Q(s,a))

        Returns updated Q-value.
        """
        if session_id not in self._q_values:
            self._q_values[session_id] = {}

        current_q = self._get_q(session_id, path)
        updated_q = current_q + alpha * (reward - current_q)
        updated_q = max(0.1, min(1.0, updated_q))
        self._q_values[session_id][path] = round(updated_q, 3)

        # Persist
        data = self._load()
        if session_id not in data:
            data[session_id] = {"q_values": {}, "go_nogo_log": [], "total_gates": 0}
        data[session_id]["q_values"] = self._q_values[session_id]
        self._save(data)

        return updated_q

    # ══════════════════════════════════════════════════════════════════════════
    # 4. HYPERDIRECT STOP (Emergency Brake — STN analog)
    # ══════════════════════════════════════════════════════════════════════════
    def emergency_stop(self, query: str) -> Optional[str]:
        """
        The subthalamic nucleus (STN) provides a fast, broad inhibition
        of all ongoing action selection when a novel/dangerous input arrives.

        Returns a redirect message if emergency stop fires, else None.
        """
        q = query.lower()
        emergency_keywords = [
            "i am in danger", "call 911", "i want to die", "medical emergency",
            "heart attack", "stroke", "overdose", "fire alarm", "help me now"
        ]
        if any(kw in q for kw in emergency_keywords):
            return (
                "I hear that this is urgent. If you are in immediate danger, "
                "please call emergency services (911 or your local emergency number) right away. "
                "I'm here to support you."
            )
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # 5. RECORD GATE EVENT
    # ══════════════════════════════════════════════════════════════════════════
    def record_gate(self, session_id: str, selected_path: str,
                    went_go: bool, suppression_type: str = None):
        """Logs the gate decision for the stats endpoint."""
        data = self._load()
        if session_id not in data:
            data[session_id] = {"q_values": {}, "go_nogo_log": [], "total_gates": 0}

        data[session_id]["total_gates"] = data[session_id].get("total_gates", 0) + 1
        log = data[session_id].get("go_nogo_log", [])
        log.append({
            "ts":              time.time(),
            "path":            selected_path,
            "go":              went_go,
            "suppressed_type": suppression_type,
        })
        data[session_id]["go_nogo_log"] = log[-50:]  # Keep last 50
        self._save(data)

    # ══════════════════════════════════════════════════════════════════════════
    # STATS
    # ══════════════════════════════════════════════════════════════════════════
    def get_stats(self, session_id: str) -> dict:
        """Returns basal ganglia stats for a session."""
        data = self._load()
        sess = data.get(session_id, {"q_values": {}, "go_nogo_log": [], "total_gates": 0})
        q_vals = self._q_values.get(session_id, sess.get("q_values", {}))

        # Merge persisted Q-values with in-memory
        merged_q = {**PATH_WEIGHTS, **sess.get("q_values", {}), **q_vals}

        go_count  = sum(1 for e in sess.get("go_nogo_log", []) if e.get("go"))
        nogo_count = sum(1 for e in sess.get("go_nogo_log", []) if not e.get("go"))

        return {
            "total_gates":    sess.get("total_gates", 0),
            "go_count":       go_count,
            "nogo_count":     nogo_count,
            "action_values":  {k: round(v, 3) for k, v in merged_q.items()},
            "recent_log":     sess.get("go_nogo_log", [])[-10:],
        }

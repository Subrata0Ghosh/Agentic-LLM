import os
import json

BRAIN_STATE_FILE = "data/brain_state.json"

class BrainState:
    def __init__(self, filepath=BRAIN_STATE_FILE):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({}, f)

    def load_all(self):
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_all(self, data):
        try:
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving brain state: {e}")

    def get_state(self, session_id):
        all_data = self.load_all()
        if session_id not in all_data:
            # Default biological state
            all_data[session_id] = {
                "dopamine": 0.3,    # Curiosity / Novelty / Exploration drive (0.1 to 1.0)
                "fatigue": 0.0,     # Sleep debt / Energy expenditure (0.0 to 1.0)
                "attention": 0.8,   # Gating focus level (0.0 to 1.0)
                "update_count": 0
            }
            self.save_all(all_data)
        return all_data[session_id]

    def save_state(self, session_id, state):
        all_data = self.load_all()
        all_data[session_id] = state
        self.save_all(all_data)

    def update_on_query(self, session_id, query_text, has_good_match):
        """
        Updates neuromodulators based on the incoming query and retrieval status.
        """
        state = self.get_state(session_id)
        
        # Determine attention based on query length/complexity
        words = len(query_text.split())
        if words > 15:
            # Complex query demands high cognitive focus
            state["attention"] = min(1.0, state["attention"] + 0.1)
        elif words < 4:
            # Short query / low focus required
            state["attention"] = max(0.4, state["attention"] - 0.1)
            
        # Determine dopamine (curiosity) based on memory familiarity
        if not has_good_match:
            # Unfamiliar topic triggers high curiosity (dopamine spike) to learn new things
            state["dopamine"] = min(1.0, state["dopamine"] + 0.3)
        else:
            # Familiar topic keeps curiosity in a comfortable state
            state["dopamine"] = max(0.3, state["dopamine"] - 0.05)
            
        state["update_count"] += 1
        self.save_state(session_id, state)
        return state

    def update_on_generation(self, session_id, tokens_generated):
        """
        Increases fatigue based on cognitive load (number of generated tokens).
        """
        state = self.get_state(session_id)
        # Brain gets tired from thinking and generating output
        fatigue_increase = tokens_generated * 0.0015
        state["fatigue"] = min(1.0, state["fatigue"] + fatigue_increase)
        
        # If very fatigued, attention drops
        if state["fatigue"] > 0.7:
            state["attention"] = max(0.3, state["attention"] - (state["fatigue"] * 0.1))
            
        self.save_state(session_id, state)
        return state

    def sleep_reset(self, session_id):
        """
        Simulates sleep: consolidates memories, resets fatigue, and replenishes attention/dopamine.
        """
        state = self.get_state(session_id)
        state["fatigue"] = 0.0
        state["dopamine"] = 0.3  # Reset to healthy baseline
        state["attention"] = 0.9  # Fully refreshed focus
        self.save_state(session_id, state)
        return state

    def get_generation_params(self, session_id):
        """
        Computes dynamic generation parameters based on biological metrics.
        - High dopamine (curiosity) increases exploration (higher temperature/top_p).
        - High fatigue reduces consistency (higher temperature, lower repetition penalty, or vice-versa).
          To simulate brain fog, we can slightly increase temperature and reduce repetition penalty,
          making responses slightly more wandering or repetitive if very tired.
        """
        state = self.get_state(session_id)
        dopamine = state["dopamine"]
        fatigue = state["fatigue"]
        
        # Baseline parameters
        temp = 0.5
        top_p = 0.9
        rep_penalty = 1.12
        
        # Adjustments based on state
        # Dopamine (exploration) pushes temperature up slightly
        temp += (dopamine - 0.3) * 0.3
        
        # Fatigue (brain fog) introduces cognitive slip
        if fatigue > 0.6:
            temp += (fatigue - 0.6) * 0.2  # Slightly more chaotic
            rep_penalty = max(1.02, rep_penalty - (fatigue - 0.6) * 0.15) # Less cognitive control
            
        # Bound variables to sensible ranges
        temp = max(0.2, min(1.0, temp))
        rep_penalty = max(1.0, min(1.3, rep_penalty))
        
        return {
            "temperature": temp,
            "top_p": top_p,
            "repetition_penalty": rep_penalty
        }

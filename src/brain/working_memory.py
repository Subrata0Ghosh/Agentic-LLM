"""
Working Memory — src/brain/working_memory.py
=============================================
Brain Region: Prefrontal Cortex (PFC) — dorsolateral and ventrolateral regions

The human working memory (WM) is capacity-limited to ~4 independent "chunks"
(Cowan's magic number 4). It is NOT a simple replay of chat history.

The PFC actively:
  1. Maintains a limited window of the most relevant context chunks.
  2. Chunks related information into single units (reduces cognitive load).
  3. Inhibits irrelevant information (executive inhibition).
  4. Assembles context on-demand rather than blindly concatenating everything.

Norepinephrine (NE) modulates WM capacity:
  - High NE → sharper focus, but narrower WM (fewer items held clearly)
  - Low NE  → broader WM, more associative items held loosely

Dopamine (DA) modulates WM updating:
  - High DA → WM updates rapidly (new info replaces old)
  - Low DA  → WM is stable and resistant to updating (perseveration)
"""

import os
import json
import time
from typing import List, Dict, Optional

WM_FILE = "data/working_memory.json"
WM_CAPACITY = 4   # The biological limit: ~4 chunks


class WorkingMemory:
    """
    Capacity-limited, chunked working memory for the PFC.
    Stores only the most recently relevant conversational chunks.
    """

    def __init__(self, filepath: str = WM_FILE, capacity: int = WM_CAPACITY):
        self.filepath = filepath
        self.capacity = capacity
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

    def get_chunks(self, session_id: str) -> List[dict]:
        data = self._load()
        return data.get(session_id, {}).get("chunks", [])

    def get_focus(self, session_id: str) -> Optional[str]:
        """Returns the current focal topic being held in WM."""
        data = self._load()
        return data.get(session_id, {}).get("focus_topic", None)

    def add_turn(self, session_id: str, role: str, content: str,
                 emotion_tag: dict = None, dopamine: float = 0.35):
        """
        Adds a new conversational turn as a chunk. If over capacity,
        evicts the oldest chunk (LRU — Least Recently Used).

        High dopamine → WM readily accepts new chunks (fast updating).
        Low dopamine  → New chunks need higher salience to enter WM.
        """
        data = self._load()
        if session_id not in data:
            data[session_id] = {"chunks": [], "focus_topic": None, "last_updated": time.time()}

        chunk = {
            "role":       role,
            "content":    content[:500],  # Truncate very long turns
            "ts":         time.time(),
            "emotion":    emotion_tag or {"valence": "neutral", "priority": 0.1},
            "relevance":  1.0,  # Newest is most relevant by default
        }

        chunks = data[session_id]["chunks"]
        chunks.append(chunk)

        # DA-modulated capacity: high DA = slightly more space for new info
        effective_capacity = self.capacity + (1 if dopamine > 0.7 else 0)

        # If over capacity, drop the oldest low-priority chunk
        if len(chunks) > effective_capacity:
            # Sort by priority*recency — keep emotionally important ones
            chunks.sort(key=lambda c: (c["emotion"].get("priority", 0) * 0.4 +
                                       (time.time() - c["ts"]) * -0.001))
            chunks = chunks[-effective_capacity:]

        data[session_id]["chunks"] = chunks
        data[session_id]["last_updated"] = time.time()
        self._save(data)

    def update_focus(self, session_id: str, topic: str):
        """Update the currently held focus topic."""
        data = self._load()
        if session_id not in data:
            data[session_id] = {"chunks": [], "focus_topic": None}
        data[session_id]["focus_topic"] = topic
        self._save(data)

    def extract_and_store_variables(self, session_id: str, text: str):
        """
        Parses text for simple explicit memory slots (like color, number, name)
        and saves them under the 'variables' dictionary for persistent access.
        """
        import re
        data = self._load()
        if session_id not in data:
            data[session_id] = {
                "chunks": [],
                "focus_topic": None,
                "variables": {},
                "last_updated": time.time()
            }
        if "variables" not in data[session_id]:
            data[session_id]["variables"] = {}

        # 1. Pattern: "remember that [key] is [value]" or "remember [key] is [value]"
        m1 = re.search(r"\bremember\s+(?:that\s+)?(?:my\s+)?([a-z0-9_\s]{2,20})\s+is\s+([a-z0-9_\s]{1,50})", text, re.IGNORECASE)
        if m1:
            k, v = m1.groups()
            data[session_id]["variables"][k.strip().lower().replace(" ", "_")] = v.strip()

        # 2. Pattern: "my favorite [key] is [value]"
        m2 = re.search(r"\bmy\s+favorite\s+([a-z0-9_\s]{2,20})\s+is\s+([a-z0-9_\s]{1,50})", text, re.IGNORECASE)
        if m2:
            k, v = m2.groups()
            data[session_id]["variables"][k.strip().lower().replace(" ", "_")] = v.strip()

        # 3. Pattern: "remember the number:? [value]" or "remember this number:? [value]"
        m3 = re.search(r"\bremember\s+(?:the|this)\s+number\s*:?\s*(\d+)", text, re.IGNORECASE)
        if m3:
            val = m3.group(1)
            data[session_id]["variables"]["number_to_remember"] = val

        # 4. Pattern: "remember these words:? [word list]"
        m4 = re.search(r"\bremember\s+(?:these\s+)?words\s*:?\s*([a-z0-9_,\s\n\-\.\'\"]{2,120})", text, re.IGNORECASE)
        if m4:
            val = m4.group(1).replace("\n", ", ").strip()
            data[session_id]["variables"]["words_to_remember"] = val

        data[session_id]["last_updated"] = time.time()
        self._save(data)

    def assemble_context(self, session_id: str, norepinephrine: float = 0.5) -> List[dict]:
        """
        Assembles the working memory context for the language model.
        NE modulates how much context is assembled:
          - High NE: Returns only the most recent, high-priority chunks.
          - Low NE:  Returns all available chunks (broader, associative).

        Returns a list of {"role": ..., "content": ...} dicts.
        """
        data = self._load()
        chunks = data.get(session_id, {}).get("chunks", [])
        if not chunks:
            selected_msgs = []
        else:
            # NE determines how much of WM is "in focus"
            # High NE = sharp narrow focus (fewer chunks, higher relevance required)
            ne_cutoff = int(self.capacity * (1.2 - norepinephrine * 0.6))
            ne_cutoff = max(1, min(len(chunks), ne_cutoff))
            selected = chunks[-ne_cutoff:]
            selected_msgs = [
                {"role": c["role"], "content": c["content"]}
                for c in selected
            ]

        # Inject persistent variable slots into the working memory context if any exist
        variables = data.get(session_id, {}).get("variables", {})
        if variables:
            slots_str = "### ACTIVE WORKING MEMORY SLOTS ###\n" + "\n".join(f"- {k}: {v}" for k, v in variables.items())
            if selected_msgs:
                selected_msgs[0]["content"] = slots_str + "\n\n" + selected_msgs[0]["content"]
            else:
                selected_msgs.append({"role": "system", "content": slots_str})

        return selected_msgs

    def inhibit_irrelevant(self, session_id: str, current_query: str):
        """
        Executive inhibition: marks chunks with low topical overlap as lower priority.
        This mimics the PFC's ability to suppress distracting information.
        """
        data = self._load()
        if session_id not in data:
            return

        query_words = set(current_query.lower().split())
        chunks = data[session_id].get("chunks", [])

        for chunk in chunks:
            content_words = set(chunk["content"].lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            # Decay relevance of chunks with little topical overlap
            chunk["relevance"] = chunk.get("relevance", 1.0) * (0.5 + overlap * 0.5)

        data[session_id]["chunks"] = chunks
        self._save(data)

    def clear(self, session_id: str):
        """Fully clear WM for a session (happens after sleep consolidation)."""
        data = self._load()
        data[session_id] = {
            "chunks": [],
            "focus_topic": None,
            "active_goal": None,
            "execution_plan": None,
            "last_updated": time.time()
        }
        self._save(data)

    def set_goal(self, session_id: str, goal: str, plan: list = None):
        """Set the active goal and plan structure in Working Memory."""
        data = self._load()
        if session_id not in data:
            data[session_id] = {"chunks": [], "focus_topic": None}
        data[session_id]["active_goal"] = goal
        data[session_id]["execution_plan"] = plan
        data[session_id]["last_updated"] = time.time()
        self._save(data)

    def get_goal(self, session_id: str) -> dict:
        """Retrieve the active goal and plan from Working Memory."""
        data = self._load()
        sess = data.get(session_id, {})
        return {
            "active_goal": sess.get("active_goal"),
            "execution_plan": sess.get("execution_plan")
        }

    def spatial_scratchpad(self, session_id: str, items: list) -> dict:
        """
        The visuospatial sketchpad of working memory (Baddeley's model).
        Stores a list of spatial/positional items for spatial reasoning tasks.
        e.g., grid positions, room layouts, relative directions.

        Returns the current scratchpad state.
        """
        data = self._load()
        if session_id not in data:
            data[session_id] = {"chunks": [], "focus_topic": None}
        data[session_id]["spatial_scratchpad"] = items[:16]  # Limit to 16 spatial items
        data[session_id]["last_updated"] = time.time()
        self._save(data)
        return {"scratchpad": data[session_id]["spatial_scratchpad"]}

    def get_spatial_scratchpad(self, session_id: str) -> list:
        """Retrieves spatial scratchpad contents."""
        data = self._load()
        return data.get(session_id, {}).get("spatial_scratchpad", [])

    def load_bica_attention(self, session_id: str, focused_concepts: list):
        """
        Loads BICA Module 2 (Attention) output into working memory.
        Marks the top concepts as high-priority for NE-modulated filtering.
        This links the BICA attention module directly to the WM inhibition system.
        """
        data = self._load()
        if session_id not in data:
            data[session_id] = {"chunks": [], "focus_topic": None}
        data[session_id]["bica_focal_concepts"] = focused_concepts[:5]
        data[session_id]["last_updated"] = time.time()
        self._save(data)

    def get_bica_focal_concepts(self, session_id: str) -> list:
        """Returns the current BICA-attention focal concepts."""
        data = self._load()
        return data.get(session_id, {}).get("bica_focal_concepts", [])


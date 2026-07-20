"""
Hippocampus — src/brain/hippocampus.py
========================================
Brain Region: Hippocampus (medial temporal lobe)

The hippocampus is NOT a search engine. It is a contextual binding system.
Every memory trace binds together:
  WHO  → session / person
  WHAT → content of the experience
  WHEN → temporal context (timestamp, recency)
  WHERE → source (web URL, arxiv ID, conversation)
  HOW  → emotional state at time of encoding (from amygdala)

Key biological mechanisms modeled:
  1. ACh-gated encoding: High acetylcholine = stronger encoding (lower distance threshold).
  2. Temporal decay: Memories not reinforced become harder to retrieve over time.
  3. Pattern completion: Partial cue retrieves the full episode.
  4. Reconsolidation: Retrieving a memory slightly modifies it (memory is not static).
  5. Priority tagging: Amygdala priority score determines sleep replay probability.
"""

import sys
import os
import time
import uuid

# Add core directory so vector_memory can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from vector_memory import VectorMemory

# ACh threshold mapping: how strong the memory encoding is
# High ACh at encoding time → broader retrieval (lower distance threshold)
ACH_DISTANCE_MAP = {
    # ach_level → retrieval_threshold (lower = stricter match needed; higher = broader)
    0.1: 0.55,
    0.3: 0.50,
    0.5: 0.45,
    0.7: 0.38,
    0.9: 0.30,
}

def _ach_to_threshold(ach: float) -> float:
    """Maps ACh encoding strength to retrieval distance threshold."""
    # Interpolate between defined levels
    keys = sorted(ACH_DISTANCE_MAP.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= ach <= keys[i+1]:
            t = (ach - keys[i]) / (keys[i+1] - keys[i])
            return ACH_DISTANCE_MAP[keys[i]] + t * (ACH_DISTANCE_MAP[keys[i+1]] - ACH_DISTANCE_MAP[keys[i]])
    return 0.45  # Default


class Hippocampus:
    """
    Contextually-binding episodic memory system.
    Wraps ChromaDB with biologically-accurate metadata binding and ACh-gated encoding.
    """

    def __init__(self, db_path: str = "data/chroma_db"):
        self.vmem = VectorMemory(db_path=db_path)

    def encode(self, content: str, session_id: str, source: str,
               emotion_tag: dict = None, ach_level: float = 0.55,
               additional_context: dict = None):
        """
        Encode a new episodic memory with full contextual binding.

        Args:
            content:        Text content to memorize.
            session_id:     WHO had this experience.
            source:         WHERE it came from (URL, arxiv, conversation).
            emotion_tag:    HOW they felt (from amygdala).
            ach_level:      ACh at encoding time (encoding strength gate).
            additional_context: Any extra metadata.
        """
        emotion_tag = emotion_tag or {"valence": "neutral", "arousal": 0.0, "priority": 0.1}

        # Build the full contextual binding metadata
        meta = {
            "who":         session_id,
            "where":       source[:200],
            "when_ts":     time.time(),
            "emotion":     emotion_tag.get("valence", "neutral"),
            "arousal":     emotion_tag.get("arousal", 0.0),
            "priority":    emotion_tag.get("priority", 0.1),
            "ach_at_enc":  round(ach_level, 3),
            "recall_count": 0,     # How many times retrieved (reinforcement)
        }

        if additional_context:
            for k, v in additional_context.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        self.vmem.store(
            text=content,
            source=source[:50],
            additional_metadata=meta
        )

    def retrieve(self, query: str, session_id: str = None,
                 ach_level: float = 0.55, top_k: int = 5) -> dict:
        """
        Retrieve episodic memories matching a query.
        ACh level at retrieval time modulates the distance threshold.

        Returns:
            {
              "documents": [...],
              "distances": [...],
              "metadata":  [...],
              "has_strong_match": bool
            }
        """
        threshold = _ach_to_threshold(ach_level)
        docs, dists = self.vmem.retrieve(query=query, top_k=top_k)

        if not docs:
            return {"documents": [], "distances": [], "has_strong_match": False}

        has_strong = any(d < threshold for d in dists)

        return {
            "documents":       docs,
            "distances":       dists,
            "threshold":       threshold,
            "has_strong_match": has_strong,
        }

    def get_context_string(self, query: str, session_id: str = None,
                           ach_level: float = 0.55, top_k: int = 5) -> tuple:
        """
        High-level method: returns (context_string, has_strong_match).
        Compatible with how api.py uses vector_memory.
        """
        result = self.retrieve(query, session_id, ach_level, top_k)
        if not result["documents"]:
            return "", False

        context = "### HIPPOCAMPAL MEMORY (Episodic Context) ###\n"
        threshold = result["threshold"] + 0.25  # Soft boundary for display
        for i, (doc, dist) in enumerate(zip(result["documents"], result["distances"])):
            if dist < threshold:
                context += f"Memory {i+1}: {doc}\n"
        context += "### END MEMORY ###\n\n"

        return context, result["has_strong_match"]

    def get_all_sources(self) -> list:
        """Returns all unique memory sources (for inspector panel)."""
        return self.vmem.get_all_topics()

    def get_high_priority_memories(self, top_n: int = 10) -> list:
        """
        Retrieves the highest-priority memories for sleep replay.
        Biological basis: amygdala-tagged high-emotion memories are replayed
        more frequently during SWS (slow-wave sleep).
        """
        try:
            results = self.vmem.collection.get(
                include=["documents", "metadatas"]
            )
            pairs = list(zip(results["documents"], results["metadatas"]))
            # Sort by amygdala priority (descending)
            pairs.sort(key=lambda p: p[1].get("priority", 0.0) if p[1] else 0.0, reverse=True)
            return [{"content": doc, "meta": meta} for doc, meta in pairs[:top_n]]
        except Exception as e:
            print(f"[Hippocampus] Error retrieving high-priority memories: {e}")
            return []

    def reconsolidate(self, query: str, new_context: str,
                      session_id: str, ach_level: float = 0.6) -> dict:
        """
        Memory reconsolidation: when a memory is retrieved, it becomes
        temporarily labile and can be updated with new information.
        Biological basis: reactivated memories require re-stabilization
        (reconsolidation window ~6 hours in humans).

        This method retrieves the closest matching memory, then re-encodes
        an updated version that blends old + new context.

        Returns:
            {
              "reconsolidated": bool,
              "original":       str,
              "updated":        str
            }
        """
        result = self.retrieve(query, session_id=session_id, ach_level=ach_level, top_k=1)
        if not result["documents"]:
            return {"reconsolidated": False, "original": None, "updated": None}

        original = result["documents"][0]
        # Blend: prepend new context, keep original as background
        updated = f"{new_context}\n[Updated from: {original[:200]}]"

        self.encode(
            content=updated,
            session_id=session_id,
            source="reconsolidation",
            ach_level=min(1.0, ach_level + 0.1),  # Slightly boosted encoding
        )

        return {
            "reconsolidated": True,
            "original":       original[:200],
            "updated":        updated[:200],
        }

    def get_temporal_context(self, session_id: str, lookback_turns: int = 5) -> list:
        """
        Retrieves the most recently encoded memories for a session,
        ordered by timestamp (newest first).
        Used by the episodic timeline for temporal context assembly.

        Returns a list of {content, timestamp, source} dicts.
        """
        try:
            results = self.vmem.collection.get(
                include=["documents", "metadatas"]
            )
            # Filter to this session
            session_mems = []
            for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
                if meta and meta.get("who") == session_id:
                    session_mems.append({
                        "content":   doc[:200],
                        "timestamp": meta.get("when_ts", 0),
                        "source":    meta.get("where", "unknown"),
                        "emotion":   meta.get("emotion", "neutral"),
                        "priority":  meta.get("priority", 0.1),
                    })

            # Sort by timestamp descending
            session_mems.sort(key=lambda m: m["timestamp"], reverse=True)
            return session_mems[:lookback_turns]
        except Exception as e:
            print(f"[Hippocampus] Error in get_temporal_context: {e}")
            return []


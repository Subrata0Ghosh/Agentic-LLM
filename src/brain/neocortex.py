"""
Neocortex — src/brain/neocortex.py
=====================================
Brain Region: Neocortex (cerebral cortex — all 6 layers)

The neocortex is the seat of structured long-term knowledge.
Unlike the hippocampus (episodic, "what happened"), the neocortex
stores SCHEMAS — abstract, generalized patterns and relationships.

Key biological properties:
  1. Schema integration: New information is integrated into existing
     conceptual frameworks (schemas), not stored as isolated facts.
  2. Two-stage consolidation: Memories must pass through hippocampus
     FIRST, then gradually transfer to neocortex via sleep.
  3. Reinforcement gate: Only reinforced (repeated) memories are
     neocortically consolidated (weak memories are pruned).
  4. Representational change: Neocortical storage is MORE ABSTRACT
     than the original episode (loses contextual details, keeps essence).

In this system, the neocortex is represented by:
  a. A JSON schema graph (semantic knowledge map)
  b. The weights of the custom PyTorch Transformer (structural patterns)
"""

import os
import json
import time

SCHEMA_FILE = "data/neocortex_schema.json"


class Neocortex:
    """
    Neocortical schema manager.
    Maintains a growing conceptual knowledge graph built from
    sleep-consolidated memories and repeated experiences.
    """

    def __init__(self, filepath: str = SCHEMA_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({
                    "concepts": {},         # concept_name → definition/summary
                    "relations": [],        # [{source, relation, target}]
                    "last_updated": time.time(),
                    "consolidation_count": 0,
                }, f, indent=2)

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {"concepts": {}, "relations": [], "consolidation_count": 0}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def integrate(self, concept: str, definition: str,
                   related_concepts: list = None, source: str = "sleep"):
        """
        Integrate a new concept into the neocortical schema.
        If concept already exists, merges/updates the definition.

        Args:
            concept:          The concept name/key.
            definition:       Abstract summary of the concept.
            related_concepts: List of related concept names (for graph edges).
            source:           How this knowledge was acquired.
        """
        data = self._load()
        concept_key = concept.lower().strip()[:50]

        if concept_key in data["concepts"]:
            # Merge: append new information
            existing = data["concepts"][concept_key]
            combined = f"{existing} | {definition}"
            data["concepts"][concept_key] = combined[:300]  # Truncate to schema size
        else:
            data["concepts"][concept_key] = definition[:300]

        # Add relational edges
        if related_concepts:
            for related in related_concepts:
                rel = {"source": concept_key,
                       "relation": "related_to",
                       "target": related.lower()[:50]}
                if rel not in data["relations"]:
                    data["relations"].append(rel)

        data["last_updated"] = time.time()
        data["consolidation_count"] += 1

        # Cap graph size
        if len(data["concepts"]) > 300:
            # Remove least recently referenced (heuristic: remove first-added)
            keys = list(data["concepts"].keys())
            for k in keys[:50]:
                del data["concepts"][k]

        if len(data["relations"]) > 500:
            data["relations"] = data["relations"][-400:]

        self._save(data)

    def recall(self, query: str, top_k: int = 5) -> list:
        """
        Query the neocortical schema for concepts related to a query.
        Returns list of (concept, definition) tuples.
        """
        data = self._load()
        query_words = set(query.lower().split())
        results = []

        for concept, definition in data["concepts"].items():
            concept_words = set(concept.split())
            def_words     = set(definition.lower().split())
            overlap = len(query_words & (concept_words | def_words))
            if overlap > 0:
                results.append((concept, definition, overlap))

        results.sort(key=lambda x: x[2], reverse=True)
        return [(c, d) for c, d, _ in results[:top_k]]

    def bulk_integrate_from_sleep(self, sleep_content: str):
        """
        Bulk-integrate knowledge from a sleep consolidation run.
        Extracts concept-definition pairs from the synthesized content.
        """
        # Simple extraction: each sentence as a potential concept
        import re
        sentences = re.split(r'[.!?]\s+', sleep_content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 4:
                continue
            # Use first 4 words as concept key, rest as definition
            words = sentence.split()
            concept = " ".join(words[:4])
            definition = sentence
            self.integrate(concept, definition, source="sleep_replay")

    def get_schema_summary(self) -> dict:
        """Returns a human-readable schema summary for the UI."""
        data = self._load()
        return {
            "concept_count":        len(data["concepts"]),
            "relation_count":       len(data["relations"]),
            "consolidation_count":  data.get("consolidation_count", 0),
            "top_concepts":         list(data["concepts"].keys())[:20],
            "last_updated":         data.get("last_updated", 0),
        }

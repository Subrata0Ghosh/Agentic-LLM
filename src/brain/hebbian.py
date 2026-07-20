"""
Hebbian Plasticity — src/brain/hebbian.py
==========================================
Brain Region: Synaptic connections (all cortical areas)

Hebb's Rule (1949): "Neurons that fire together, wire together."
This is the most fundamental learning rule in biological neural networks.
It's CONTINUOUS — always happening, not just during sleep.

Unlike the offline fine-tuning (which is like sleep), Hebbian learning
is tonic: it modifies connection strengths during WAKING experience.

Biological basis:
  - Long-Term Potentiation (LTP): repeated co-activation → stronger synapse
  - Long-Term Depression (LTD): weak/infrequent co-activation → weaker synapse
  - Synaptic scaling: overall activity is homeostically balanced

Implementation:
  - Tracks concept co-occurrences within each conversational turn
  - On each turn, strengthens semantic graph edges for co-occurring concepts
  - Edge weights decay slowly over time (LTD / synaptic depression)
  - Provides insight into which concept pairs are most strongly associated
"""

import re
import time
from typing import List


# Stop words — not tracked for Hebbian associations
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "my", "your", "our", "their", "with", "from",
    "into", "and", "or", "but", "for", "not", "what", "how", "why", "when",
    "where", "who", "which", "i", "you", "we", "they", "he", "she",
    "to", "of", "in", "on", "at", "by", "up", "as", "if", "so", "no",
}

MIN_WORD_LEN = 4   # Only track content words of ≥ 4 chars


class HebbianPlasticity:
    """
    Continuous synaptic plasticity — Hebb's rule on concept co-occurrence.
    Works by extracting key concepts from each turn and strengthening
    their connections in the semantic graph.
    """

    def __init__(self, semantic_graph):
        """
        Args:
            semantic_graph: A SemanticGraph instance to update.
        """
        self.graph = semantic_graph
        self._turn_concepts: List[str] = []  # Concepts from the current turn

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key content words (concepts) from text."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        concepts = [w for w in words
                    if len(w) >= MIN_WORD_LEN and w not in STOP_WORDS]
        # Deduplicate while preserving order
        seen, unique = set(), []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:20]  # Max 20 concepts per turn

    def process_turn(self, text: str, co_strength: float = 0.3):
        """
        Process a single conversational turn.
        Extracts concepts and strengthens their co-occurrence edges in the
        semantic graph. All concept pairs that appear together are linked.

        Args:
            text:        The text of the current turn.
            co_strength: Base Hebbian strength (modulated by DA externally).
        """
        concepts = self._extract_concepts(text)
        self._turn_concepts = concepts

        if len(concepts) < 2:
            return  # Not enough concepts for co-occurrence

        # Add all concepts as nodes
        for concept in concepts:
            self.graph.add_node(concept, definition="", source="hebbian")

        # Strengthen edges for all pairs (within window)
        # Use a sliding window of 5 to approximate "local" co-occurrence
        window = 5
        for i in range(len(concepts)):
            for j in range(i + 1, min(i + window, len(concepts))):
                # Distance decay: closer concepts → stronger association
                distance_weight = co_strength * (1.0 - (j - i) / window * 0.5)
                self.graph.add_hebbian_edge(concepts[i], concepts[j],
                                             co_occurrence_strength=distance_weight)

    def apply_decay(self, decay_factor: float = 0.002):
        """
        Apply synaptic depression: slightly weaken all edges.
        Called periodically (e.g., during sleep or every N turns).
        Implements LTD (Long-Term Depression) for unused connections.
        """
        data = self.graph._load()
        for edge in data["edges"]:
            edge["weight"] = max(0.05, edge["weight"] - decay_factor)
        # Remove near-zero edges
        data["edges"] = [e for e in data["edges"] if e["weight"] > 0.05]
        self.graph._save(data)

    def get_strongest_associations(self, concept: str, top_k: int = 5) -> list:
        """Returns the strongest Hebbian associations for a concept."""
        data = self.graph._load()
        norm_c = self.graph._norm(concept)
        edges = [
            {"target": e["tgt"], "weight": e["weight"], "rel": e["rel"]}
            for e in data["edges"]
            if e["src"] == norm_c
        ]
        edges.sort(key=lambda x: x["weight"], reverse=True)
        return edges[:top_k]

    def get_last_turn_concepts(self) -> List[str]:
        """Returns the concepts extracted from the most recent turn."""
        return self._turn_concepts

"""
Semantic Graph — src/brain/semantic_graph.py
==============================================
Brain Region: Association Cortex + Neocortex (concept binding)

The neocortex doesn't store concepts in isolation — it stores them as a
GRAPH of relationships. This is why humans can:
  - Reason analogically: "A is to B as C is to ?"
  - Make multi-hop inferences: "If A→B and B→C, then A→C"
  - Cluster related knowledge: "What are all things related to X?"
  - Identify contradictions: "A implies B but you said not-B"

Implemented using networkx (lightweight, no GPU needed).

Key capabilities:
  1. graph_search(query, hops=2)  — multi-hop traversal
  2. analogy(a, b, c)             — A:B = C:?
  3. get_cluster(concept)         — all concepts within 2 hops
  4. add_hebbian_edge(a, b, w)    — strengthen co-occurring concepts
  5. find_contradictions()        — detect conflicting belief edges
"""

import os
import json
import time

GRAPH_FILE = "data/semantic_graph.json"


class SemanticGraph:
    """
    Directed weighted concept graph for relational knowledge reasoning.
    Stored as JSON adjacency list for persistence; loaded into memory on access.
    """

    def __init__(self, filepath: str = GRAPH_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {"nodes": {}, "edges": []}

    def _save(self, data: dict):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _norm(self, s: str) -> str:
        return s.lower().strip()[:50]

    def add_node(self, concept: str, definition: str = "", source: str = ""):
        data = self._load()
        key = self._norm(concept)
        if key not in data["nodes"]:
            data["nodes"][key] = {
                "definition": definition[:200],
                "source": source,
                "added_ts": time.time(),
                "access_count": 0,
            }
        else:
            # Update definition if richer
            if len(definition) > len(data["nodes"][key].get("definition", "")):
                data["nodes"][key]["definition"] = definition[:200]
        self._save(data)

    def add_edge(self, source: str, target: str,
                  relation: str = "related_to", weight: float = 0.5):
        """Add or strengthen an edge between two concepts."""
        data = self._load()
        src, tgt = self._norm(source), self._norm(target)

        # Add nodes if missing
        for k in [src, tgt]:
            if k not in data["nodes"]:
                data["nodes"][k] = {"definition": "", "added_ts": time.time(), "access_count": 0}

        # Check if edge exists
        for edge in data["edges"]:
            if edge["src"] == src and edge["tgt"] == tgt and edge["rel"] == relation:
                edge["weight"] = min(1.0, edge["weight"] + weight * 0.2)  # Strengthen
                self._save(data)
                return

        data["edges"].append({
            "src": src, "tgt": tgt,
            "rel": relation, "weight": round(weight, 3),
            "ts":  time.time()
        })
        # Cap
        if len(data["edges"]) > 2000:
            # Remove weakest edges
            data["edges"].sort(key=lambda e: e["weight"])
            data["edges"] = data["edges"][200:]

        self._save(data)

    def add_hebbian_edge(self, concept_a: str, concept_b: str,
                          co_occurrence_strength: float = 0.3):
        """
        Hebbian plasticity: 'neurons that fire together, wire together'.
        Strengthens connection between co-occurring concepts.
        """
        self.add_edge(concept_a, concept_b, "co_occurs", co_occurrence_strength)
        self.add_edge(concept_b, concept_a, "co_occurs", co_occurrence_strength)

    def graph_search(self, query: str, hops: int = 2) -> list:
        """
        Multi-hop graph traversal from the most relevant concept.
        Returns all concepts reachable within N hops with their definitions.
        """
        data = self._load()
        query_words = set(query.lower().split())

        # Find best matching start node
        best_node = None
        best_score = 0
        for node_key, node_data in data["nodes"].items():
            node_words = set(node_key.split())
            def_words  = set((node_data.get("definition","")).lower().split())
            score = len(query_words & (node_words | def_words))
            if score > best_score:
                best_score = score
                best_node = node_key

        if not best_node or best_score == 0:
            return []

        # BFS from best_node up to 'hops' steps
        visited = {best_node}
        queue = [(best_node, 0)]
        results = []

        # Build adjacency for fast lookup
        adj: dict = {}
        for edge in data["edges"]:
            adj.setdefault(edge["src"], []).append((edge["tgt"], edge["weight"], edge["rel"]))

        while queue:
            current, depth = queue.pop(0)
            node_info = data["nodes"].get(current, {})
            results.append({
                "concept":    current,
                "definition": node_info.get("definition", ""),
                "depth":      depth,
            })

            if depth < hops:
                for neighbor, weight, rel in adj.get(current, []):
                    if neighbor not in visited and weight > 0.15:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        # Sort by depth (closer = more relevant)
        results.sort(key=lambda x: x["depth"])
        return results[:15]

    def analogy(self, a: str, b: str, c: str) -> list:
        """
        Analogy: A is to B as C is to ?
        Finds nodes related to C in the same way B relates to A.
        """
        data = self._load()
        na, nb, nc = self._norm(a), self._norm(b), self._norm(c)

        # Find relation type between A and B
        a_to_b_relations = {
            e["rel"] for e in data["edges"]
            if e["src"] == na and e["tgt"] == nb
        }
        if not a_to_b_relations:
            a_to_b_relations = {"related_to"}

        # Find nodes related to C by the same relation
        answers = []
        for edge in data["edges"]:
            if edge["src"] == nc and edge["rel"] in a_to_b_relations:
                answers.append({
                    "answer": edge["tgt"],
                    "relation": edge["rel"],
                    "weight": edge["weight"]
                })

        answers.sort(key=lambda x: x["weight"], reverse=True)
        return answers[:5]

    def get_cluster(self, concept: str, max_hops: int = 2) -> list:
        """Returns all concepts within max_hops of a given concept."""
        return self.graph_search(concept, hops=max_hops)

    def get_context_string(self, query: str, hops: int = 2) -> str:
        """Returns a formatted context string for DMN/generation use."""
        results = self.graph_search(query, hops=hops)
        if not results:
            return ""
        lines = ["### SEMANTIC GRAPH CONTEXT ###"]
        for r in results:
            depth_marker = "  " * r["depth"] + "→ " if r["depth"] > 0 else ""
            line = f"{depth_marker}{r['concept']}"
            if r.get("definition"):
                line += f": {r['definition'][:100]}"
            lines.append(line)
        lines.append("### END GRAPH CONTEXT ###")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        data = self._load()
        return {
            "node_count": len(data["nodes"]),
            "edge_count": len(data["edges"]),
            "top_nodes":  list(data["nodes"].keys())[:10],
        }

    def causal_chain(self, concept_a: str, concept_b: str, max_hops: int = 4) -> dict:
        """
        Finds a causal chain of relationships between concept_a and concept_b.
        Returns the path of concepts and relationship types along the chain.

        Analogous to: "Does A cause B? What is the mechanism?"
        Uses BFS over the directed semantic graph.
        """
        data = self._load()
        edges = data.get("edges", [])
        nodes = data.get("nodes", {})

        # Build adjacency list: concept -> [(neighbor, relation, weight), ...]
        adj = {}
        for edge in edges:
            src = edge.get("source", "").lower()
            tgt = edge.get("target", "").lower()
            rel = edge.get("relation", "related_to")
            wt  = edge.get("weight", 0.5)
            if src not in adj:
                adj[src] = []
            adj[src].append((tgt, rel, wt))

        a = concept_a.lower()
        b = concept_b.lower()

        if a not in adj:
            return {"found": False, "path": [], "chain": "No causal connection found in graph."}

        # BFS
        from collections import deque
        queue = deque([(a, [a], [])])   # (current, path_nodes, path_relations)
        visited = {a}

        while queue:
            curr, path_nodes, path_rels = queue.popleft()
            if len(path_nodes) > max_hops + 1:
                continue
            for neighbor, rel, wt in adj.get(curr, []):
                if neighbor == b:
                    full_path = path_nodes + [neighbor]
                    full_rels = path_rels + [rel]
                    chain_str = " → ".join(
                        f"{full_path[i]} [{full_rels[i]}] {full_path[i+1]}"
                        for i in range(len(full_rels))
                    )
                    return {"found": True, "path": full_path, "relations": full_rels, "chain": chain_str}
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path_nodes + [neighbor], path_rels + [rel]))

        return {"found": False, "path": [], "chain": f"No path from '{concept_a}' to '{concept_b}' within {max_hops} hops."}

    def analogical_match(self, source_a: str, source_b: str, target_a: str) -> dict:
        """
        Analogical reasoning: If A:B, then C:?
        Finds the concept X such that target_a:X mirrors source_a:source_b.

        Method:
          1. Find the relation type on edge source_a → source_b
          2. Find all edges from target_a with the same relation type
          3. Return the best matching target
        """
        data = self._load()
        edges = data.get("edges", [])
        nodes = data.get("nodes", {})

        a = source_a.lower()
        b = source_b.lower()
        c = target_a.lower()

        # Find relation between a and b
        ab_relation = None
        for edge in edges:
            if edge.get("source", "").lower() == a and edge.get("target", "").lower() == b:
                ab_relation = edge.get("relation", "related_to")
                break

        if not ab_relation:
            return {
                "found": False,
                "answer": None,
                "analogy": f"{source_a}:{source_b} = {target_a}:? (no AB relation found)"
            }

        # Find edge from c with same relation
        candidates = []
        for edge in edges:
            if (edge.get("source", "").lower() == c and
                    edge.get("relation", "") == ab_relation):
                candidates.append((edge.get("target", ""), edge.get("weight", 0.5)))

        if not candidates:
            return {
                "found": False,
                "answer": None,
                "analogy": f"{source_a}:{source_b} = {target_a}:? (no matching edge from {target_a} with relation '{ab_relation}')"
            }

        # Return highest-weight candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0][0]
        return {
            "found": True,
            "answer": best,
            "relation": ab_relation,
            "analogy": f"{source_a}:{source_b} = {target_a}:{best} (via '{ab_relation}')",
            "alternatives": [c[0] for c in candidates[1:4]],
        }

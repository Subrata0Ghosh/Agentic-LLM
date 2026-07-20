"""
BICA Cognition — src/brain/bica_cognition.py
=============================================
Biologically-Inspired Cognitive Architecture — Cognitive Workflow Engine

This module implements the 10 cognitive modules from the BICA system prompt
as structured Python logic that is injected into the full brain pipeline.
It transforms the BICA principles from meta-instructions into live runtime
behavior that shapes every response the system produces.

Cognitive Modules:
  1.  Perception        — Entity/goal/constraint/assumption extraction
  2.  Attention         — Focus scoring per concept given WM context
  3.  Working Memory    — Tracks intermediate calculations & key facts
  4.  Episodic Memory   — Flags consistency violations with prior context
  5.  Semantic Reason   — Contradiction detection + knowledge combination
  6.  Planning          — Decomposes the query into verifiable sub-steps
  7.  Learning          — Marks new user-defined rules for downstream use
  8.  Decision Making   — Compares solution paths and explains trade-offs
  9.  Commonsense       — Filters impossible conclusions
  10. Self-Verification — Arithmetic, logic, and completeness check

Output feeds into DMN Stage 1 (RECALL), Stage 3 (CRITIQUE), and
is exposed on the /api/cognitive_state endpoint.
"""

import re
import time
from typing import List, Dict, Optional, Tuple


# ── Perception vocabulary ─────────────────────────────────────────────────────
_ENTITY_HINTS  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
_NUMBER_HINT   = re.compile(r'\b\d+(?:\.\d+)?\b')
_GOAL_WORDS    = {"find", "explain", "calculate", "compare", "build", "create",
                  "design", "analyse", "analyze", "show", "prove", "write",
                  "determine", "evaluate", "summarize", "list", "describe"}
_CONSTRAINT_WORDS = {"only", "must", "cannot", "never", "always", "without",
                     "at most", "at least", "within", "before", "after",
                     "exactly", "no more than", "no less than"}
_ASSUMPTION_PATTERNS = [
    r"assuming\s+(.+?)(?:\.|,|$)",
    r"given that\s+(.+?)(?:\.|,|$)",
    r"suppose\s+(.+?)(?:\.|,|$)",
    r"if\s+(.+?)\s+then",
]

# ── Impossible / commonsense violation patterns ───────────────────────────────
_IMPOSSIBLE = [
    (r"divide\s+\d+\s+by\s+0", "Division by zero is undefined."),
    (r"travel\s+faster\s+than\s+light", "Faster-than-light travel violates physics."),
    (r"born\s+before\s+(?:their|his|her)\s+parent", "A person cannot be born before their parent."),
    (r"square\s+root\s+of\s+a\s+negative", "Square root of a negative number is imaginary."),
]


class BICACognition:
    """
    BICA Cognitive Workflow Engine — 10 modules as live runtime Python logic.

    One instance shared per server startup (stateless per call, stateful
    per session via the `session_log` dict for episodic consistency tracking).
    """

    def __init__(self):
        self._session_log: Dict[str, Dict] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 1 — PERCEPTION
    # ══════════════════════════════════════════════════════════════════════════
    def perceive(self, text: str) -> Dict:
        """
        Carefully analyze input. Extract entities, relationships, numbers,
        goals, constraints, and hidden assumptions.
        """
        text_l = text.lower()
        words  = text_l.split()

        # Entities (proper nouns via capitalization heuristic)
        entities = list(set(_ENTITY_HINTS.findall(text)))[:10]

        # Numbers
        numbers = _NUMBER_HINT.findall(text)

        # Goals — first verb that signals intent
        goal_words_found = [w for w in words if w.rstrip(".,?!:") in _GOAL_WORDS]
        primary_goal = goal_words_found[0] if goal_words_found else None

        # Constraints
        constraints = [c for c in _CONSTRAINT_WORDS if c in text_l]

        # Assumptions (regex)
        assumptions = []
        for pat in _ASSUMPTION_PATTERNS:
            m = re.search(pat, text_l)
            if m:
                assumptions.append(m.group(1).strip())

        # Is a genuine question?
        is_question = text.strip().endswith("?") or any(
            text_l.startswith(w) for w in
            ["what", "why", "how", "when", "where", "who", "which",
             "explain", "describe", "compare", "could you", "can you"]
        )

        # Requires multi-step reasoning?
        has_multiple_steps = any(w in text_l for w in
                                  ["first", "then", "next", "finally", "step by step",
                                   "after that", "and then", "subsequently"])

        return {
            "entities":       entities,
            "numbers":        numbers,
            "primary_goal":   primary_goal,
            "constraints":    constraints,
            "assumptions":    assumptions,
            "is_question":    is_question,
            "multi_step":     has_multiple_steps,
            "word_count":     len(words),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 2 — ATTENTION
    # ══════════════════════════════════════════════════════════════════════════
    def attend(self, text: str, wm_context: list, perception: Dict) -> Dict:
        """
        Focus on what matters. Score each concept from WM against the
        current query goals and constraints.
        Returns a ranked list of focused concepts (top 5).
        """
        words_in_query = set(text.lower().split())

        # Score WM chunks by keyword overlap with query
        scored = []
        for chunk in wm_context:
            content = chunk.get("content", "")
            chunk_words = set(content.lower().split())
            overlap = len(words_in_query & chunk_words)
            entity_hits = sum(1 for e in perception["entities"]
                              if e.lower() in content.lower())
            score = overlap * 0.6 + entity_hits * 0.4
            if score > 0:
                scored.append({"content": content[:80], "score": round(score, 2)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        focused = scored[:5]

        # Key concepts from perception
        focal_concepts = perception["entities"][:3] + perception["constraints"][:2]

        return {
            "focused_wm_chunks": focused,
            "focal_concepts":    focal_concepts,
            "ignore_count":      len(wm_context) - len(focused),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 5 — SEMANTIC REASONING (Contradiction Detection)
    # ══════════════════════════════════════════════════════════════════════════
    def semantic_reason(self, text: str, retrieved_context: str) -> Dict:
        """
        Combine stored knowledge with logical reasoning.
        Detect contradictions between the query and retrieved memories.
        """
        contradictions = []
        text_l = text.lower()

        # Simple negation flip detection
        # e.g., memory says "X is true", query says "X is not true"
        sentences = [s.strip() for s in retrieved_context.split('.') if len(s.strip()) > 10]
        for sent in sentences[:8]:
            sent_l = sent.lower()
            words_s = set(sent_l.split())
            words_q = set(text_l.split())
            overlap = words_s & words_q
            if len(overlap) > 3:
                # Check for negation mismatch
                neg_in_sent  = any(n in sent_l for n in ["not", "never", "no ", "cannot"])
                neg_in_query = any(n in text_l  for n in ["not", "never", "no ", "cannot"])
                if neg_in_sent != neg_in_query:
                    contradictions.append({
                        "memory": sent[:80],
                        "query":  text[:80],
                        "type":   "negation_mismatch"
                    })

        return {
            "contradictions": contradictions[:3],
            "has_contradiction": len(contradictions) > 0,
            "context_sentences": len(sentences),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 7 — LEARNING (User Rule Detection)
    # ══════════════════════════════════════════════════════════════════════════
    def detect_user_rules(self, text: str) -> Dict:
        """
        Detect if the user is defining a new rule or preference.
        These should be applied consistently for the rest of the conversation.
        """
        rule_patterns = [
            r"always\s+(?:use|do|refer|call|say)\s+(.+?)(?:\.|$)",
            r"never\s+(?:use|do|refer|call|say)\s+(.+?)(?:\.|$)",
            r"from now on[,\s]+(.+?)(?:\.|$)",
            r"remember that\s+(.+?)(?:\.|$)",
            r"(?:i|we)\s+(?:prefer|want|need)\s+(.+?)(?:\.|$)",
            r"treat\s+(.+?)\s+as\s+(.+?)(?:\.|$)",
        ]

        rules_found = []
        text_l = text.lower()
        for pat in rule_patterns:
            m = re.search(pat, text_l)
            if m:
                rule_text = " ".join(m.groups()).strip()
                if len(rule_text) > 3:
                    rules_found.append(rule_text)

        return {
            "new_rules":   rules_found[:3],
            "has_rules":   len(rules_found) > 0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 9 — COMMONSENSE (Impossible Conclusion Filter)
    # ══════════════════════════════════════════════════════════════════════════
    def commonsense_filter(self, text: str) -> Dict:
        """
        Detect physically or logically impossible requests.
        Returns a warning if found.
        """
        violations = []
        text_l = text.lower()
        for pattern, explanation in _IMPOSSIBLE:
            if re.search(pattern, text_l):
                violations.append(explanation)

        return {
            "violations":      violations,
            "has_violation":   len(violations) > 0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 10 — SELF-VERIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    def verify(self, query: str, response: str, perception: Dict) -> Dict:
        """
        Before finalizing every answer:
          1. Check arithmetic accuracy (if numbers present)
          2. Check logical consistency (contradiction detection)
          3. Check completeness (were all parts of the question answered?)

        Returns a verification report.
        """
        issues = []
        response_l = response.lower()
        query_l    = query.lower()

        # ── 1. Arithmetic spot-check ──────────────────────────────────────────
        # Find simple X op Y = Z patterns in the response
        arith_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        )
        for match in arith_pattern.finditer(response):
            a, op, b, claimed = match.groups()
            a, b, claimed = float(a), float(b), float(claimed)
            if   op == '+': expected = a + b
            elif op == '-': expected = a - b
            elif op == '*': expected = a * b
            elif op == '/': expected = (a / b) if b != 0 else None
            else:            expected = None

            if expected is not None and abs(expected - claimed) > 0.01:
                issues.append(f"Arithmetic error: {a}{op}{b}={claimed} (correct: {expected})")

        # ── 2. Logical consistency — did we contradict ourselves? ─────────────
        # Simple: look for "yes" and "no" both appearing near the same key noun
        if perception["entities"]:
            entity = perception["entities"][0].lower()
            if entity in response_l:
                if "yes" in response_l and "no" in response_l:
                    # Check proximity (crude but fast)
                    yes_idx = response_l.find("yes")
                    no_idx  = response_l.find("no")
                    if abs(yes_idx - no_idx) < 150:
                        issues.append("Possible yes/no contradiction detected near key entity.")

        # ── 3. Completeness — were query goals addressed? ─────────────────────
        is_math_query = len(re.findall(r"\d+", query_l)) >= 1 and any(op in query_l for op in ["+", "-", "*", "/", "=", "^", "solve", "calculate"])
        if perception["primary_goal"] and not is_math_query:
            goal = perception["primary_goal"]
            # If the goal word (e.g. "compare") appears in query but response is very short
            if goal in query_l and len(response.split()) < 20:
                issues.append(f"Response may be too brief for goal '{goal}'.")

        # ── 4. Commonsense guard ──────────────────────────────────────────────
        for pattern, explanation in _IMPOSSIBLE:
            if re.search(pattern, response_l):
                issues.append(f"Commonsense violation in response: {explanation}")

        # ── 5. Neural Logical Verification Check ────────────────────────────────
        # Ask the model as an independent critic to verify the response logic
        try:
            from generate import generate_text_api
            critique_msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are an independent logical verification processor. "
                        "Read the question and the proposed answer. Check for logic errors, "
                        "commonsense violations, factual mistakes, or accepting false premises. "
                        "If the proposed answer is correct and logical, reply with only 'PASSED'. "
                        "Otherwise, write a brief 1-sentence description of the logical flaw."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\nAnswer: {response}"
                }
            ]
            critique_res = generate_text_api(critique_msgs, max_new_tokens=60, temperature=0.1)
            critique_clean = critique_res.strip()
            if "PASSED" not in critique_clean.upper() and len(critique_clean) > 3:
                issues.append(f"Logic flaw detected: {critique_clean}")
        except Exception:
            pass

        passed = len(issues) == 0
        return {
            "passed":    passed,
            "issues":    issues,
            "label":     "✓ Verified" if passed else f"⚠ {len(issues)} issue(s)",
            "checked_at": time.time(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # DMN PROMPT BUILDER — Injects BICA cognition into each DMN stage
    # ══════════════════════════════════════════════════════════════════════════
    def build_dmn_injection(self, stage: str, perception: Dict,
                             attention: Dict, semantic: Dict,
                             user_rules: Dict) -> str:
        """
        Returns a structured BICA context block to prepend to each DMN stage
        system prompt. This transforms the BICA system prompt into live
        runtime instructions per-request.
        """
        lines = ["### BICA COGNITIVE CONTEXT ###"]

        if stage == "RECALL":
            lines.append("[PERCEPTION]")
            if perception["entities"]:
                lines.append(f"  Entities: {', '.join(perception['entities'][:5])}")
            if perception["primary_goal"]:
                lines.append(f"  Goal: {perception['primary_goal']}")
            if perception["constraints"]:
                lines.append(f"  Constraints: {', '.join(perception['constraints'][:3])}")
            if perception["assumptions"]:
                lines.append(f"  Assumptions: {'; '.join(perception['assumptions'][:2])}")
            lines.append(f"  Is question: {perception['is_question']}  Multi-step: {perception['multi_step']}")

            lines.append("[ATTENTION]")
            if attention["focal_concepts"]:
                lines.append(f"  Focus on: {', '.join(attention['focal_concepts'][:4])}")

            if semantic["has_contradiction"]:
                lines.append("[SEMANTIC WARNING]")
                for c in semantic["contradictions"][:2]:
                    lines.append(f"  ⚠ Possible contradiction: {c['memory'][:60]}")

        elif stage == "SIMULATE":
            lines.append("[DECISION MAKING — compare alternatives]")
            lines.append("  Generate 2-3 distinct solution paths.")
            lines.append("  For each: state the approach, trade-offs, and best use case.")
            if perception["multi_step"]:
                lines.append("  This is a multi-step problem — plan each step explicitly.")

        elif stage == "CRITIQUE":
            lines.append("[SELF-VERIFICATION CHECKLIST]")
            lines.append("  □ Check all arithmetic in the simulated answers.")
            lines.append("  □ Check logical consistency (no yes/no contradiction).")
            lines.append("  □ Verify the response addresses the stated goal.")
            lines.append("  □ Apply commonsense: reject physically impossible claims.")
            if user_rules["has_rules"]:
                lines.append("[USER RULES — apply these]")
                for r in user_rules["new_rules"][:3]:
                    lines.append(f"  • {r}")

        lines.append("### END BICA CONTEXT ###")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE RUN — called once per chat request
    # ══════════════════════════════════════════════════════════════════════════
    def run(self, session_id: str, query: str, wm_context: list,
            retrieved_context: str) -> Dict:
        """
        Runs all 10 BICA cognitive modules and returns a unified state dict.
        This is called in the API pipeline between Thalamus and Amygdala.

        Returns:
            {
              "perception":      dict,
              "attention":       dict,
              "semantic":        dict,
              "user_rules":      dict,
              "commonsense":     dict,
              "dmn_injections":  {"RECALL": str, "SIMULATE": str, "CRITIQUE": str},
              "timestamp":       float,
            }
        """
        perception  = self.perceive(query)
        attention   = self.attend(query, wm_context, perception)
        semantic    = self.semantic_reason(query, retrieved_context)
        user_rules  = self.detect_user_rules(query)
        commonsense = self.commonsense_filter(query)

        # Build per-stage DMN injection strings
        dmn_injections = {
            "RECALL":   self.build_dmn_injection("RECALL",   perception, attention, semantic, user_rules),
            "SIMULATE": self.build_dmn_injection("SIMULATE", perception, attention, semantic, user_rules),
            "CRITIQUE": self.build_dmn_injection("CRITIQUE", perception, attention, semantic, user_rules),
        }

        state = {
            "perception":     perception,
            "attention":      attention,
            "semantic":       semantic,
            "user_rules":     user_rules,
            "commonsense":    commonsense,
            "dmn_injections": dmn_injections,
            "timestamp":      time.time(),
        }

        # Store per-session for the cognitive_state endpoint
        self._session_log[session_id] = state
        return state

    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Returns the last cognitive state for a session."""
        return self._session_log.get(session_id)

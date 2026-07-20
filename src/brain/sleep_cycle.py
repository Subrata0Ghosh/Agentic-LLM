"""
Sleep Cycle — src/brain/sleep_cycle.py
========================================
Brain Region: Various — coordinates cortical-hippocampal dialogue during sleep

Human sleep is NOT rest. It is active processing with 4 distinct phases:

  Phase 1 — SLOW WAVE SLEEP (SWS) / Memory Selection:
    The hippocampus replays high-priority episodic memories to the neocortex.
    Selection is driven by amygdala priority score (emotional memories first).
    Only reinforced (recalled ≥2x) OR high-emotion memories are consolidated.
    Low-value memories are tagged for decay.

  Phase 2 — REM / Associative Dreaming:
    The brain makes NOVEL CONNECTIONS between memory clusters.
    This is the source of creativity, insight, and "aha moments".
    The LLM is used to generate associative links between high-priority memories.
    These novel links are stored back as new synthetic memories.

  Phase 3 — CLEARANCE / Memory Pruning:
    Weak, unreinforced memories are pruned (forgotten).
    Neuromodulators reset to biological baseline.
    Working memory is cleared.

  Phase 4 — NEOCORTICAL FINE-TUNING:
    The synthesized SWS+REM output is used to fine-tune the custom PyTorch
    Transformer (CustomLLM) weights — permanently encoding the new knowledge
    into the neocortex's structural patterns.
"""

import sys
import os
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from generate import generate_text_api
from online_train import online_finetune

SLEEP_LOG_FILE = "data/sleep_log.json"


class SleepCycle:
    """
    4-phase sleep and memory consolidation coordinator.
    """

    def __init__(self):
        os.makedirs("data", exist_ok=True)

    def _log(self, session_id: str, phase: str, message: str):
        """Append to sleep log."""
        entry = {"ts": time.time(), "session": session_id,
                 "phase": phase, "message": message}
        try:
            log = []
            if os.path.exists(SLEEP_LOG_FILE):
                with open(SLEEP_LOG_FILE, "r") as f:
                    log = json.load(f)
            log.append(entry)
            log = log[-100:]  # Keep last 100 entries
            with open(SLEEP_LOG_FILE, "w") as f:
                json.dump(log, f, indent=2)
        except Exception:
            pass
        print(f"[Sleep:{phase}] {message}")

    def phase1_sws(self, session_id: str, hippocampus) -> list:
        """
        SLOW WAVE SLEEP: Retrieve and rank memories by priority.
        Returns selected memories for neocortical consolidation.
        """
        self._log(session_id, "SWS", "Retrieving high-priority episodic memories...")

        candidates = hippocampus.get_high_priority_memories(top_n=15)
        if not candidates:
            self._log(session_id, "SWS", "No memories found to consolidate.")
            return []

        # Select only memories with priority > 0.2 (quality gate)
        selected = [
            m for m in candidates
            if m["meta"] and m["meta"].get("priority", 0.0) > 0.2
        ]

        self._log(session_id, "SWS",
                  f"Selected {len(selected)}/{len(candidates)} memories for replay.")
        return selected

    def phase2_rem(self, session_id: str, sws_memories: list,
                   gen_params: dict) -> str:
        """
        REM SLEEP: Generate novel associative connections between SWS memories.
        This is where creative synthesis and insight emerge.
        """
        if len(sws_memories) < 2:
            return ""

        self._log(session_id, "REM", "Associative dreaming — synthesizing cross-memory links...")

        # Take top 4 memories for REM association
        top_mems = sws_memories[:4]
        memory_texts = "\n\n".join([
            f"Memory {i+1}: {m['content'][:200]}"
            for i, m in enumerate(top_mems)
        ])

        system = (
            "You are the brain's REM sleep process. Your task is to find "
            "SURPRISING and NON-OBVIOUS connections between these memory fragments. "
            "Generate 3 novel insights, analogies, or generalizations that link these memories. "
            "Think creatively — this is a dream state. Output numbered insights only."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Memory fragments from today:\n\n{memory_texts}\n\nGenerate 3 novel cross-memory insights:"}
        ]

        try:
            insights = generate_text_api(
                messages,
                max_new_tokens=200,
                temperature=min(1.0, gen_params.get("temperature", 0.5) + 0.3),
                top_p=0.97,
                repetition_penalty=1.05
            )
            self._log(session_id, "REM", f"Generated {len(insights.split())} words of novel associations.")
            return insights
        except Exception as e:
            self._log(session_id, "REM", f"REM synthesis failed: {e}")
            return ""

    def phase3_clearance(self, session_id: str, neuromodulators,
                          working_memory):
        """
        CLEARANCE: Reset neuromodulators and clear working memory.
        This is the brain's "maintenance" phase.
        """
        self._log(session_id, "CLEARANCE", "Resetting neuromodulators to baseline...")
        neuromodulators.sleep_reset(session_id)

        self._log(session_id, "CLEARANCE", "Clearing working memory...")
        working_memory.clear(session_id)

        self._log(session_id, "CLEARANCE", "Clearance complete. Brain refreshed.")

    def phase4_finetune(self, session_id: str, sws_content: str,
                         rem_content: str, max_iters: int = 20) -> dict:
        """
        NEOCORTICAL FINE-TUNING: Train custom PyTorch Transformer on
        consolidated SWS + REM content to permanently encode new knowledge.
        """
        corpus = (sws_content + "\n\n" + rem_content).strip()
        if not corpus or len(corpus.split()) < 15:
            self._log(session_id, "FINETUNE", "Insufficient content for fine-tuning.")
            return {"success": False, "reason": "Insufficient content"}

        self._log(session_id, "FINETUNE",
                  f"Fine-tuning CustomLLM on {len(corpus.split())} words...")

        # Append to corpus file
        try:
            with open("data/ai_corpus_cleaned.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n# Sleep Consolidation [{session_id}]:\n{corpus}")
        except Exception as e:
            self._log(session_id, "FINETUNE", f"Corpus append failed: {e}")

        # Run fine-tuning
        try:
            online_finetune(corpus, max_iters=max_iters)
            self._log(session_id, "FINETUNE",
                      "CustomLLM weights updated successfully.")
            return {"success": True}
        except Exception as e:
            self._log(session_id, "FINETUNE", f"Fine-tuning failed: {e}")
            return {"success": False, "reason": str(e)}

    def run_full_cycle(self, session_id: str, hippocampus,
                        neuromodulators, working_memory,
                        gen_params: dict, max_finetune_iters: int = 20) -> str:
        """
        Runs the complete 4-phase sleep cycle.
        Returns a human-readable summary.
        """
        start_time = time.time()
        self._log(session_id, "START", "Beginning full sleep cycle...")
        summary_lines = []

        # Phase 1: SWS
        sws_memories = self.phase1_sws(session_id, hippocampus)
        sws_content = "\n".join([m["content"][:300] for m in sws_memories])
        summary_lines.append(f"SWS: Replayed {len(sws_memories)} high-priority memories.")

        # Phase 2: REM
        rem_content = ""
        if sws_memories:
            rem_content = self.phase2_rem(session_id, sws_memories, gen_params)
            if rem_content:
                summary_lines.append("REM: Generated novel associative insights.")

        # Phase 3: Clearance
        self.phase3_clearance(session_id, neuromodulators, working_memory)
        summary_lines.append("CLEARANCE: Neuromodulators reset, working memory cleared.")

        # Phase 4: Fine-tuning
        result = self.phase4_finetune(session_id, sws_content, rem_content,
                                       max_iters=max_finetune_iters)
        if result["success"]:
            summary_lines.append("FINE-TUNE: CustomLLM weights updated.")
        else:
            summary_lines.append(f"FINE-TUNE: Skipped ({result.get('reason', 'unknown')})")

        elapsed = round(time.time() - start_time, 1)
        summary_lines.append(f"Total sleep cycle duration: {elapsed}s")

        return "\n".join(summary_lines)

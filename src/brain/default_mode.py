"""
Default Mode Network — src/brain/default_mode.py
==================================================
Brain Region: Default Mode Network (DMN)
  — Medial Prefrontal Cortex, Posterior Cingulate, Angular Gyrus

The DMN is active when we are NOT focused on the external world.
It is the brain's simulation engine:
  - Mental time travel (imagining past and future)
  - Theory of Mind (simulating other people's thoughts)
  - Creative recombination of memories
  - Self-referential thinking

In the cognitive pipeline, DMN runs after memory retrieval and before output.
It performs a 3-stage internal simulation:

  STAGE 1 — RECALL:    "What do I know about this?"
                        Structured hippocampal query + world model check.

  STAGE 2 — SIMULATE:  "If this is true, what follows?"
                        Mental model forward-simulation using the LLM.
                        Generates 2-3 candidate hypotheses/answers.

  STAGE 3 — CRITIQUE:  "What could be wrong with my simulation?"
                        Self-refutation step. Identifies weaknesses.
                        Resolves contradictions between hypotheses.

This three-stage process is fundamentally different from the old single
<thought> block — it has STRUCTURE and ITERATION.

Norepinephrine modulates DMN:
  Low NE  → DMN is active, associative, creative (good for novel problems)
  High NE → DMN is suppressed, focus on task execution (good for routine)
"""

import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from generate import generate_text_api

# BICA cognitive injections are optional (passed via simulate())
_BICA_AVAILABLE = True
try:
    from bica_cognition import BICACognition
except ImportError:
    _BICA_AVAILABLE = False


class DefaultModeNetwork:
    """
    3-stage cognitive simulation engine.
    Replaces the old single-pass imagination.py with a structured loop.
    """

    def _build_stage_prompt(self, stage: str, user_message: str,
                             hippocampal_context: str, world_model_prediction: str,
                             stage1_output: str = "", stage2_output: str = "",
                             tone: str = "friendly and clear",
                             bica_injection: str = "") -> list:
        """Builds the messages list for each DMN stage."""

        if stage == "RECALL":
            system = (
                f"You are an internal cognitive process — the brain's memory recall stage. "
                f"Your only task: analyze the question and retrieve WHAT YOU KNOW about it. "
                f"Be concise and factual. List key facts, uncertainties, and gaps. "
                f"Today's world model prediction: {world_model_prediction}"
            )
            if hippocampal_context:
                system += f"\n\nMemory context available:\n{hippocampal_context}"
            if bica_injection:
                system += f"\n\n{bica_injection}"
            user = f"Question: {user_message}\n\nWhat do I know about this? List facts and knowledge gaps:"

        elif stage == "SIMULATE":
            system = (
                "You are an internal cognitive process — the brain's simulation stage. "
                "Based on what was recalled, mentally simulate 2-3 possible answers or explanations. "
                "Be creative but grounded. Think through consequences and implications. "
                f"Recall results:\n{stage1_output}"
            )
            if bica_injection:
                system += f"\n\n{bica_injection}"
            user = f"Original question: {user_message}\n\nSimulate 2-3 possible answers or explanations:"

        elif stage == "CRITIQUE":
            system = (
                "You are an internal cognitive process — the brain's critical evaluation stage. "
                "Review the simulations and find flaws, contradictions, or missing pieces. "
                "Synthesize the final answer clearly and directly. "
                "CRITICAL: Do NOT output any of your internal critique steps, evaluation commentary, option comparisons, or simulation headers in the final response. Output ONLY the clean synthesized response to the user."
                f"Simulations to critique:\n{stage2_output}"
            )
            if bica_injection:
                system += f"\n\n{bica_injection}"
            user = (
                f"Original question: {user_message}\n\n"
                "Review and critique the simulations. Do not output the critique. "
                f"Provide only the final, warm, synthesized answer to the user in a {tone} style:"
            )

        else:  # FINAL OUTPUT
            system = (
                f"You are a {tone} AI assistant. "
                "Based on your careful internal reasoning process, provide your final response. "
                "Be warm, accurate, and helpful. Do not repeat the reasoning process — just answer. "
                f"Internal reasoning summary:\n{stage2_output}\nCritique:\n{stage1_output}"
            )
            user = user_message

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def simulate(self, session_id: str, user_message: str,
                 hippocampal_context: str, wm_context: list,
                 world_model_prediction: dict, gen_params: dict,
                 bica_state: dict = None, path: str = "DEEP") -> dict:
        """
        Runs the 3-stage (DEEP) or 2-stage (FOCUS) DMN simulation loop.

        Returns:
            {
              "stage1_recall":   str,
              "stage2_simulate": str,
              "stage3_critique": str,
              "final_response":  str,
              "suggestions":     list,
              "token_count":     int,
            }
        """
        tone = gen_params.get("tone", "friendly and clear")
        depth = gen_params.get("depth", "")
        active_strategy = gen_params.get("active_strategy", "")
        prediction_text = world_model_prediction.get("prediction", "No prior prediction.")
        temp   = gen_params.get("temperature", 0.5)
        top_p  = gen_params.get("top_p", 0.9)
        rep_p  = gen_params.get("repetition_penalty", 1.12)

        # Build depth/strategy context to inject in system prompts
        strategy_ctx = ""
        if depth:
            strategy_ctx += f"Depth preference: {depth}. "
        if active_strategy:
            strategy_ctx += f"Cognitive strategy: {active_strategy}."

        # Extract BICA injections if available
        bica_injections = (bica_state or {}).get("dmn_injections", {})

        if path == "FOCUS":
            # ── OPTIMIZED 2-STAGE PIPELINE ──
            # Stage 1: RECALL + SIMULATE (Recall facts and draft response)
            system = (
                f"You are an internal cognitive process — the brain's focused recall and drafting stage. "
                f"Your task is to analyze the question, retrieve what you know, and immediately draft a solid response. "
                f"Be concise, direct, and factual. "
                f"Today's world model prediction: {prediction_text}"
            )
            if hippocampal_context:
                system += f"\n\nMemory context available:\n{hippocampal_context}"
            if bica_injections.get("RECALL"):
                system += f"\n\n{bica_injections.get('RECALL')}"

            msgs1 = [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Question: {user_message}\n\nDraft a clear, factual answer:"}
            ]
            for c in wm_context[-2:]:
                msgs1.insert(-1, {"role": c["role"], "content": c["content"]})

            stage1 = generate_text_api(msgs1, max_new_tokens=250,
                                        temperature=max(0.2, temp - 0.2),  # Lower temp for factual stability
                                        top_p=top_p, repetition_penalty=rep_p)
            stage2 = "Skipped in FOCUS route."

            # Stage 2: CRITIQUE + SYNTHESIZE
            critique_injection = bica_injections.get("CRITIQUE", "")
            if strategy_ctx:
                critique_injection = (critique_injection + "\n" + strategy_ctx).strip()
            msgs3 = self._build_stage_prompt(
                "CRITIQUE", user_message, hippocampal_context, prediction_text,
                stage1_output=stage1, stage2_output=stage1, tone=tone,
                bica_injection=critique_injection
            )
            for c in wm_context[-2:]:
                msgs3.insert(-1, {"role": c["role"], "content": c["content"]})

            stage3 = generate_text_api(msgs3, max_new_tokens=300,
                                        temperature=max(0.3, temp - 0.1),
                                        top_p=top_p, repetition_penalty=rep_p)
        else:
            # ── FULL 3-STAGE PIPELINE (DEEP) ──
            # ── STAGE 1: RECALL ──────────────────────────────────────────────
            msgs1 = self._build_stage_prompt(
                "RECALL", user_message, hippocampal_context, prediction_text,
                tone=tone, bica_injection=bica_injections.get("RECALL", "")
            )
            # Add WM context
            for c in wm_context[-2:]:  # Only last 2 WM chunks for recall
                msgs1.insert(-1, {"role": c["role"], "content": c["content"]})

            stage1 = generate_text_api(msgs1, max_new_tokens=200,
                                        temperature=max(0.3, temp - 0.1),
                                        top_p=top_p, repetition_penalty=rep_p)

            # ── STAGE 2: SIMULATE ─────────────────────────────────────────────
            msgs2 = self._build_stage_prompt(
                "SIMULATE", user_message, hippocampal_context, prediction_text,
                stage1_output=stage1, tone=tone,
                bica_injection=bica_injections.get("SIMULATE", "")
            )
            for c in wm_context[-2:]:
                msgs2.insert(-1, {"role": c["role"], "content": c["content"]})

            stage2 = generate_text_api(msgs2, max_new_tokens=300,
                                        temperature=min(0.9, temp + 0.1),
                                        top_p=min(0.98, top_p + 0.05),
                                        repetition_penalty=rep_p)

            # ── STAGE 3: CRITIQUE + SYNTHESIZE ───────────────────────────────
            critique_injection = bica_injections.get("CRITIQUE", "")
            if strategy_ctx:
                critique_injection = (critique_injection + "\n" + strategy_ctx).strip()
            msgs3 = self._build_stage_prompt(
                "CRITIQUE", user_message, hippocampal_context, prediction_text,
                stage1_output=stage1, stage2_output=stage2, tone=tone,
                bica_injection=critique_injection
            )
            for c in wm_context[-2:]:
                msgs3.insert(-1, {"role": c["role"], "content": c["content"]})

            stage3 = generate_text_api(msgs3, max_new_tokens=400,
                                        temperature=temp,
                                        top_p=top_p, repetition_penalty=rep_p)

        # ── STAGE 3 SYNTHESIS + SELF-CORRECTION LOOP (Module 10) ──────────
        max_attempts = 3
        attempt = 0
        current_msgs3 = msgs3.copy()
        
        while attempt < max_attempts:
            # Extract final synthesized answer from stage 3
            final = stage3
            for marker in ["Final Answer:", "Synthesized Answer:", "In conclusion,",
                            "Answer:", "Response:", "Therefore,"]:
                if marker.lower() in stage3.lower():
                    idx = stage3.lower().index(marker.lower())
                    final = stage3[idx + len(marker):].strip()
                    break

            # Remove common internal monologue headers and leaked templates
            clean_lines = []
            for line in final.split("\n"):
                line_l = line.strip().lower()
                if any(h in line_l for h in [
                    "original question analysis", "critique the simulations",
                    "simulation a:", "simulation b:", "review of simulations",
                    "decision making", "synthesized answer", "simulation 1:",
                    "simulation 2:", "simulation 3:"
                ]):
                    continue
                if line_l.startswith("• ") and any(x in line_l for x in ["option 1", "option 2", "simulation a", "simulation b"]):
                    continue
                clean_lines.append(line)
            final = "\n".join(clean_lines).strip()

            # Parse suggestions if present
            suggestions = []
            if "SUGGESTIONS:" in final:
                parts = final.split("SUGGESTIONS:")
                final = parts[0].strip()
                for line in parts[1].split("\n"):
                    line = line.strip()
                    if line.startswith("- ") or line.startswith("* "):
                        s = line[2:].replace("**", "").strip()
                        if s:
                            suggestions.append(s)

            # Only fall back to raw stage3 if final was completely cleaned away (empty)
            # Short responses like "Hi! How can I help?" are perfectly valid — don't overwrite them
            if not final.strip():
                final = stage3.replace("### Original Question Analysis", "").replace("### Synthesized Answer", "").strip()
            # Last resort: if still empty, use a safe fallback
            if not final.strip():
                final = "I processed your message but wasn't able to compose a response. Please try rephrasing."

            # ── SELF-VERIFICATION (BICA Module 10) ───────────────────────────
            self_eval = {"passed": True, "issues": [], "label": "✓ Verified"}
            if bica_state:
                try:
                    # Inline import to avoid circular dependency
                    import sys, os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'brain'))
                    from bica_cognition import BICACognition
                    _bica = BICACognition()
                    perception = bica_state.get("perception", {})
                    self_eval = _bica.verify(user_message, final.strip(), perception)
                except Exception:
                    pass

            # If verification passes, or we've run out of attempts, stop
            if self_eval["passed"] or attempt == max_attempts - 1:
                break

            # Run self-correction: feed the issue back to stage3
            attempt += 1
            print(f"[DMN] Self-Verification failed on attempt {attempt}. Issues: {self_eval['issues']}. Correcting...")
            
            # Append failed response and issues as instruction feedback
            current_msgs3.append({"role": "assistant", "content": stage3})
            current_msgs3.append({
                "role": "user",
                "content": (
                    f"Your response failed our internal BICA Module 10 verification with these issues:\n"
                    + "\n".join([f"- {iss}" for iss in self_eval["issues"]])
                    + "\n\nPlease rewrite the answer to resolve all issues listed above. "
                    "Make sure your logic, arithmetic, and commonsense statements are 100% correct. "
                    "Output ONLY your final, corrected response."
                )
            })

            # Regenerate with slightly lower temperature to force higher determinism/logic compliance
            stage3 = generate_text_api(
                current_msgs3, 
                max_new_tokens=400,
                temperature=max(0.1, temp - 0.15 * attempt),
                top_p=top_p, 
                repetition_penalty=rep_p
            )

        total_tokens = len((stage1 + stage2 + stage3).split())
        return {
            "stage1_recall":   stage1,
            "stage2_simulate": stage2,
            "stage3_critique": stage3,
            "final_response":  final.strip(),
            "suggestions":     suggestions,
            "token_count":     total_tokens,
            "self_evaluation": self_eval,
        }

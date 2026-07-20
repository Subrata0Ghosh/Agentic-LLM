"""
Active Inference / Epistemic Foraging — src/brain/active_inference.py
======================================================================
Brain Region: Anterior Cingulate Cortex (ACC) + Hippocampus

Active Inference (Karl Friston) formalizes curiosity as an epistemic drive:
Curiosity = desire to perform actions that maximize INFORMATION GAIN
           = minimize prediction error most efficiently

Key difference from passive web search:
  Passive: "Search for the query text" (keyword matching)
  Active:  "Generate a targeted research QUESTION to resolve the biggest
            uncertainty in my world model, then search for THAT question"

The ACC monitors prediction errors and decides:
  1. How large is the knowledge gap?
  2. What is the most efficient question to fill it?
  3. Should I ask the USER for clarification instead of searching?
  4. After searching, did I actually reduce prediction error?

Dopamine modulates epistemic drive:
  High DA (surprise) → Strong curiosity drive, aggressive information seeking
  Low DA (familiar)  → No need to seek, prediction error already low
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from scraper import search_and_scrape


class ActiveInference:
    """
    Epistemic foraging agent — the brain's curiosity system.
    Generates targeted research questions and evaluates information gain.
    """

    def should_forage(self, error_magnitude: float, da_level: float, force_forage: bool = False) -> bool:
        """
        Decision: should we actively search for new information?
        Threshold modulated by current dopamine (DA) level.
        High DA (surprise) = lower threshold for foraging.
        """
        if force_forage:
            return True
        # High DA drives curiosity lower threshold
        threshold = 0.45 - (da_level - 0.35) * 0.3
        threshold = max(0.20, min(0.60, threshold))
        return error_magnitude > threshold

    def generate_research_question(self, original_query: str,
                                   prediction: dict,
                                   topics_in_world_model: list,
                                   is_math: bool = False) -> str:
        """
        Generates a targeted research question that addresses the specific
        gap in the world model, rather than just searching the raw query.

        This is the key innovation vs. naive keyword search:
        We synthesize what we DON'T know, then search for THAT.
        """
        if is_math:
            return f"{original_query} step by step math formula solution"

        known_topics = prediction.get("topics_found", [])
        confidence = prediction.get("confidence", 0.1)

        if confidence < 0.1:
            # Almost total ignorance — search the query directly but refined
            return f"{original_query} detailed explanation overview"

        elif known_topics:
            # Partial knowledge — search for the SPECIFIC GAP
            # e.g., we know about "machine learning" but query asks about "transformers"
            query_words = set(original_query.lower().split())
            known_words = set(" ".join(known_topics).lower().split())
            gap_words = query_words - known_words - {"the", "a", "is", "are", "what", "how"}
            if gap_words:
                gap_phrase = " ".join(list(gap_words)[:4])
                return f"{gap_phrase} {original_query}"
            else:
                return f"recent advances {original_query}"
        else:
            # No prior knowledge but partial context — straightforward enriched search
            return f"{original_query} comprehensive guide examples"

    def forage(self, session_id: str, original_query: str, prediction: dict,
               dopamine: float, num_results: int = 5, force_forage: bool = False, is_math: bool = False) -> dict:
        """
        Execute epistemic foraging:
        1. Generate a targeted research question.
        2. Search and scrape.
        3. Estimate information gain.

        Returns:
            {
              "research_question": str
              "new_content":       str (scraped text)
              "info_gain":         float estimate
              "foraged":           bool
            }
        """
        if not self.should_forage(
            error_magnitude=1.0 - prediction.get("confidence", 0.5),
            da_level=dopamine,
            force_forage=force_forage
        ):
            return {"foraged": False, "new_content": "", "info_gain": 0.0,
                    "research_question": ""}

        research_q = self.generate_research_question(
            original_query=original_query,
            prediction=prediction,
            topics_in_world_model=prediction.get("topics_found", []),
            is_math=is_math
        )

        print(f"[ActiveInference] Foraging with targeted question: '{research_q}'")

        try:
            new_content = search_and_scrape(research_q, num_results=num_results)
        except Exception as e:
            print(f"[ActiveInference] Foraging failed: {e}")
            new_content = ""

        # Estimate information gain by content richness
        info_gain = min(1.0, len(new_content.split()) / 500.0) if new_content else 0.0

        return {
            "research_question": research_q,
            "new_content":       new_content,
            "info_gain":         round(info_gain, 3),
            "foraged":           bool(new_content),
        }

    def needs_user_clarification(self, error_magnitude: float,
                                  info_gain: float, foraged: bool) -> bool:
        """
        Determines if the knowledge gap is too large even after foraging.
        In that case, the agent should ask the USER for clarification
        (just as humans say "can you clarify what you mean by X?").
        """
        if not foraged:
            return error_magnitude > 0.85
        return error_magnitude > 0.7 and info_gain < 0.15

    def generate_clarification_question(self, original_query: str,
                                         prediction: dict) -> str:
        """
        Generates a specific clarification question targeted at the most
        ambiguous aspect of the original query.
        """
        topics = prediction.get("topics_found", [])
        if topics:
            return (f"I want to give you the most accurate answer. "
                    f"Could you clarify what specifically about '{topics[0]}' "
                    f"you're asking? For example, are you asking about "
                    f"practical usage, theory, comparison, or something else?")
        else:
            return (f"I want to make sure I understand your question correctly. "
                    f"Could you give me a bit more context about what you're "
                    f"trying to learn or solve with: \"{original_query}\"?")

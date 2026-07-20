"""
Autonomous Dreamer — src/brain/autonomous_dreamer.py
=====================================================
Brain Region: Default Mode Network (during idle/sleep)

During sleep and rest, the human brain doesn't go dark.
The Default Mode Network INCREASES activity during idle periods.
This is when the brain:
  - Replays experiences (memory consolidation)
  - Makes unexpected connections between memories
  - Generates creative insights ("aha moments" after sleeping on a problem)
  - Imagines future scenarios (mental time travel)

The Autonomous Dreamer runs in the BACKGROUND as a server thread.
When the system is idle (no new queries), it:
  1. Samples 3 random high-priority hippocampal memories
  2. Runs a mini REM-style synthesis on them
  3. Stores the resulting novel associations back into hippocampus
  4. Strengthens related semantic graph edges (Hebbian)

This means: the longer the system runs without being queried,
the MORE it learns and connects ideas — like a human brain at rest.
"""

import time
import threading
import json
import os
import sys
import random

DREAMER_STATE_FILE = "data/dreamer_state.json"
IDLE_THRESHOLD_SECS = 120   # Start dreaming after 2 minutes idle
DREAM_INTERVAL_SECS = 300   # Dream every 5 minutes while idle


class AutonomousDreamer:
    """
    Background daemon that generates novel associations during idle time.
    """

    def __init__(self, hippocampus, semantic_graph, generate_fn,
                  state_file: str = DREAMER_STATE_FILE):
        self.hippocampus    = hippocampus
        self.semantic_graph = semantic_graph
        self.generate       = generate_fn
        self.state_file     = state_file

        self._last_activity = time.time()
        self._running       = False
        self._thread        = None
        self._dream_count   = 0
        self._paused        = False

        os.makedirs("data", exist_ok=True)
        self._load_state()

    def _load_state(self):
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self._dream_count = state.get("dream_count", 0)
        except Exception:
            pass

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump({"dream_count": self._dream_count,
                       "last_dream": time.time()}, f)

    def notify_activity(self):
        """Call this every time a user query arrives to reset idle timer."""
        self._last_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_activity) > IDLE_THRESHOLD_SECS

    def _run_dream_session(self):
        """
        Single dream session: sample memories, synthesize associations,
        store insights back into hippocampus.
        """
        # Sample high-priority memories
        candidates = self.hippocampus.get_high_priority_memories(top_n=20)
        if len(candidates) < 2:
            return

        # Randomly select 3 memories for cross-association
        sample = random.sample(candidates, min(3, len(candidates)))
        memory_texts = "\n\n".join([
            f"Fragment {i+1}: {m['content'][:200]}"
            for i, m in enumerate(sample)
        ])

        system = (
            "You are the brain's dreaming process — free-associating between memory fragments. "
            "Generate ONE surprising, novel insight or connection between these fragments. "
            "Be creative. Think laterally. One insight only, 2-3 sentences."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Memory fragments:\n\n{memory_texts}\n\nNovel insight:"}
        ]

        try:
            insight = self.generate(
                messages,
                max_new_tokens=100,
                temperature=0.9,
                top_p=0.97,
                repetition_penalty=1.05
            )

            if insight and len(insight.split()) > 5:
                # Store the dream insight as a new hippocampal memory
                self.hippocampus.encode(
                    content=f"[Dream Insight]: {insight}",
                    session_id="dreamer",
                    source="autonomous_dream",
                    emotion_tag={"valence": "positive", "arousal": 0.2, "priority": 0.3},
                    ach_level=0.5,
                )

                # Strengthen semantic graph with Hebbian edges from the dream
                dream_words = [w.lower() for w in insight.split()
                               if len(w) > 4 and w.isalpha()][:10]
                for i in range(len(dream_words) - 1):
                    self.semantic_graph.add_hebbian_edge(
                        dream_words[i], dream_words[i+1],
                        co_occurrence_strength=0.15
                    )

                self._dream_count += 1
                self._save_state()
                print(f"[Dreamer] Dream #{self._dream_count}: {insight[:80]}...")

        except Exception as e:
            print(f"[Dreamer] Dream synthesis failed: {e}")

    def _dream_loop(self):
        """Background thread loop."""
        while self._running:
            time.sleep(30)  # Check every 30 seconds
            if self._paused or not self._running:
                continue
            if self._is_idle():
                print(f"[Dreamer] Idle detected. Starting dream session #{self._dream_count + 1}...")
                self._run_dream_session()
                time.sleep(DREAM_INTERVAL_SECS)

    def start(self):
        """Start the autonomous dreamer background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._dream_loop,
                                         daemon=True, name="AutonomousDreamer")
        self._thread.start()
        print("[Dreamer] Autonomous dreamer started.")

    def stop(self):
        self._running = False
        print("[Dreamer] Autonomous dreamer stopped.")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def get_status(self) -> dict:
        idle_secs = round(time.time() - self._last_activity, 1)
        return {
            "running":      self._running,
            "paused":       self._paused,
            "dream_count":  self._dream_count,
            "idle_seconds": idle_secs,
            "is_dreaming":  self._is_idle() and self._running and not self._paused,
        }

"""
BICA API — src/api/api.py
==========================
Biologically-Inspired Cognitive Architecture — Full Brain Pipeline v2

Request processing order follows human brain anatomy:

  1.  THALAMUS          → Salience scoring & routing decision
  2.  NEUROMODULATORS   → Tick (decay) + state retrieval
  3.  AMYGDALA          → Emotional tagging of the input
  4.  THEORY OF MIND    → Calibrate for user expertise/emotional state
  5.  PROCEDURAL MEMORY → Fast habit check (bypass deep pipeline if hit)
  6.  WORKING MEMORY    → Context assembly with NE-modulated attention
  7.  PREDICTIVE CORTEX → World model prediction + prediction error
  8.  HIPPOCAMPUS       → Contextual episodic memory retrieval (ACh-gated)
  9.  SEMANTIC GRAPH    → Multi-hop relational knowledge search
  10. ACTIVE INFERENCE  → Epistemic foraging if prediction error > threshold
  11. DEFAULT MODE NET  → 3-stage simulation (Recall → Simulate → Critique)
  12. AMYGDALA (out)    → Tag response, adjust SE
  13. NEUROMODULATORS   → Emit DA signal, update all modulators
  14. PREDICTIVE CORTEX → Update world model
  15. HIPPOCAMPUS       → Encode new memory (ACh-gated)
  16. HEBBIAN           → Strengthen co-occurring concept edges
  17. EPISODIC TIMELINE → Record event on temporal timeline
  18. METACOGNITION     → Check cognitive load, detect confusion, auto-sleep
  19. PROCEDURAL MEMORY → Observe for habit formation
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import sys
import os
import uuid

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(_HERE, '..', 'core'))
sys.path.append(os.path.join(_HERE, '..', 'brain'))

# ── Core imports ──────────────────────────────────────────────────────────────
from memory import add_to_memory, load_memory
from researcher import research_topic
from scraper import search_and_scrape

# ── Brain module imports ──────────────────────────────────────────────────────
from neuromodulators    import NeuromodulatorSystem
from thalamus           import Thalamus
from amygdala           import Amygdala
from working_memory     import WorkingMemory
from hippocampus        import Hippocampus
from predictive_cortex  import PredictiveCortex
from active_inference   import ActiveInference
from default_mode       import DefaultModeNetwork
from sleep_cycle        import SleepCycle
from procedural_memory  import ProceduralMemory
from neocortex          import Neocortex

# ── Power Upgrade modules ──────────────────────────────────────────────────────
from reward_system      import RewardSystem
from episodic_timeline  import EpisodicTimeline
from metacognition      import Metacognition
from semantic_graph     import SemanticGraph
from hebbian            import HebbianPlasticity
from theory_of_mind     import TheoryOfMind
from visual_cortex      import VisualCortex
from autonomous_dreamer import AutonomousDreamer
from generate           import generate_text_api

# ── Cognitive Capabilities modules ─────────────────────────────────────────────
from parietal_lobe        import ParietalLobe
from planning_pfc         import PlanningPFC
from cognitive_flexibility import CognitiveFlexibility

# ── BICA v3 New Modules ────────────────────────────────────────────────────────
from bica_cognition import BICACognition
from cerebellum     import Cerebellum
from basal_ganglia  import BasalGanglia

# ── Brain singleton instances ─────────────────────────────────────────────────
nm    = NeuromodulatorSystem()
thal  = Thalamus()
amyg  = Amygdala()
wm    = WorkingMemory()
hipp  = Hippocampus()
pc    = PredictiveCortex()
ai    = ActiveInference()
dmn   = DefaultModeNetwork()
slp   = SleepCycle()
proc  = ProceduralMemory()
neo   = Neocortex()

# Power upgrade singletons
rwd   = RewardSystem()
tline = EpisodicTimeline()
mcog  = Metacognition()
sgraph = SemanticGraph()
hebb  = HebbianPlasticity(sgraph)
tom   = TheoryOfMind()
vcx   = VisualCortex()
dreamer = AutonomousDreamer(hipp, sgraph, generate_text_api)

# Cognitive Capabilities singletons
parietal = ParietalLobe()
plan_pfc = PlanningPFC()
flex     = CognitiveFlexibility()

# BICA v3 singletons
bica  = BICACognition()
cbllm = Cerebellum()
bg    = BasalGanglia()

# Correction Learning
from correction_learning import CorrectionLearning
cl    = CorrectionLearning()

# ── Lifespan for Background Dreamer ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start autonomous dreamer thread
    if "SPACE_ID" not in os.environ:
        dreamer.start()
    else:
        print("[Dreamer] Running on Hugging Face Spaces - background dreamer thread disabled to comply with ZeroGPU lifecycle.")
    yield
    # Stop autonomous dreamer thread
    if "SPACE_ID" not in os.environ:
        dreamer.stop()

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="BICA — Biologically-Inspired Cognitive Architecture", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="static"), name="static")
background_jobs = {}


# ── Request Models ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str
    max_tokens: int = 200

class LearnRequest(BaseModel):
    topic: str

class SleepRequest(BaseModel):
    session_id: str
    max_iters: int = 20

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    positive: bool
    topic: str = "general"
    response_snippet: str = ""

class VisionRequest(BaseModel):
    session_id: str
    image_b64: str
    candidate_labels: list = ["an object", "a person", "a screen with code", "a chart/diagram", "an empty desk"]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/home")
def read_index():
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.post("/bica/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    import traceback
    print(f"[api/chat] Received message: '{req.message}' for session {req.session_id}")
    try:
        return await _chat_endpoint_inner(req, background_tasks)
    except Exception as e:
        err_msg = f"Exception in chat_endpoint: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return {
            "response": f"⚠️ Internal Brain Error:\n```\n{err_msg}\n```",
            "suggestions": [],
            "path": "ERROR",
            "pipeline_trace": [],
            "brain_state": {},
            "monologue": {"stage1_recall": "Exception occurred.", "stage2_simulate": "", "stage3_critique": ""},
            "was_web_searched": False,
            "emotion_tag": {"valence": "neutral", "arousal": 0.0},
            "confidence": 0.0,
            "cognitive_load": 0.0,
            "self_verification": {"passed": False, "issues": [str(e)], "label": "❌ Error"},
            "cerebellum": {
                "query_type": "factual",
                "perseverating": False,
                "length_verdict": "ok",
            },
            "basal_ganglia": {
                "go":  True,
                "gate": "pass",
            },
            "bica_cognition": {
                "goal":         "error",
                "entities":     [],
                "is_question":  False,
                "multi_step":   False,
                "has_rules":    False,
                "contradiction": False,
            },
        }

async def _chat_endpoint_inner(req: ChatRequest, background_tasks: BackgroundTasks):
    sid = req.session_id
    msg = req.message
    pipeline_trace = []

    # Notify dreamer of user activity to reset idle timer
    dreamer.notify_activity()

    # ── Log to simple memory for chat history ────────────────────────────────
    add_to_memory(sid, "user", msg)
    wm.extract_and_store_variables(sid, msg)

    # ── Correction Learning Loop ─────────────────────────────────────────────
    cl_res = cl.process_message(sid, msg, hipp, neo, nm)
    if cl_res:
        add_to_memory(sid, "ai", cl_res["response"])
        tline.record_event(sid, "ai", cl_res["response"], topic="correction_learning", emotion="neutral")
        return {
            "response":        cl_res["response"],
            "suggestions":     cl_res.get("suggestions", []),
            "path":            "LEARNING_CORRECTION",
            "pipeline_trace":  cl_res.get("pipeline_trace", []),
            "brain_state":     nm.get_generation_params(sid)["raw"],
            "monologue":       cl_res.get("monologue", {"stage1_recall": "Correction dialogue active", "stage2_simulate": "", "stage3_critique": ""}),
            "was_web_searched": cl_res.get("was_web_searched", False),
            "emotion_tag":     {"valence": "neutral", "arousal": 0.2},
            "confidence":      0.9,
            "cognitive_load":    0.1,
            "auto_sleep_triggered": False,
            "active_strategy":   "learning_correction",
            "plan_status":       None,
            "curr_step_goal":    None,
            "self_verification": {"passed": True, "issues": [], "label": "✓ Learning Mode"},
            "cerebellum": {
                "query_type":    "factual",
                "perseverating": False,
                "length_verdict": "ok",
            },
            "basal_ganglia": {
                "go":  True,
                "gate": "pass",
            },
            "bica_cognition": {
                "goal":         "correction_learning",
                "entities":     [],
                "is_question":  False,
                "multi_step":   False,
                "has_rules":    False,
                "contradiction": False,
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — THALAMUS: Salience gating & routing
    # ══════════════════════════════════════════════════════════════════════════
    nm_state = nm.get_state(sid)
    nm.tick(sid)  # Each turn, modulators decay toward baseline

    # ── Basal Ganglia emergency stop (STN hyperdirect path) ───────────────────
    emergency_msg = bg.emergency_stop(msg)
    if emergency_msg:
        add_to_memory(sid, "ai", emergency_msg)
        return {
            "response":        emergency_msg,
            "suggestions":     [],
            "path":            "EMERGENCY",
            "pipeline_trace":  [{"region": "BasalGanglia", "emergency_stop": True}],
            "brain_state":     nm_state,
            "monologue":       {"stage1": "Emergency stop triggered.", "stage2": "", "stage3": ""},
            "was_web_searched": False,
            "emotion_tag":     {"valence": "urgent", "arousal": 1.0, "priority": 1.0},
            "confidence":      1.0,
            "self_verification": {"passed": True, "issues": [], "label": "\u2713 Safety Override"},
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — AMYGDALA: Emotional tag of input (before routing for salience)
    # ══════════════════════════════════════════════════════════════════════════
    emotion_tag = amyg.tag(msg, serotonin_level=nm_state.get("serotonin", 0.6))
    pipeline_trace.append({"region": "Amygdala", "valence": emotion_tag["valence"],
                            "arousal": emotion_tag["arousal"]})

    # Route with emotion_tag so emotional_load feeds into thalamic salience
    path, salience = thal.route(msg, nm_state=nm_state, emotion_tag=emotion_tag)
    pipeline_trace.insert(0, {"region": "Thalamus", "path": path,
                               "salience": salience["salience"],
                               "emotional_load": salience.get("emotional_load", 0.0)})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — THEORY OF MIND (Observe query)
    # ══════════════════════════════════════════════════════════════════════════
    tom.observe_query(sid, msg)
    tom_profile = tom.get_generation_profile(sid)
    pipeline_trace.append({"region": "TheoryOfMind", "user_expertise": tom_profile["expertise"],
                            "user_emotional_state": tom_profile["emotional_state"]})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3.5 — CEREBELLUM: Timing prediction + query classification
    # ══════════════════════════════════════════════════════════════════════════
    timing_prediction = cbllm.predict_response_length(msg)
    pipeline_trace.append({"region": "Cerebellum", "query_type": timing_prediction["query_type"],
                            "expected_words": f"{timing_prediction['min_words']}-{timing_prediction['max_words']}"}) 
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — PROCEDURAL MEMORY: Habit fast-path check
    # ══════════════════════════════════════════════════════════════════════════
    habit = proc.check_habit(msg)
    if habit and path == "REFLEX":
        # Ultra-fast path: return cached habit response
        pipeline_trace.append({"region": "ProceduralMemory", "hit": True,
                                "strength": habit["strength"]})
        nm.on_familiar_input(sid)
        add_to_memory(sid, "ai", habit["cached_response"])
        wm.add_turn(sid, "user", msg, emotion_tag, dopamine=nm_state.get("dopamine", 0.35))
        wm.add_turn(sid, "assistant", habit["cached_response"], {"valence": "neutral", "arousal": 0.1, "priority": 0.1}, dopamine=nm_state.get("dopamine", 0.35))
        gen_params = nm.get_generation_params(sid)
        tline.record_event(sid, "ai", habit["cached_response"], topic="reflex_habit", emotion="neutral")
        return {
            "response":        habit["cached_response"],
            "suggestions":     [],
            "path":            "HABIT",
            "pipeline_trace":  pipeline_trace,
            "brain_state":     gen_params["raw"],
            "monologue":       {"stage1": "Habit memory activated.", "stage2": "", "stage3": ""},
            "was_web_searched": False,
            "emotion_tag":     emotion_tag,
            "confidence":      0.95,
        }

    pipeline_trace.append({"region": "ProceduralMemory", "hit": False})

    # ── REFLEX (no habit) — handle greetings and brief follow-ups ────────────
    if path == "REFLEX":
        nm.on_familiar_input(sid)
        gen_params = nm.get_generation_params(sid)
        # Combine default tone with Theory of Mind calibration
        tone = tom_profile.get("tone", gen_params.get("tone", "friendly and clear"))
        
        # Assemble working memory context to preserve conversation history
        wm_context = wm.assemble_context(sid, norepinephrine=nm_state.get("norepinephrine", 0.5))
        
        from generate import generate_text_api
        msgs = [
            {"role": "system", "content": f"You are a {tone} AI assistant. Reply naturally and briefly. {tom_profile['depth']}"}
        ]
        # Insert working memory context (up to last 2 turns)
        for c in wm_context[-2:]:
            msgs.append({"role": c["role"], "content": c["content"]})
        # Append current user message
        msgs.append({"role": "user", "content": msg})
        
        response = generate_text_api(msgs, max_new_tokens=80,
                                      temperature=gen_params["temperature"],
                                      top_p=gen_params["top_p"],
                                      repetition_penalty=gen_params["repetition_penalty"])
        # Cerebellum perseveration check on REFLEX responses
        persev = cbllm.detect_perseveration(sid, response)
        if persev["is_perseverating"]:
            response += " (Let me rephrase that for variety.)"
        add_to_memory(sid, "ai", response)
        wm.add_turn(sid, "user", msg, emotion_tag, dopamine=nm_state.get("dopamine", 0.35))
        wm.add_turn(sid, "assistant", response, {"valence": "neutral", "arousal": 0.15, "priority": 0.1}, dopamine=nm_state.get("dopamine", 0.35))
        proc.observe(msg, response, dopamine=nm_state.get("dopamine", 0.35))
        tline.record_event(sid, "ai", response, topic="reflex_greeting", emotion="neutral")
        return {
            "response":        response,
            "suggestions":     [],
            "path":            "REFLEX",
            "pipeline_trace":  pipeline_trace,
            "brain_state":     nm_state,
            "monologue":       {"stage1": "Simple greeting detected.", "stage2": "", "stage3": ""},
            "was_web_searched": False,
            "emotion_tag":     emotion_tag,
            "confidence":      0.90,
            "self_verification": {"passed": True, "issues": [], "label": "✓ Verified"},
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4.5 — COGNITIVE FLEXIBILITY: Strategy Shift Check
    # ══════════════════════════════════════════════════════════════════════════
    flex_result = flex.evaluate_strategy(sid, msg, nm_state.get("prediction_error", 0.0), False)
    if flex_result["strategy_shifted"]:
        nm.update_state_custom(sid, {
            "norepinephrine": min(1.0, nm_state.get("norepinephrine", 0.5) + flex_result["ne_boost"]),
            "dopamine": min(1.0, nm_state.get("dopamine", 0.35) + flex_result["da_boost"])
        })
        nm_state = nm.get_state(sid) # Refresh state after boost
    pipeline_trace.append({"region": "CognitiveFlexibility", "strategy": flex_result["active_strategy"]})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4.6 — PARIETAL LOBE: Mathematical & Symbolic Gating
    # ══════════════════════════════════════════════════════════════════════════
    math_context = ""
    is_pure_math = False
    math_failed = False
    if parietal.is_math_query(msg):
        math_res = parietal.solve(msg)
        pipeline_trace.append({"region": "ParietalLobe", "parsed": math_res["success"]})
        if math_res["success"]:
            math_context = f"### PARIETAL LOBE MATH VERIFICATION ###\nResult: {math_res['result']}\nSteps:\n" + "\n".join(math_res["steps"])
            # If query is purely arithmetic or a simple equation, fast-path return the calculation
            if len(msg.split()) <= 6 and not any(w in msg.lower() for w in ["explain", "why", "detail"]):
                is_pure_math = True
                add_to_memory(sid, "ai", f"Result: {math_res['result']}")
                tline.record_event(sid, "ai", f"Math Result: {math_res['result']}", topic="math", emotion="neutral")
                return {
                    "response":        f"Result: {math_res['result']}\n\nCalculation steps:\n" + "\n".join(math_res["steps"]),
                    "suggestions":     [],
                    "path":            "MATHEMATICAL",
                    "pipeline_trace":  pipeline_trace,
                    "brain_state":     nm.get_generation_params(sid)["raw"],
                    "monologue":       {"stage1": "Parietal Lobe symbolic solving.", "stage2": "", "stage3": ""},
                    "was_web_searched": False,
                    "emotion_tag":     emotion_tag,
                    "confidence":      0.99,
                }
        else:
            math_failed = True

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4.7 — DL-PFC: Goal planning & Sequential Step Tracker
    # ══════════════════════════════════════════════════════════════════════════
    active_plan = plan_pfc.get_active_plan(sid)
    if not active_plan and plan_pfc.is_complex_goal(msg):
        active_plan = plan_pfc.create_plan(sid, msg)
        pipeline_trace.append({"region": "PlanningPFC", "new_plan": True, "steps_count": len(active_plan["steps"])})
    elif active_plan and active_plan["status"] == "running":
        pipeline_trace.append({"region": "PlanningPFC", "active_plan": True, "current_step": active_plan["current_step_idx"] + 1})

    plan_context = ""
    if active_plan and active_plan["status"] == "running":
        curr_step_idx = active_plan["current_step_idx"]
        curr_step = active_plan["steps"][curr_step_idx]
        wm.set_goal(sid, active_plan["original_query"], active_plan["steps"])
        plan_context = f"### ACTIVE PLAN SEQUENCE STEP ###\nStep {curr_step_idx + 1} of {len(active_plan['steps'])}: {curr_step['goal']}"

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — WORKING MEMORY: Assemble focused context
    # ══════════════════════════════════════════════════════════════════════════
    wm.inhibit_irrelevant(sid, msg)
    wm_context = wm.assemble_context(sid, norepinephrine=nm_state.get("norepinephrine", 0.5))
    pipeline_trace.append({"region": "WorkingMemory", "chunks_in_wm": len(wm_context)})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5.5 — BICA COGNITION: Perception, Attention, Semantic Reason
    # ══════════════════════════════════════════════════════════════════════════
    # Pre-retrieve for BICA semantic reasoning (lightweight, no ACh gate)
    pre_context, _ = hipp.get_context_string(msg, session_id=sid, ach_level=0.5, top_k=3)
    bica_state = bica.run(
        session_id=sid,
        query=msg,
        wm_context=wm_context,
        retrieved_context=pre_context,
    )
    # Feed BICA attention into WM focal concepts
    wm.load_bica_attention(sid, bica_state["attention"]["focal_concepts"])
    pipeline_trace.append({
        "region":       "BICACognition",
        "goal":         bica_state["perception"].get("primary_goal"),
        "entities":     bica_state["perception"]["entities"][:3],
        "has_rules":    bica_state["user_rules"]["has_rules"],
        "contradiction":bica_state["semantic"]["has_contradiction"],
    })

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — PREDICTIVE CORTEX: Generate world-model prediction
    # ══════════════════════════════════════════════════════════════════════════
    ach = nm_state.get("acetylcholine", 0.55)
    retrieved_context, has_strong_match = hipp.get_context_string(
        msg, session_id=sid, ach_level=ach
    )
    prediction = pc.predict(sid, msg, retrieved_context)
    pipeline_trace.append({"region": "PredictiveCortex", "confidence": prediction["confidence"],
                            "topics_found": prediction["topics_found"]})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — SEMANTIC GRAPH CONTEXT
    # ══════════════════════════════════════════════════════════════════════════
    graph_context = sgraph.get_context_string(msg, hops=2)
    # Combine all contextual streams: episodic, semantic graph, parietal logic, and sequential plan targets
    combined_context = f"{retrieved_context}\n\n{graph_context}\n\n{math_context}\n\n{plan_context}".strip()
    pipeline_trace.append({"region": "SemanticGraph", "has_context": bool(graph_context)})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — ACTIVE INFERENCE: Epistemic foraging if needed
    # ══════════════════════════════════════════════════════════════════════════
    was_web_searched = False
    forage_result = ai.forage(
        session_id=sid,
        original_query=msg,
        prediction=prediction,
        dopamine=nm_state.get("dopamine", 0.35),
        num_results=5,
        force_forage=math_failed,
        is_math=parietal.is_math_query(msg)
    )

    if forage_result["foraged"]:
        was_web_searched = True
        new_content = forage_result["new_content"]
        pipeline_trace.append({
            "region": "ActiveInference",
            "foraged": True,
            "research_q": forage_result["research_question"],
            "info_gain": forage_result["info_gain"]
        })

        # Store the new knowledge in hippocampus with ACh-gated encoding
        nm.on_learning_event(sid)
        updated_ach = nm.get_state(sid).get("acetylcholine", 0.55)
        hipp.encode(
            content=new_content,
            session_id=sid,
            source=f"web_search: {forage_result['research_question'][:40]}",
            emotion_tag=emotion_tag,
            ach_level=updated_ach,
        )

        # Re-retrieve with richer context
        retrieved_context, has_strong_match = hipp.get_context_string(
            msg, session_id=sid, ach_level=updated_ach
        )
        combined_context = f"{retrieved_context}\n\n{graph_context}\n\n{math_context}\n\n{plan_context}".strip()
    else:
        pipeline_trace.append({"region": "ActiveInference", "foraged": False})

    # Check if user clarification is needed
    if ai.needs_user_clarification(
        error_magnitude=1.0 - prediction["confidence"],
        info_gain=forage_result.get("info_gain", 0.0),
        foraged=forage_result["foraged"]
    ):
        clarify_q = ai.generate_clarification_question(msg, prediction)
        add_to_memory(sid, "ai", clarify_q)
        wm.add_turn(sid, "user", msg, emotion_tag, dopamine=nm_state.get("dopamine", 0.35))
        wm.add_turn(sid, "assistant", clarify_q, dopamine=nm_state.get("dopamine", 0.35))
        tline.record_event(sid, "ai", clarify_q, topic="clarification", emotion="neutral")
        return {
            "response":        clarify_q,
            "suggestions":     [],
            "path":            "CLARIFICATION",
            "pipeline_trace":  pipeline_trace,
            "brain_state":     nm.get_generation_params(sid)["raw"],
            "monologue":       {"stage1": "Knowledge gap too large. Requesting clarification.", "stage2": "", "stage3": ""},
            "was_web_searched": was_web_searched,
            "emotion_tag":     emotion_tag,
            "confidence":      0.50,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — DEFAULT MODE NETWORK: 3-stage cognitive simulation
    # ══════════════════════════════════════════════════════════════════════════
    gen_params = nm.get_generation_params(sid)
    # Merge Theory of Mind preferences directly into gen_params
    gen_params["tone"] = tom_profile["tone"]
    # Pass depth/strategy as gen_params so DMN injects them in system prompt (not user_message)
    gen_params["depth"] = tom_profile["depth"]
    gen_params["active_strategy"] = flex_result["active_strategy"]
    dmn_result = dmn.simulate(
        session_id=sid,
        user_message=msg,
        hippocampal_context=combined_context,
        wm_context=wm_context,
        world_model_prediction=prediction,
        gen_params=gen_params,
        bica_state=bica_state,
        path=path,
    )
    final_response = dmn_result["final_response"]
    # Safety net: never return an empty response to the frontend
    if not final_response or not final_response.strip():
        final_response = "I processed your message through the brain pipeline but couldn't synthesize a response. Please try again."
    self_verification = dmn_result.get("self_evaluation", {"passed": True, "issues": [], "label": "✓ Verified"})
    pipeline_trace.append({"region": "DMN",
                            "token_count": dmn_result["token_count"],
                            "verified": self_verification["passed"]})

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — POST-PROCESSING: Update all brain systems
    # ══════════════════════════════════════════════════════════════════════════

    # Compute prediction error & update neuromodulators
    error = pc.compute_error(sid, prediction, final_response, combined_context)
    nm.on_prediction_error(sid, error["error_magnitude"])

    # Update world model (perception side of active inference)
    pc.update_world_model(sid, msg, final_response)

    # Tag the response emotionally
    resp_tag = amyg.tag_response(final_response, serotonin_level=nm_state.get("serotonin", 0.6))
    new_se = amyg.adjust_se_on_response(nm_state.get("serotonin", 0.6), resp_tag)
    nm.on_social_interaction(sid, positive=resp_tag.get("positive_interaction", False))

    # Update working memory with this turn
    wm.add_turn(sid, "user", msg, emotion_tag, dopamine=nm_state.get("dopamine", 0.35))
    wm.add_turn(sid, "assistant", final_response, resp_tag,
                dopamine=nm_state.get("dopamine", 0.35))

    # Encode to hippocampus if this is new/important information
    if error["should_learn"]:
        final_ach = nm.get_state(sid).get("acetylcholine", 0.55)
        full_exchange = f"Q: {msg}\nA: {final_response}"
        hipp.encode(
            content=full_exchange,
            session_id=sid,
            source="conversation",
            emotion_tag=emotion_tag,
            ach_level=final_ach,
        )

    # Integrate into neocortical schema
    neo_matches = neo.recall(msg, top_k=2)
    if not neo_matches or error["should_learn"]:
        neo.integrate(msg[:40], final_response[:200], source="conversation")

    # Three-Factor Hebbian plasticity rule gated by Dopamine (DA), Acetylcholine (ACh), and Norepinephrine (NE):
    # ΔW = η * (DA * ACh * NE) * (ri * rj)
    da_factor = nm_state.get("dopamine", 0.35)
    ach_factor = nm_state.get("acetylcholine", 0.55)
    ne_factor = nm_state.get("norepinephrine", 0.50)
    co_strength = 0.8 * da_factor * ach_factor * ne_factor
    hebb.process_turn(f"{msg} {final_response}", co_strength=co_strength)

    # Record event on temporal timeline
    tline.record_event(sid, "user", msg, topic=prediction["topics_found"][0] if prediction["topics_found"] else "general", emotion=emotion_tag["valence"])
    tline.record_event(sid, "ai", final_response, topic=prediction["topics_found"][0] if prediction["topics_found"] else "general", emotion=resp_tag["valence"], priority=resp_tag.get("priority", 0.1))

    # Metacognition conflict check and statistics updating
    mcog_result = mcog.update(
        session_id=sid,
        prediction_error=error["error_magnitude"],
        wm_chunks=len(wm_context),
        ne_level=nm_state.get("norepinephrine", 0.5),
        reward_accuracy=rwd.get_accuracy(sid).get("session_accuracy")
    )
    pipeline_trace.append({
        "region": "Metacognition",
        "cognitive_load": mcog_result["cognitive_load"],
        "confidence": mcog_result["confidence"]
    })

    # Trigger Theory of Mind updates if metacog detects confusion
    if mcog_result["is_confused"]:
        tom.observe_confusion(sid)

    # Observe for habit formation
    proc.observe(msg, final_response, dopamine=nm_state.get("dopamine", 0.35))

    # Update persistent chat memory
    add_to_memory(sid, "ai", final_response)

    # Final brain state snapshot
    final_state = nm.get_generation_params(sid)

    # ── Cerebellum: post-response perseveration check + length validation ────
    persev_result  = cbllm.detect_perseveration(sid, final_response)
    length_verdict = cbllm.validate_response_length(msg, final_response)
    pipeline_trace.append({
        "region":             "Cerebellum",
        "perseverating":      persev_result["is_perseverating"],
        "length_verdict":     length_verdict["verdict"],
    })

    # ── Basal Ganglia: Go/No-Go gate on final response ───────────────────────
    gono = bg.check_go_nogo(final_response, is_perseverating=persev_result["is_perseverating"])
    bg.record_gate(sid, path, gono["go"], gono["suppressed_type"])
    # Update action Q-value based on prediction error (lower error = positive reward)
    reward_signal = 1.0 - error["error_magnitude"]
    bg.update_action_value(sid, path, reward=reward_signal)
    pipeline_trace.append({
        "region": "BasalGanglia",
        "go":     gono["go"],
        "gate":   gono["suppressed_type"] or "pass",
    })

    # If No-Go fired, append a brief note
    if not gono["go"] and gono["reason"]:
        final_response += f"\n\n[Note: {gono['reason']}]"

    # Encode completed plan summary to hippocampus
    if active_plan and active_plan["status"] == "running":
        advanced = plan_pfc.advance_step(sid)
        if advanced and advanced["status"] == "completed":
            plan_summary = plan_pfc.summarize_completed_plan(sid)
            if plan_summary:
                final_ach = nm.get_state(sid).get("acetylcholine", 0.55)
                hipp.encode(
                    content=plan_summary,
                    session_id=sid,
                    source="plan_completion",
                    emotion_tag=emotion_tag,
                    ach_level=final_ach,
                )

    # Auto-sleep triggered by high cognitive load or confusion
    if mcog_result["should_sleep"]:
        pipeline_trace.append({"region": "Metacognition", "triggered_auto_sleep": True})
        # Launch sleep cycle in background
        job_id = str(uuid.uuid4())
        background_jobs[job_id] = {
            "status": "running",
            "topic": "Auto-triggered Sleep",
            "message": "Metacognition triggered auto-sleep due to high cognitive load/confusion."
        }
        def auto_sleep_task(session_id: str, jid: str):
            try:
                slp.run_full_cycle(
                    session_id=session_id,
                    hippocampus=hipp,
                    neuromodulators=nm,
                    working_memory=wm,
                    gen_params=final_state,
                    max_finetune_iters=15
                )
                background_jobs[jid]["status"] = "completed"
                background_jobs[jid]["message"] = "Auto-triggered sleep consolidation completed successfully. Cognitive load cleared."
            except Exception as e:
                background_jobs[jid]["status"] = "failed"
                background_jobs[jid]["message"] = str(e)
        background_tasks.add_task(auto_sleep_task, sid, job_id)

    # Advance plan step sequence if running
    plan_status = None
    curr_step_goal = None
    if active_plan and active_plan["status"] == "running":
        advanced = plan_pfc.advance_step(sid)
        if advanced:
            plan_status = {
                "current_step": advanced["current_step_idx"] + 1,
                "total_steps": len(advanced["steps"]),
                "status": advanced["status"]
            }
            if advanced["status"] == "running":
                curr_step_goal = advanced["steps"][advanced["current_step_idx"]]["goal"]

    return {
        "response":        final_response,
        "suggestions":     dmn_result.get("suggestions", []),
        "path":            path,
        "pipeline_trace":  pipeline_trace,
        "brain_state":     final_state["raw"],
        "monologue": {
            "stage1_recall":   dmn_result.get("stage1_recall",   ""),
            "stage2_simulate": dmn_result.get("stage2_simulate", ""),
            "stage3_critique": dmn_result.get("stage3_critique", ""),
        },
        "prediction_error": error["error_magnitude"],
        "was_web_searched":  was_web_searched,
        "emotion_tag":       emotion_tag,
        "confidence":        mcog_result["confidence"],
        "cognitive_load":    mcog_result["cognitive_load"],
        "auto_sleep_triggered": mcog_result["should_sleep"],
        "active_strategy":   flex_result["active_strategy"],
        "plan_status":       plan_status,
        "curr_step_goal":    curr_step_goal,
        "self_verification": self_verification,
        "cerebellum": {
            "query_type":    timing_prediction["query_type"],
            "perseverating": persev_result["is_perseverating"],
            "length_verdict":length_verdict["verdict"],
        },
        "basal_ganglia": {
            "go":  gono["go"],
            "gate":gono["suppressed_type"] or "pass",
        },
        "bica_cognition": {
            "goal":         bica_state["perception"].get("primary_goal"),
            "entities":     bica_state["perception"]["entities"][:5],
            "is_question":  bica_state["perception"]["is_question"],
            "multi_step":   bica_state["perception"]["multi_step"],
            "has_rules":    bica_state["user_rules"]["has_rules"],
            "contradiction":bica_state["semantic"]["has_contradiction"],
        },
    }


# ── Sleep Endpoint ─────────────────────────────────────────────────────────────
@app.post("/bica/sleep")
async def sleep_endpoint(req: SleepRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = {
        "status": "running",
        "topic": "Sleep Consolidation",
        "message": "Initializing 4-phase sleep cycle..."
    }

    def sleep_task(session_id: str, max_iters: int, jid: str):
        try:
            gen_params = nm.get_generation_params(session_id)
            summary = slp.run_full_cycle(
                session_id=session_id,
                hippocampus=hipp,
                neuromodulators=nm,
                working_memory=wm,
                gen_params=gen_params,
                max_finetune_iters=max_iters,
            )
            # Apply Hebbian synaptic decay during sleep
            hebb.apply_decay(decay_factor=0.005)
            background_jobs[jid]["status"]  = "completed"
            background_jobs[jid]["message"] = summary
        except Exception as e:
            background_jobs[jid]["status"]  = "failed"
            background_jobs[jid]["message"] = str(e)

    background_tasks.add_task(sleep_task, req.session_id, req.max_iters, job_id)
    return {"message": "4-phase sleep cycle started.", "job_id": job_id}


# ── Feedback (Reward System) Endpoint ──────────────────────────────────────────
@app.post("/bica/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    result = rwd.record_feedback(
        session_id=req.session_id,
        message_id=req.message_id,
        positive=req.positive,
        topic=req.topic,
        response_snippet=req.response_snippet
    )
    # Update Theory of Mind model with the feedback
    tom.observe_feedback(req.session_id, req.positive)
    # Apply neuromodulator delta directly
    nm.update_state_custom(req.session_id, {
        "dopamine": max(0.1, min(1.0, nm.get_state(req.session_id).get("dopamine", 0.35) + result["da_delta"])),
        "norepinephrine": max(0.1, min(1.0, nm.get_state(req.session_id).get("norepinephrine", 0.5) + result["ne_delta"]))
    })
    return {"message": "Feedback recorded", "result": result}


# ── Vision (Visual Cortex) Endpoint ─────────────────────────────────────────────
@app.post("/bica/vision")
async def vision_endpoint(req: VisionRequest):
    try:
        res = vcx.analyze(req.image_b64, candidate_labels=req.candidate_labels)
        # Encode description into hippocampus episodic memory
        nm.on_learning_event(req.session_id)
        ach = nm.get_state(req.session_id).get("acetylcholine", 0.55)
        hipp.encode(
            content=res["description"],
            session_id=req.session_id,
            source="vision_stream",
            emotion_tag={"valence": "neutral", "arousal": 0.3, "priority": 0.4},
            ach_level=ach
        )
        tline.record_event(req.session_id, "user", "[Uploaded Image]", topic="vision", emotion="neutral")
        tline.record_event(req.session_id, "ai", f"Visual perception encoded: {res['description']}", topic="vision", emotion="neutral")
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Vision analysis failed: {str(e)}")


# ── Episodic Timeline Endpoint ─────────────────────────────────────────────────
@app.get("/bica/timeline/{session_id}")
async def timeline_endpoint(session_id: str):
    events = tline.get_timeline_for_ui(session_id)
    summary = tline.get_session_summary(session_id)
    return {"events": events, "summary": summary}


# ── Theory of Mind Endpoint ────────────────────────────────────────────────────
@app.get("/bica/user_model/{session_id}")
async def user_model_endpoint(session_id: str):
    profile = tom.get_generation_profile(session_id)
    summary = tom.get_user_summary(session_id)
    return {"profile": profile, "summary": summary}


# ── Autonomous Dreamer Endpoint ────────────────────────────────────────────────
@app.get("/bica/dreamer")
async def dreamer_endpoint():
    return dreamer.get_status()

@app.post("/bica/dreamer/toggle")
async def dreamer_toggle_endpoint():
    status = dreamer.get_status()
    if status["paused"]:
        dreamer.resume()
        return {"message": "Dreamer resumed", "paused": False}
    else:
        dreamer.pause()
        return {"message": "Dreamer paused", "paused": True}


# ── Metacognition Endpoint ─────────────────────────────────────────────────────
@app.get("/bica/metacog/{session_id}")
async def metacog_endpoint(session_id: str):
    return mcog.get_stats(session_id)


# ── Brain Stats Endpoint ───────────────────────────────────────────────────────
@app.get("/bica/brain_stats/{session_id}")
async def brain_stats_endpoint(session_id: str):
    nm_state = nm.get_state(session_id)
    gen_params = nm.get_generation_params(session_id)
    schema = neo.get_schema_summary()
    pc_schema = pc.get_schema_summary(session_id)
    habits = proc.get_all_habits()
    graph_stats = sgraph.get_stats()
    tom_profile = tom.get_user_summary(session_id)
    metacog_stats = mcog.get_stats(session_id)
    active_plan = plan_pfc.get_active_plan(session_id)
    
    return {
        "neuromodulators": nm_state,
        "generation_params": gen_params,
        "neocortex_schema": schema,
        "world_model": pc_schema,
        "habits": habits,
        "semantic_graph": graph_stats,
        "user_model": tom_profile,
        "metacognition": metacog_stats,
        "active_plan": active_plan,
        "active_strategy": flex.evaluate_strategy(session_id, "", 0.0, False)["active_strategy"]
    }


# ── Brain Map (live state) ─────────────────────────────────────────────────────
@app.get("/bica/brain_map/{session_id}")
async def brain_map(session_id: str):
    nm_s = nm.get_state(session_id)
    m_stats = mcog.get_stats(session_id)
    active_strat = flex.evaluate_strategy(session_id, "", 0.0, False)["active_strategy"]
    bg_stats = bg.get_stats(session_id)
    cb_stats = cbllm.get_stats(session_id)
    return {
        "regions": {
            "thalamus":        {"active": True, "label": "Sensory Filter"},
            "amygdala":        {"active": True, "label": "Emotional Tagger"},
            "hippocampus":     {"active": True, "label": "Episodic Memory"},
            "prefrontal_cortex":{"active": True, "label": "Working & Plan"},
            "dmn":             {"active": True, "label": "Imagination"},
            "neocortex":       {"active": True, "label": "Schema Store"},
            "cerebellum":      {"active": True, "label": "Timing & Sequence",
                                "perseveration_count": cb_stats.get("perseveration_count", 0)},
            "basal_ganglia":   {"active": True, "label": "Go / No-Go Gate",
                                "go_count":   bg_stats.get("go_count", 0),
                                "nogo_count": bg_stats.get("nogo_count", 0)},
            "acc":             {"active": True, "label": "Set-Shift & ACC"},
            "tpj":             {"active": True, "label": "Theory of Mind"},
            "parietal":        {"active": True, "label": "Parietal Math"},
            "bica_cognition":  {"active": True, "label": "BICA Cognitive Engine"},
        },
        "neuromodulators": {
            "dopamine":       round(nm_s.get("dopamine", 0.35), 3),
            "serotonin":      round(nm_s.get("serotonin", 0.60), 3),
            "norepinephrine": round(nm_s.get("norepinephrine", 0.50), 3),
            "acetylcholine":  round(nm_s.get("acetylcholine", 0.55), 3),
        },
        "cognitive_load":  m_stats.get("last_cog_load", 0.0),
        "confidence":      m_stats.get("last_confidence", 0.5),
        "active_strategy": active_strat,
    }


# ── Schema (Neocortex) Endpoint ────────────────────────────────────────────────
@app.get("/bica/schema")
async def schema_endpoint():
    return neo.get_schema_summary()


# ── Learn / Researcher ─────────────────────────────────────────────────────────
@app.post("/bica/learn")
async def learn_endpoint(req: LearnRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = {"status": "running", "topic": req.topic, "message": "Scraping..."}

    def learning_task(topic: str, jid: str):
        try:
            new_text = search_and_scrape(topic)
            if new_text:
                hipp.encode(
                    content=new_text,
                    session_id="global_learn",
                    source=f"learn:{topic[:30]}",
                    ach_level=0.85,  # Max encoding strength during deliberate learning
                )
                neo.bulk_integrate_from_sleep(new_text[:1000])
                background_jobs[jid]["status"]  = "completed"
                background_jobs[jid]["message"] = f"Learned and encoded: {topic}"
            else:
                background_jobs[jid]["status"]  = "failed"
                background_jobs[jid]["message"] = "No content found."
        except Exception as e:
            background_jobs[jid]["status"]  = "failed"
            background_jobs[jid]["message"] = str(e)

    background_tasks.add_task(learning_task, req.topic, job_id)
    return {"message": f"Learning '{req.topic}' with max encoding strength.", "job_id": job_id}


@app.post("/bica/read_paper")
async def read_paper_endpoint(req: LearnRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = {"status": "running", "topic": req.topic, "message": "Reading paper..."}

    def research_task(topic: str, jid: str):
        try:
            res_msg = research_topic(topic)
            background_jobs[jid]["status"]  = "completed"
            background_jobs[jid]["message"] = res_msg
        except Exception as e:
            background_jobs[jid]["status"]  = "failed"
            background_jobs[jid]["message"] = str(e)

    background_tasks.add_task(research_task, req.topic, job_id)
    return {"message": f"Researching '{req.topic}'.", "job_id": job_id}


@app.get("/bica/job_status/{job_id}")
async def job_status(job_id: str):
    job = background_jobs.get(job_id)
    return job if job else {"status": "not_found"}


@app.get("/bica/memory_inspector")
async def memory_inspector():
    sources = hipp.get_all_sources()
    schema = neo.get_schema_summary()
    return {"topics": sources, "schema": schema}


# ── BICA v3 New Endpoints ────────────────────────────────────────────────────────

@app.get("/bica/cognitive_state/{session_id}")
async def cognitive_state_endpoint(session_id: str):
    """
    Returns a unified snapshot of all 10 BICA cognitive module states.
    Used by the live UI panel for per-session cognitive monitoring.
    """
    bica_snapshot = bica.get_session_state(session_id) or {}
    mcog_stats = mcog.get_stats(session_id)
    wm_goal = wm.get_goal(session_id)
    focal = wm.get_bica_focal_concepts(session_id)
    active_plan = plan_pfc.get_active_plan(session_id)
    nm_state = nm.get_state(session_id)

    return {
        "session_id":   session_id,
        "perception":   bica_snapshot.get("perception", {}),
        "attention": {
            "focal_concepts":    focal,
            "focused_wm_chunks": bica_snapshot.get("attention", {}).get("focused_wm_chunks", []),
        },
        "working_memory": {
            "active_goal":     wm_goal.get("active_goal"),
            "spatial_pad":     wm.get_spatial_scratchpad(session_id),
        },
        "reasoning_mode":  flex.evaluate_strategy(session_id, "", 0.0, False)["active_strategy"],
        "plan_status":     active_plan,
        "user_rules":      bica_snapshot.get("user_rules", {}),
        "commonsense":     bica_snapshot.get("commonsense", {}),
        "contradiction":   bica_snapshot.get("semantic", {}).get("has_contradiction", False),
        "self_verification": bica_snapshot.get("last_verification", {"passed": True, "label": "No data yet"}),
        "confidence":      mcog_stats.get("last_confidence", 0.5),
        "cognitive_load":  mcog_stats.get("last_cog_load", 0.0),
        "neuromodulators": {k: round(nm_state.get(k, 0.0), 3)
                            for k in ["dopamine", "serotonin", "norepinephrine", "acetylcholine"]},
        "recovery_action": mcog.recovery_action(session_id),
    }


@app.get("/bica/basal_ganglia/{session_id}")
async def basal_ganglia_endpoint(session_id: str):
    """
    Returns Basal Ganglia Go/No-Go stats for a session.
    Includes action Q-values, suppression log, and gate counts.
    """
    return bg.get_stats(session_id)


@app.get("/bica/cerebellum/{session_id}")
async def cerebellum_endpoint(session_id: str):
    """
    Returns Cerebellum timing and perseveration stats for a session.
    """
    stats = cbllm.get_stats(session_id)
    return {
        **stats,
        "timing_map": {
            qtype: {"min": mn, "max": mx}
            for qtype, (mn, mx) in {
                "greeting": (10, 40), "math": (20, 60),
                "explanation": (80, 300), "comparison": (100, 400),
                "planning": (150, 500), "creative": (100, 400),
                "factual": (30, 150), "general": (50, 200),
            }.items()
        },
    }

@app.get("/bica/debug_log")
def debug_log():
    if os.path.exists("data/generation_debug.log"):
        with open("data/generation_debug.log", "r", encoding="utf-8") as f:
            return {"log": f.read()}
    return {"log": "No log file found."}

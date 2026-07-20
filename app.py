import sys
import os

# Try importing spaces for ZeroGPU support, with local fallback mock
try:
    import spaces
except ImportError:
    import types
    mock_spaces = types.ModuleType("spaces")
    def mock_gpu(f):
        return f
    mock_spaces.GPU = mock_gpu
    sys.modules["spaces"] = mock_spaces

import spaces
import nltk
import gradio as gr
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Set path relative to the root
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'brain'))

# Ensure NLTK datasets are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK: {e}")

# Monkey-patch gr.routes.App.create_app to inject BICA API routes and static files.
# This is necessary because demo.launch() creates the actual FastAPI app internally.
original_create_app = gr.routes.App.create_app

@classmethod
def custom_create_app(cls, *args, **kwargs):
    # Call the original create_app to get the Gradio FastAPI app
    app = original_create_app(*args, **kwargs)
    
    # Import BICA API components and register routes
    from src.api.api import app as bica_app, dreamer
    app.include_router(bica_app.router)
    
    # Mount static files for the BICA UI
    app.mount("/ui", StaticFiles(directory="static"), name="static")
    
    # Prioritize BICA routes (/api/, /ui, /home) over Gradio routes to prevent Gradio wildcard interception
    bica_prefixes = ("/bica/", "/bica", "/ui", "/home")
    bica_routes = []
    gradio_routes = []
    for r in app.routes:
        if hasattr(r, "path") and r.path.startswith(bica_prefixes):
            bica_routes.append(r)
        else:
            gradio_routes.append(r)
    app.router.routes.clear()
    app.router.routes.extend(bica_routes + gradio_routes)
    print(f"[app] Reordered routes: prioritized {len(bica_routes)} BICA routes over {len(gradio_routes)} Gradio routes.")
    
    # Add CORS middleware to match BICA original API setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Disable caching for BICA routes to force client update and bypass CDN/browser caches
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(("/ui", "/home", "/bica")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    # Register BICA dreamer lifecycle event handlers
    @app.on_event("startup")
    async def startup_event():
        if "SPACE_ID" not in os.environ:
            print("[Dreamer] Starting autonomous dreamer thread...")
            dreamer.start()
        else:
            print("[Dreamer] Running on Hugging Face Spaces - background dreamer thread disabled to comply with ZeroGPU lifecycle.")
        
    @app.on_event("shutdown")
    async def shutdown_event():
        if "SPACE_ID" not in os.environ:
            dreamer.stop()
        
    return app

gr.routes.App.create_app = custom_create_app


@spaces.GPU
def dummy_gpu_fn():
    """This function satisfies ZeroGPU event-handler scanning during startup."""
    pass


# ── Build Gradio demo ─────────────────────────────────────────────────────────
with gr.Blocks(title="BICA v3 Active", css="footer {visibility: hidden}") as demo:
    gr.HTML("<iframe src='/home' style='width: 100%; height: 950px; border: none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'></iframe>")

    # Hidden button to satisfy ZeroGPU static event-handler analyzer
    dummy_btn = gr.Button("GPU Init", visible=False)
    dummy_btn.click(fn=dummy_gpu_fn, inputs=[], outputs=[])

# Enable queue system, mandatory for Hugging Face ZeroGPU spaces
demo.queue()

# ── Launch ────────────────────────────────────────────────────────────────────
print("Launching Gradio/BICA server...")

on_hf_spaces = "SPACE_ID" in os.environ

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=on_hf_spaces,   # Required on HF Spaces when localhost is not accessible
)

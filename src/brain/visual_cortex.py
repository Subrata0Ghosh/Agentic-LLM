"""
Visual Cortex — src/brain/visual_cortex.py
============================================
Brain Region: Primary Visual Cortex (V1–V5) + Ventral Stream (object recognition)

The human brain processes visual information in ~100ms, creating
rich multi-modal memories where every experience has a visual component.

This module adds image understanding to the BICA system:
  - Accepts base64-encoded images via API
  - Extracts semantic meaning using CLIP (Contrastive Language-Image Pre-Training)
  - Creates a text description that gets encoded into hippocampal memory
  - Falls back gracefully to PIL-based analysis if CLIP is unavailable

How visual memories are stored:
  Hippocampus receives: "Visual memory: [description] at [timestamp]"
  This memory has emotion_tag from amygdala + ach_level encoding strength

Supported input formats:
  - base64-encoded JPEG/PNG (via API upload)
  - URL to an image (fetched and processed)
"""

import os
import base64
import io
import time
from typing import Optional, Tuple


class VisualCortex:
    """
    Multi-modal vision processing for episodic memory integration.
    """

    def __init__(self):
        self._clip_available = False
        self._clip_model = None
        self._clip_preprocess = None
        self._try_load_clip()

    def _try_load_clip(self):
        """Attempt to load CLIP model. Fail gracefully if not installed."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Force CPU on Hugging Face Spaces to avoid ZeroGPU thread allocation errors
            device = "cpu" if "SPACE_ID" in os.environ else None
            if device:
                print("[VisualCortex] Running on Hugging Face Spaces - forcing CPU execution")
                
            # Use CLIP-based sentence transformer (lightweight)
            self._st_model = SentenceTransformer("clip-ViT-B-32", device=device)
            self._clip_available = True
            print("[VisualCortex] CLIP model loaded successfully.")
        except ImportError:
            print("[VisualCortex] sentence-transformers with CLIP not available. "
                  "Using fallback text description mode.")
            self._clip_available = False
        except Exception as e:
            print(f"[VisualCortex] CLIP load failed: {e}. Using fallback.")
            self._clip_available = False

    def _decode_image(self, image_b64: str):
        """Decode base64 image to PIL Image."""
        try:
            from PIL import Image
            img_bytes = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Image decode failed: {e}")

    def _basic_image_analysis(self, image) -> str:
        """
        Fallback: basic image analysis without deep learning.
        Returns a structural description (size, dominant colors, etc.)
        """
        try:
            width, height = image.size
            aspect = "wide" if width > height * 1.2 else ("tall" if height > width * 1.2 else "square")

            # Sample center pixel for dominant color estimation
            img_small = image.resize((50, 50))
            pixels = list(img_small.getdata())
            avg_r = sum(p[0] for p in pixels) // len(pixels)
            avg_g = sum(p[1] for p in pixels) // len(pixels)
            avg_b = sum(p[2] for p in pixels) // len(pixels)

            brightness = (avg_r + avg_g + avg_b) // 3
            bright_desc = "bright" if brightness > 180 else ("dark" if brightness < 80 else "medium-brightness")

            # Dominant color heuristic
            if avg_r > avg_g and avg_r > avg_b:
                color = "warm/red-toned"
            elif avg_g > avg_r and avg_g > avg_b:
                color = "green-toned"
            elif avg_b > avg_r and avg_b > avg_g:
                color = "cool/blue-toned"
            else:
                color = "neutral/gray-toned"

            return (f"Image ({width}×{height}px, {aspect} format, {bright_desc}, {color}). "
                    f"No deep visual analysis available — install sentence-transformers for CLIP.")
        except Exception:
            return "Image received but could not be analyzed."

    def analyze(self, image_b64: str, candidate_labels: list = None) -> dict:
        """
        Main entry point: analyze an image and return a textual description
        suitable for hippocampal encoding.

        Args:
            image_b64:        Base64-encoded image string.
            candidate_labels: Optional list of labels to score against (CLIP zero-shot).

        Returns:
            {
              description: str   — human-readable description for memory encoding
              embedding:   list  — float vector (if CLIP available, else [])
              method:      str   — "clip" | "basic"
              confidence:  float
            }
        """
        image = self._decode_image(image_b64)

        if self._clip_available and candidate_labels:
            try:
                # CLIP zero-shot image classification
                img_embedding = self._st_model.encode(image)
                label_embeddings = self._st_model.encode(candidate_labels)

                # Cosine similarities
                import torch
                img_t = torch.tensor(img_embedding)
                lab_t = torch.tensor(label_embeddings)
                sims = torch.nn.functional.cosine_similarity(
                    img_t.unsqueeze(0), lab_t, dim=-1
                )
                best_idx = int(sims.argmax())
                best_label = candidate_labels[best_idx]
                best_conf = float(sims[best_idx])

                description = (f"Visual memory: {best_label} (confidence {best_conf:.2f}). "
                                f"Image dimensions: {image.size[0]}×{image.size[1]}px.")
                return {
                    "description": description,
                    "embedding":   img_embedding.tolist(),
                    "method":      "clip",
                    "confidence":  round(best_conf, 3),
                    "top_label":   best_label,
                }
            except Exception as e:
                print(f"[VisualCortex] CLIP analysis failed: {e}. Falling back.")

        # Fallback
        description = self._basic_image_analysis(image)
        return {
            "description": description,
            "embedding":   [],
            "method":      "basic",
            "confidence":  0.3,
        }

    def is_available(self) -> bool:
        return self._clip_available

    def get_status(self) -> dict:
        return {
            "clip_available": self._clip_available,
            "mode":           "clip" if self._clip_available else "basic_analysis",
        }

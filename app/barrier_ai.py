"""
AI-powered barrier detection for WiFi RF propagation modeling.

Tries Anthropic Claude first; falls back to Google Gemini if Claude fails or
its API key is not configured. Both providers receive the same prompt and
return wall segments in the caller's pixel coordinate space.

Requires at least one of:
  - anthropic package  (pip install anthropic)  + ANTHROPIC_API_KEY env var
  - google-generativeai (pip install google-generativeai) + GEMINI_API_KEY env var
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re

from PIL import Image

from .wall_analyzer import WallSegment

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Vision API recommended upper bound; stays within limits after JPEG encoding
_MAX_SIDE_PX = 1568
_MAX_BYTES_B64 = 4_800_000  # ~5 MB base64 limit

_SYSTEM = (
    "You are a specialized architectural analyst for WiFi RF (radio frequency) propagation "
    "modeling. You identify structural barriers in floor plans with high precision."
)

_USER = """Analyze this architectural floor plan and identify ALL walls and structural barriers that attenuate WiFi signals.

Return ONLY a raw JSON array — no markdown, no explanation, start with [ :
[{"start":{"x":5.0,"y":10.0},"end":{"x":95.0,"y":20.0},"material":"brick","confidence":0.85}]

Coordinates are percentages of image dimensions (0.0–100.0).

Material options (choose the best fit):
  concrete           – plain/unreinforced concrete
  reinforced_concrete – structural concrete with rebar (pillars, beams, load-bearing walls)
  brick              – ceramic brick / masonry (alvenaria, tijolo, bloco)
  drywall            – gypsum board / steel frame partition
  glass              – glazed facade, curtain wall, glass block
  wood               – timber frame, wood panel
  stone              – granite, marble, natural stone
  metal              – steel sheet, aluminium partition, metal stud

Brazilian NBR 6492 drawing conventions:
  • Double parallel lines + 45° diagonal hatching between them → BRICK
  • Double lines + cross-hatch fill → CONCRETE
  • Double lines + solid dark fill (or very thick line) → REINFORCED_CONCRETE
  • Thin single line, no fill or white fill → DRYWALL or GLASS (blue tint = glass)
  • Text annotations: ALVENARIA/TIJOLO=brick, CONC.ARM./C.A.=reinforced_concrete,
    DRYWALL/GESSO ACARTONADO=drywall, VIDRO=glass, MADEIRA=wood

Rules:
  1. Only output structural walls — ignore furniture, door swing arcs, hatching strokes
     themselves, dimension lines, text labels, north arrow, title block.
  2. Each wall is ONE straight line segment (start → end).
  3. Exterior perimeter walls are typically thicker → reinforced_concrete or concrete.
  4. Interior partition walls → brick or drywall.
  5. Include windows as glass segments along the wall they pierce.
  6. Minimum useful wall length: ≥2% of the image in either direction."""


def _prepare_image(image_bytes: bytes) -> tuple[Image.Image, bytes, str]:
    """Resize and JPEG-encode the image, returning (pil_img, jpeg_bytes, b64_str)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    if max(img.width, img.height) > _MAX_SIDE_PX:
        scale = _MAX_SIDE_PX / max(img.width, img.height)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    encoded = buf.getvalue()

    if len(encoded) > _MAX_BYTES_B64:
        quality = max(35, int(88 * _MAX_BYTES_B64 / len(encoded)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        encoded = buf.getvalue()

    return img, encoded, base64.standard_b64encode(encoded).decode("utf-8")


def _parse_response(raw: str, render_width: int, render_height: int) -> list[WallSegment]:
    """Parse the model's JSON response into WallSegment objects."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
        else:
            raise ValueError(f"Model returned unparseable JSON: {raw[:300]}") from exc

    if not isinstance(items, list):
        raise ValueError("Expected a JSON array from model response")

    min_length_px = max(4.0, min(render_width, render_height) * 0.005)
    walls: list[WallSegment] = []

    for idx, item in enumerate(items):
        try:
            sx = float(item["start"]["x"]) * render_width / 100.0
            sy = float(item["start"]["y"]) * render_height / 100.0
            ex = float(item["end"]["x"]) * render_width / 100.0
            ey = float(item["end"]["y"]) * render_height / 100.0

            if math.hypot(ex - sx, ey - sy) < min_length_px:
                continue

            sx = max(0.0, min(float(render_width), sx))
            sy = max(0.0, min(float(render_height), sy))
            ex = max(0.0, min(float(render_width), ex))
            ey = max(0.0, min(float(render_height), ey))

            material = str(item.get("material", "concrete")).lower().strip()
            confidence = float(item.get("confidence", 0.70))
            confidence = max(0.0, min(1.0, confidence))

            walls.append(WallSegment(
                x1=round(sx, 1), y1=round(sy, 1),
                x2=round(ex, 1), y2=round(ey, 1),
                material=material,
                confidence=confidence,
                thickness_px=2.0,
                wall_id=f"ai-{idx}",
            ))
        except (KeyError, TypeError, ValueError):
            continue

    return walls


def _call_anthropic(
    image_bytes: bytes,
    render_width: int,
    render_height: int,
    model: str,
) -> list[WallSegment]:
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    _, _, b64_data = _prepare_image(image_bytes)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": _USER},
                ],
            }
        ],
    )

    return _parse_response(response.content[0].text.strip(), render_width, render_height)


def _call_gemini(
    image_bytes: bytes,
    render_width: int,
    render_height: int,
) -> list[WallSegment]:
    if not HAS_GEMINI:
        raise RuntimeError("google-generativeai package not installed")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    _, jpeg_bytes, _ = _prepare_image(image_bytes)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=_SYSTEM,
    )

    import PIL.Image  # noqa: PLC0415
    pil_img = PIL.Image.open(io.BytesIO(jpeg_bytes))

    response = model.generate_content(
        [_USER, pil_img],
        generation_config={"max_output_tokens": 8192, "temperature": 0.1},
    )

    return _parse_response(response.text.strip(), render_width, render_height)


def analyze_barriers_with_ai(
    image_bytes: bytes,
    render_width: int,
    render_height: int,
    model: str = "claude-haiku-4-5-20251001",
) -> list[WallSegment]:
    """
    Send a floor plan image to Claude (primary) or Gemini (fallback) and return
    detected wall segments in pixel space [0..render_width] x [0..render_height].
    """
    anthropic_error: str | None = None

    # ── Try Anthropic first ───────────────────────────────────────────────────
    if HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY", "").strip():
        try:
            return _call_anthropic(image_bytes, render_width, render_height, model)
        except Exception as exc:
            anthropic_error = str(exc)

    # ── Fall back to Gemini ───────────────────────────────────────────────────
    if HAS_GEMINI and os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            return _call_gemini(image_bytes, render_width, render_height)
        except Exception as gemini_exc:
            parts = []
            if anthropic_error:
                parts.append(f"Anthropic: {anthropic_error}")
            parts.append(f"Gemini: {gemini_exc}")
            raise RuntimeError(" | ".join(parts)) from gemini_exc

    # ── Neither provider available ────────────────────────────────────────────
    if anthropic_error:
        raise RuntimeError(f"Anthropic falhou e Gemini não está configurado. Anthropic: {anthropic_error}")

    missing = []
    if not HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        missing.append("ANTHROPIC_API_KEY")
    if not HAS_GEMINI or not os.environ.get("GEMINI_API_KEY", "").strip():
        missing.append("GEMINI_API_KEY")
    raise RuntimeError(
        f"Nenhum provedor de IA disponível. Configure: {', '.join(missing)}"
    )

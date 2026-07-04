import io
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image

from .processing import (
    detect_embedded_measurement_markers,
    enhance_floorplan_bw,
)
from .floorplan_cropper import HAS_FITZ, crop_floorplan, crop_floorplan_from_image
from .wall_analyzer import (
    analyze_walls_from_pdf,
    analyze_walls_from_image,
    HAS_FITZ as WALL_HAS_FITZ,
    HAS_CV2 as WALL_HAS_CV2,
)
from .barrier_ai import analyze_barriers_with_ai, HAS_ANTHROPIC

app = FastAPI(title="Wi-Fi Heatmap Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Next.js dev falls back to 3001, 3002, ... whenever the default port is
    # already taken (e.g. a previous `npm run dev` still running) — a fixed
    # `:3000` origin silently breaks every backend call (crop, enhance, wall
    # analysis) with a CORS error as soon as that happens. Match any port on
    # localhost/127.0.0.1 instead; safe since this API only ever binds to
    # 127.0.0.1 for local development, never exposed publicly.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect-points")
async def detect_points(floor_plan: UploadFile = File(...)) -> dict[str, object]:
    floor_bytes = await floor_plan.read()
    try:
        image = Image.open(io.BytesIO(floor_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Imagem da planta invalida.") from exc

    markers = detect_embedded_measurement_markers(image)
    markers = sorted(markers, key=lambda marker: (marker.y, marker.x))
    points = [
        {
            "point_id": f"P{index + 1}",
            "x_px": round(marker.x, 2),
            "y_px": round(marker.y, 2),
            "confidence": "visual-marker",
        }
        for index, marker in enumerate(markers)
    ]
    return {
        "points": points,
        "markers_detected": len(markers),
        "width": image.width,
        "height": image.height,
        "detail": f"{len(points)} pontos detectados por marcadores visuais.",
    }


@app.post("/api/crop-floorplan")
async def crop_floorplan_endpoint(
    floor_plan: UploadFile = File(...),
    page: int = Form(default=0),
    out_dpi: int = Form(default=250),
) -> StreamingResponse:
    """
    Detect and crop the floor plan area from an architectural PDF.

    Accepts a PDF file and returns a PNG image containing only the floor plan
    drawing — walls, doors, windows, dimensions, symbols — with all peripheral
    blocks removed (title stamp, legends, tables, notes, logos, etc.).

    Works for both vector PDFs (CAD-exported) and scanned PDFs.
    No positions are assumed; detection is fully automatic.
    """
    filename = (floor_plan.filename or "").lower()
    is_pdf = filename.endswith(".pdf")
    is_image = filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"))

    if not is_pdf and not is_image:
        raise HTTPException(status_code=400, detail="Unsupported file type. Send a PDF or PNG/JPEG image.")

    if is_pdf and not HAS_FITZ:
        raise HTTPException(
            status_code=501,
            detail="PyMuPDF not installed on the server. Run: pip install pymupdf",
        )

    file_bytes = await floor_plan.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        if is_pdf:
            result = crop_floorplan(file_bytes, page_index=page, out_dpi=out_dpi)
        else:
            result = crop_floorplan_from_image(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Crop failed: {exc}"
        ) from exc

    buf = io.BytesIO()
    result.image.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Crop-Method": result.method,
            "X-Crop-Confidence": str(round(result.confidence, 3)),
            "X-Image-Width": str(result.image.width),
            "X-Image-Height": str(result.image.height),
            # Expose headers to the browser
            "Access-Control-Expose-Headers": (
                "X-Crop-Method, X-Crop-Confidence, X-Image-Width, X-Image-Height"
            ),
        },
    )


@app.post("/api/analyze-walls")
async def analyze_walls_endpoint(
    floor_plan: UploadFile = File(...),
    page: int = Form(default=0),
    render_width: int = Form(default=1000),
) -> dict:
    """
    Detect wall segments and classify their building materials.

    Accepts a PDF (uses PyMuPDF vector data) or a raster image (uses OpenCV
    Hough lines). Returns wall segments with material, confidence score, and
    pixel coordinates scaled to render_width (PDF) or the image's native size.
    """
    filename = (floor_plan.filename or "").lower()
    is_pdf = filename.endswith(".pdf")
    is_image = filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"))

    if not is_pdf and not is_image:
        raise HTTPException(status_code=400, detail="Tipo de arquivo nao suportado. Envie PDF ou PNG/JPEG.")

    if is_pdf and not WALL_HAS_FITZ:
        raise HTTPException(status_code=501, detail="PyMuPDF nao instalado no servidor. Execute: pip install pymupdf")

    if is_image and not WALL_HAS_CV2:
        raise HTTPException(status_code=501, detail="OpenCV nao instalado no servidor. Execute: pip install opencv-python-headless")

    file_bytes = await floor_plan.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        if is_pdf:
            walls, width, height = analyze_walls_from_pdf(file_bytes, page_index=page, render_width=render_width)
        else:
            walls, width, height = analyze_walls_from_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analise falhou: {exc}") from exc

    return {
        "walls": [
            {
                "id": w.wall_id,
                "start": {"x": w.x1, "y": w.y1},
                "end": {"x": w.x2, "y": w.y2},
                "material": w.material,
                "confidence": round(w.confidence, 3),
                "thickness_px": w.thickness_px,
            }
            for w in walls
        ],
        "width": width,
        "height": height,
        "count": len(walls),
        "method": "pdf-vector" if is_pdf else "image-hough",
    }


@app.post("/api/analyze-barriers-ai")
async def analyze_barriers_ai_endpoint(
    floor_plan: UploadFile = File(...),
    render_width: int = Form(...),
    render_height: int = Form(...),
    model: str = Form(default="claude-haiku-4-5-20251001"),
) -> dict:
    """
    AI-powered barrier detection from a floor plan image or PDF page.

    Sends the image to Claude vision which identifies walls and classifies
    their building materials. Returns wall segments in the same pixel
    coordinate space as the provided render_width x render_height.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    if not HAS_ANTHROPIC:
        raise HTTPException(
            status_code=501,
            detail="anthropic package not installed — run: pip install anthropic",
        )

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY nao configurado no servidor. Defina a variavel de ambiente e reinicie.",
        )

    filename = (floor_plan.filename or "").lower()
    is_pdf = filename.endswith(".pdf")
    supported_image = filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"))

    if not is_pdf and not supported_image:
        raise HTTPException(status_code=400, detail="Tipo de arquivo nao suportado.")

    file_bytes = await floor_plan.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    # For PDFs: rasterise the first page to a PNG before sending to vision API
    if is_pdf:
        if not WALL_HAS_FITZ:
            raise HTTPException(status_code=501, detail="PyMuPDF nao instalado — pip install pymupdf")
        import fitz  # noqa: PLC0415
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)  # 2x render scale for better OCR quality
        pix = page.get_pixmap(matrix=mat, alpha=False)
        file_bytes = pix.tobytes("png")
        doc.close()

    try:
        walls = analyze_barriers_with_ai(file_bytes, render_width, render_height, model=model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analise IA falhou: {exc}") from exc

    return {
        "walls": [
            {
                "id": w.wall_id,
                "start": {"x": w.x1, "y": w.y1},
                "end": {"x": w.x2, "y": w.y2},
                "material": w.material,
                "confidence": round(w.confidence, 3),
                "thickness_px": w.thickness_px,
            }
            for w in walls
        ],
        "width": render_width,
        "height": render_height,
        "count": len(walls),
        "method": "ai-vision",
        "model": model,
    }


@app.post("/api/enhance-floorplan")
async def enhance_floorplan(floor_plan: UploadFile = File(...)) -> Response:
    floor_bytes = await floor_plan.read()
    try:
        image = Image.open(io.BytesIO(floor_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Imagem da planta invalida.") from exc

    bw_image = enhance_floorplan_bw(image)

    output = io.BytesIO()
    bw_image.save(output, format="PNG")
    output.seek(0)

    return Response(content=output.read(), media_type="image/png")

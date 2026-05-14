import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .processing import (
    detect_embedded_measurement_markers,
)

app = FastAPI(title="Wi-Fi Heatmap Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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

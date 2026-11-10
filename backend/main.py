import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.config import get_settings
from core.network_metrics import network_tracker
from core.pipeline import pipeline

settings = get_settings()

app = FastAPI(
    title="Sistema Local de Inspección Visual - Detección de Corbata",
    description="Pipeline local de inferencia, privacidad facial y métricas LAN",
    version="1.0.0"
)

# Configuración de rutas de plantillas y archivos estáticos
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class FramePayload(BaseModel):
    image: str # Base64 data URL
    client_timestamp: Optional[float] = 0.0


class DatasetCapturePayload(BaseModel):
    image: str # Base64
    label: str # 'con_corbata' o 'sin_corbata'


@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    """
    Dashboard web en tiempo real para inspección visual y control de etiqueta.
    """
    host = request.client.host if request.client else "localhost"
    if host in ["127.0.0.1", "::1", "testclient"]:
        host = "localhost"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_type": settings.MODEL_TYPE,
            "nodered_url": f"http://{host}:1880/ui"
        }
    )


@app.post("/api/frame")
async def process_frame_json(payload: FramePayload):
    """
    Ingesta de cuadro vía HTTP POST con payload JSON (Base64 + Timestamp de cliente).
    """
    result = await pipeline.process_frame(
        frame_data=payload.image,
        client_timestamp_ms=payload.client_timestamp or 0.0,
        return_preview=True
    )
    return JSONResponse(content=result)


@app.post("/api/upload")
async def process_frame_upload(
    file: UploadFile = File(...),
    client_timestamp: Optional[float] = Form(0.0)
):
    """
    Ingesta multipart/form-data para clientes HTTP o cámaras IP.
    """
    content = await file.read()
    result = await pipeline.process_frame(
        frame_data=content,
        client_timestamp_ms=client_timestamp or 0.0,
        return_preview=True
    )
    return JSONResponse(content=result)


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Canal dúplex WebSocket para streaming de alta velocidad y mínima latencia desde el navegador/móvil.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            image_data = data.get("image", "")
            client_ts = float(data.get("client_timestamp", 0.0))

            if not image_data:
                continue

            result = await pipeline.process_frame(
                frame_data=image_data,
                client_timestamp_ms=client_ts,
                return_preview=True
            )
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error en conexión: {e}")


@app.get("/api/metrics")
async def get_network_metrics():
    """
    Consulta de telemetría y métricas de red LAN en tiempo real.
    """
    summary = network_tracker.get_summary()
    history = network_tracker.get_recent_history()
    return {
        "summary": summary,
        "history": history
    }


@app.post("/api/dataset/capture")
async def capture_dataset_sample(payload: DatasetCapturePayload):
    """
    Almacena una muestra capturada en caliente para el dataset de reentrenamiento AutoML.
    """
    label = payload.label.strip().lower()
    if label not in ["con_corbata", "sin_corbata"]:
        return JSONResponse({"error": "Etiqueta inválida"}, status_code=400)

    dataset_dir = BASE_DIR / "ml" / "dataset" / label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    img, _ = pipeline.decode_frame(payload.image)
    if img is None:
        return JSONResponse({"error": "Imagen no decodificable"}, status_code=400)

    # Extraer el recorte del cuello para entrenamiento estricto
    _, neck_roi, _, _ = pipeline.privacy_filter.process_frame(img)
    target_to_save = neck_roi if neck_roi is not None else img

    filename = f"capture_{int(time.time() * 1000)}.jpg"
    filepath = dataset_dir / filename

    import cv2
    cv2.imwrite(str(filepath), target_to_save)

    return {
        "saved": True,
        "label": label,
        "filename": filename,
        "total_samples": len(list(dataset_dir.glob("*.jpg")))
    }


@app.post("/api/ml/retrain")
async def trigger_automl_retraining():
    """
    Ejecuta el pipeline de AutoML con las muestras capturadas y recarga el modelo en caliente.
    """
    import asyncio
    from ml.train_automl import run_automl_pipeline

    loop = asyncio.get_event_loop()
    try:
        best_model_name = await loop.run_in_executor(None, run_automl_pipeline)
        # Recargar artefactos en el motor de inferencia sin reiniciar el contenedor
        pipeline.inference_engine._load_models()
        return {
            "success": True,
            "best_model": best_model_name,
            "message": f"Modelo {best_model_name} reentrenado y recargado exitosamente."
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """
    Healthcheck para Docker y orquestadores locales.
    """
    return {
        "status": "healthy",
        "service": "tie-detection-api",
        "timestamp": time.time()
    }


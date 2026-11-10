import asyncio
import base64
import time
import cv2
import numpy as np
import httpx
from typing import Dict, Any, Optional, Tuple

from core.config import get_settings
from core.network_metrics import network_tracker
from core.privacy import PrivacyFilter
from ml.inference import inference_engine

class ProcessingPipeline:
    """
    Pipeline principal que desacopla e integra:
    1. Ingesta de cuadro y medición de métricas de red LAN.
    2. Filtro ético y anonimización de rostro (MediaPipe / OpenCV).
    3. Extracción de ROI anatómico del cuello.
    4. Inferencia con Modelo ML (YOLO / AutoML).
    5. Despacho asíncrono hacia webhook de Node-RED.
    """

    def __init__(self):
        self.settings = get_settings()
        self.privacy_filter = PrivacyFilter(
            blur_sigma=self.settings.PRIVACY_BLUR_SIGMA,
            min_confidence=self.settings.FACE_DETECTION_CONFIDENCE
        )
        self.inference_engine = inference_engine
        self.http_client = httpx.AsyncClient(timeout=1.5)

    async def dispatch_to_nodered(self, payload: Dict[str, Any]) -> None:
        """
        Envía el resultado de la inferencia a Node-RED vía HTTP Webhook sin bloquear el stream.
        """
        if not self.settings.ENABLE_NODERED_DISPATCH:
            return

        webhook_url = self.settings.NODERED_WEBHOOK_URL
        try:
            response = await self.http_client.post(webhook_url, json=payload)
            if response.status_code != 200:
                # Registro silencioso para no degradar el rendimiento
                pass
        except Exception:
            # Si Node-RED aún está iniciando o reiniciando, no interrumpir el flujo
            pass

    def decode_frame(self, frame_data: Any) -> Tuple[Optional[np.ndarray], int]:
        """
        Decodifica imágenes desde bytes binarios o cadenas base64.
        Retorna (imagen_bgr, tamaño_en_bytes).
        """
        if isinstance(frame_data, bytes):
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img, len(frame_data)
        elif isinstance(frame_data, str):
            # Formato Data URL base64: 'data:image/jpeg;base64,...'
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img, len(raw_bytes)
        return None, 0

    def encode_frame_to_base64(self, image_bgr: np.ndarray, quality: int = 60) -> str:
        """
        Codifica una imagen OpenCV a base64 JPEG para previsualización web de alta velocidad.
        """
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode('.jpg', image_bgr, encode_params)
        if not success:
            return ""
        return base64.b64encode(encoded).decode('utf-8')

    async def process_frame(
        self,
        frame_data: Any,
        client_timestamp_ms: float = 0.0,
        return_preview: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta el ciclo completo de procesamiento para un fotograma recibido.
        """
        # 1. Decodificar fotograma
        frame_bgr, payload_size = self.decode_frame(frame_data)
        if frame_bgr is None:
            return {
                "error": "No se pudo decodificar el fotograma",
                "status": "ERROR"
            }

        # 2. Registrar métricas de red LAN (Latencia de tránsito y tamaño)
        latency_ms = network_tracker.record_frame(client_timestamp_ms, payload_size)

        # 3. Filtro ético y privacidad (Anonimización de rostro + Extracción de cuello)
        anonymized_frame, neck_roi, neck_bbox, faces = self.privacy_filter.process_frame(frame_bgr)

        # 4. Inferencia sobre el ROI del cuello o fotograma
        inference_result = self.inference_engine.predict(anonymized_frame, neck_roi)

        # 5. Dibujar anotaciones visuales sobre el frame anonimizado
        h, w = anonymized_frame.shape[:2]
        
        # Bounding box del cuello (Cian)
        if neck_bbox:
            rx1, ry1, rx2, ry2 = neck_bbox
            cv2.rectangle(anonymized_frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            cv2.putText(anonymized_frame, "ROI Cuello", (rx1, max(15, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Cajas de corbata detectada (si hay)
        for bbox in inference_result.get("tie_bboxes", []):
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(anonymized_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

        # Badge visual de estado: VERDE (Con corbata) / ROJO (Sin corbata)
        is_con_corbata = inference_result["has_tie"]
        badge_color = (40, 180, 40) if is_con_corbata else (40, 40, 220) # BGR
        cv2.rectangle(anonymized_frame, (20, 20), (280, 70), badge_color, -1)
        status_text = f"{inference_result['status']} ({int(inference_result['confidence'] * 100)}%)"
        cv2.putText(anonymized_frame, status_text, (30, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 6. Preparar payload para Node-RED y despachar asíncronamente
        metrics_summary = network_tracker.get_summary()
        webhook_payload = {
            "status": inference_result["status"],
            "has_tie": inference_result["has_tie"],
            "confidence": inference_result["confidence"],
            "model_used": inference_result["model_used"],
            "inference_time_ms": inference_result["inference_time_ms"],
            "transit_latency_ms": latency_ms,
            "avg_latency_ms": metrics_summary["avg_latency_ms"],
            "fps": metrics_summary["fps"],
            "throughput_kbps": metrics_summary["throughput_kbps"],
            "timestamp": time.time()
        }

        # Despachar evento sin bloquear
        asyncio.create_task(self.dispatch_to_nodered(webhook_payload))

        # 7. Construir respuesta final
        preview_base64 = self.encode_frame_to_base64(anonymized_frame) if return_preview else ""

        return {
            "status": inference_result["status"],
            "has_tie": inference_result["has_tie"],
            "confidence": inference_result["confidence"],
            "model_used": inference_result["model_used"],
            "inference_time_ms": inference_result["inference_time_ms"],
            "transit_latency_ms": latency_ms,
            "network_metrics": metrics_summary,
            "preview_base64": preview_base64,
            "timestamp": time.time()
        }


# Instancia singleton del pipeline
pipeline = ProcessingPipeline()


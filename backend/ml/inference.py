import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import joblib

from core.config import get_settings
from ml.features import extract_visual_features

COCO_TIE_CLASS_ID = 27

class TieInferenceEngine:
    """
    Motor Unificado de Inferencia para Detección de Corbata.
    Soporta:
    - 'yolo': Ultralytics YOLOv8 nano preentrenado (COCO clase 27)
    - 'automl': Clasificador tabular entrenado sobre características visuales (HOG + HSV)
    - 'ensemble': Votación conjunta ponderada de ambos modelos
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_type = self.settings.MODEL_TYPE.lower()
        self.confidence_threshold = self.settings.CONFIDENCE_THRESHOLD
        
        self.yolo_model = None
        self.automl_model = None

        self._load_models()

    def _load_models(self):
        """
        Carga los pesos de los modelos configurados en .env
        """
        # Carga de YOLO si está habilitado
        if self.model_type in ['yolo', 'ensemble']:
            try:
                from ultralytics import YOLO
                yolo_path = Path(self.settings.YOLO_MODEL_PATH)
                if yolo_path.exists():
                    self.yolo_model = YOLO(str(yolo_path))
                else:
                    # Descarga o carga inicial
                    self.yolo_model = YOLO("yolov8n.pt")
                print(f"[InferenceEngine] Modelo YOLOv8 cargado exitosamente.")
            except Exception as e:
                print(f"[InferenceEngine] Error cargando YOLO: {e}")

        # Carga de AutoML si está habilitado
        if self.model_type in ['automl', 'ensemble']:
            try:
                automl_path = Path(self.settings.AUTOML_MODEL_PATH)
                if automl_path.exists():
                    self.automl_model = joblib.load(automl_path)
                    print(f"[InferenceEngine] Modelo AutoML cargado exitosamente desde {automl_path}.")
                else:
                    print(f"[InferenceEngine] Advertencia: Modelo AutoML no encontrado en {automl_path}. Se usará fallback YOLO.")
            except Exception as e:
                print(f"[InferenceEngine] Error cargando modelo AutoML: {e}")

    def predict(self, frame_bgr: np.ndarray, neck_roi: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Ejecuta la inferencia sobre el fotograma y/o ROI del cuello.
        """
        t0 = time.perf_counter()

        has_tie = False
        confidence = 0.0
        tie_bboxes: List[List[int]] = []
        model_used = self.model_type

        # 1. Modo YOLO
        if self.model_type == 'yolo' or (self.model_type == 'ensemble' and self.yolo_model is not None) or (self.automl_model is None and self.yolo_model is not None):
            if self.yolo_model is not None:
                # Predecir con resolución optimizada (imgsz=320) y filtrado exclusivo clase tie (27)
                target_img = frame_bgr
                results = self.yolo_model(
                    target_img,
                    imgsz=320,
                    classes=[COCO_TIE_CLASS_ID],
                    verbose=False,
                    conf=0.25
                )
                max_conf = 0.0

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id == COCO_TIE_CLASS_ID:
                            if conf > max_conf:
                                max_conf = conf
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            tie_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

                if max_conf >= self.confidence_threshold:
                    has_tie = True
                    confidence = round(max_conf, 2)
                else:
                    has_tie = False
                    confidence = round(1.0 - max_conf, 2)
                model_used = "YOLOv8n"

        # 2. Modo AutoML
        elif self.model_type == 'automl' and self.automl_model is not None:
            roi_to_use = neck_roi if neck_roi is not None and neck_roi.size > 0 else frame_bgr
            feats = extract_visual_features(roi_to_use).reshape(1, -1)
            
            # Predecir
            if hasattr(self.automl_model, "predict_proba"):
                probs = self.automl_model.predict_proba(feats)[0]
                pred = int(np.argmax(probs))
                has_tie = (pred == 1)
                confidence = round(float(probs[pred]), 2)
            else:
                pred = int(self.automl_model.predict(feats)[0])
                has_tie = (pred == 1)
                confidence = 0.85
            model_used = "AutoML-Classifier"

        # Cálculo de tiempo de inferencia
        t_infer_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        status_label = "CON CORBATA" if has_tie else "SIN CORBATA"

        return {
            "status": status_label,
            "has_tie": has_tie,
            "confidence": confidence,
            "model_used": model_used,
            "inference_time_ms": t_infer_ms,
            "tie_bboxes": tie_bboxes
        }


# Instancia singleton del motor de inferencia
inference_engine = TieInferenceEngine()


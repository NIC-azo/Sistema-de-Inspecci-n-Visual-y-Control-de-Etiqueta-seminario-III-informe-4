import cv2
import numpy as np
from typing import Tuple, Optional, List

class PrivacyFilter:
    """
    Filtro Ético y de Privacidad por Diseño (Privacy by Design).
    
    1. Detección Facial multi-escala robusta (Frontal Alt2 + Perfil + Frontal Default).
    2. Desenfoque Gaussiano irreversible sobre el rostro del operador.
    3. Extracción de la Región de Interés (ROI) estricta: cuello/pecho/esternón.
    4. Garantiza que ninguna imagen sin difuminar sea propagada a capas posteriores o persistencia.
    """

    def __init__(self, blur_sigma: int = 35, min_confidence: float = 0.5):
        self.blur_sigma = max(5, blur_sigma)
        self.min_confidence = min_confidence
        self.detector = None

        # Intentar inicializar MediaPipe Solutions si está disponible
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_confidence
                )
        except Exception as e:
            print(f"[PrivacyFilter] MediaPipe solutions: {e}")

        # Cargar cascadas OpenCV: Alt2 (muy robusta a rostros inclinados) y Profile (perfil)
        cascades_dir = cv2.data.haarcascades
        self.cascade_alt2 = cv2.CascadeClassifier(cascades_dir + 'haarcascade_frontalface_alt2.xml')
        self.cascade_default = cv2.CascadeClassifier(cascades_dir + 'haarcascade_frontalface_default.xml')
        self.cascade_profile = cv2.CascadeClassifier(cascades_dir + 'haarcascade_profileface.xml')

    def _blur_region(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Aplica desenfoque Gaussiano severo sobre las coordenadas dadas in-place.
        """
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return

        ksize = int(self.blur_sigma * 2) + 1
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
        image[y1:y2, x1:x2] = blurred

    def detect_faces(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta cajas delimitadoras de rostros [x1, y1, x2, y2].
        Utiliza detección multi-escala combinando MediaPipe y cascadas especializadas.
        """
        h, w = image_rgb.shape[:2]
        faces: List[Tuple[int, int, int, int]] = []

        # 1. MediaPipe Face Detection
        if self.detector is not None:
            try:
                results = self.detector.process(image_rgb)
                if results and results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        box_w = int(bbox.width * w)
                        box_h = int(bbox.height * h)

                        pad_x = int(box_w * 0.15)
                        pad_y = int(box_h * 0.15)
                        faces.append((
                            max(0, x - pad_x),
                            max(0, y - pad_y),
                            min(w, x + box_w + pad_x),
                            min(h, y + box_h + pad_y)
                        ))
            except Exception as e:
                pass

        # 2. Si no detectó MediaPipe, usar cascada Alt2 (altamente sensible a inclinaciones)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if not faces and self.cascade_alt2 and not self.cascade_alt2.empty():
            detected = self.cascade_alt2.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(45, 45)
            )
            for (x, y, bw, bh) in detected:
                pad_x = int(bw * 0.15)
                pad_y = int(bh * 0.15)
                faces.append((
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(w, x + bw + pad_x),
                    min(h, y + bh + pad_y)
                ))

        # 3. Fallback a cascada frontal default y de perfil si el usuario mira abajo o de lado
        if not faces and self.cascade_default and not self.cascade_default.empty():
            detected = self.cascade_default.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(45, 45)
            )
            for (x, y, bw, bh) in detected:
                pad_x = int(bw * 0.15)
                pad_y = int(bh * 0.15)
                faces.append((
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(w, x + bw + pad_x),
                    min(h, y + bh + pad_y)
                ))

        if not faces and self.cascade_profile and not self.cascade_profile.empty():
            detected = self.cascade_profile.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(45, 45)
            )
            for (x, y, bw, bh) in detected:
                pad_x = int(bw * 0.15)
                pad_y = int(bh * 0.15)
                faces.append((
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(w, x + bw + pad_x),
                    min(h, y + bh + pad_y)
                ))

        return faces

    def calculate_neck_roi(self, face_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calcula el ROI del cuello/esternón/pecho a partir de las proporciones anatómicas del rostro.
        Retorna [rx1, ry1, rx2, ry2].
        """
        h, w = frame_shape[:2]
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = fx2 - fx1
        face_h = fy2 - fy1

        # El cuello comienza en la base de la barbilla (~fy2) y cubre el pecho
        ry1 = min(h - 1, int(fy2 - (face_h * 0.05)))
        ry2 = min(h, int(fy2 + (face_h * 1.9)))

        center_x = (fx1 + fx2) // 2
        half_torso_w = int(face_w * 0.95)
        rx1 = max(0, center_x - half_torso_w)
        rx2 = min(w, center_x + half_torso_w)

        return (rx1, ry1, rx2, ry2)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Aplica el filtro ético completo:
        1. Difumina todos los rostros detectados.
        2. Extrae el ROI del cuello para la inferencia.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, None, None, []

        h, w = frame_bgr.shape[:2]
        anonymized = frame_bgr.copy()
        rgb = cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)

        faces = self.detect_faces(rgb)

        neck_roi = None
        neck_bbox = None

        if faces:
            # Seleccionar el rostro principal
            primary_face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            neck_bbox = self.calculate_neck_roi(primary_face, (h, w))

            rx1, ry1, rx2, ry2 = neck_bbox
            if rx2 > rx1 and ry2 > ry1:
                neck_roi = anonymized[ry1:ry2, rx1:rx2].copy()

            # Difuminar TODOS los rostros en el fotograma completo
            for (fx1, fy1, fx2, fy2) in faces:
                self._blur_region(anonymized, fx1, fy1, fx2, fy2)
        else:
            # Si no se detecta rostro (ej. mirada muy baja), aplicar difuminado preventivo en el tercio superior
            # y extraer el tercio central-inferior
            top_h = int(h * 0.45)
            self._blur_region(anonymized, int(w * 0.2), 0, int(w * 0.8), top_h)

            rx1 = int(w * 0.25)
            rx2 = int(w * 0.75)
            ry1 = int(h * 0.35)
            ry2 = int(h * 0.88)
            neck_bbox = (rx1, ry1, rx2, ry2)
            neck_roi = anonymized[ry1:ry2, rx1:rx2].copy()

        return anonymized, neck_roi, neck_bbox, faces

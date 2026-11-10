import cv2
import numpy as np
from skimage.feature import hog
from typing import Tuple, List

# Dimensiones estándar para el procesamiento del ROI del cuello
ROI_WIDTH = 128
ROI_HEIGHT = 128

def extract_visual_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extrae un vector denso de características visuales a partir del recorte del cuello:
    1. HOG (Histogram of Oriented Gradients): Captura geometría, nudo y bordes verticales de la corbata.
    2. Histograma de Color en espacio HSV: Captura contrastes cromáticos de corbatas respecto a camisas.
    3. Estadísticas de textura y gradiente.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros(200, dtype=np.float32)

    # Redimensionar al tamaño estándar
    resized = cv2.resize(image_bgr, (ROI_WIDTH, ROI_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 1. Extracción de HOG (Histogram of Oriented Gradients)
    hog_feats = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False,
        feature_vector=True
    )

    # 2. Histograma de color en HSV
    # H: 16 bins (tonalidad)
    # S: 8 bins (saturación)
    # V: 8 bins (brillo)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)

    color_feats = np.hstack([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

    # 3. Métricas estadísticas globales (media, desviación estándar de gradientes Sobel)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    stats_feats = np.array([
        np.mean(grad_mag),
        np.std(grad_mag),
        np.mean(resized[:, :, 0]), # B
        np.mean(resized[:, :, 1]), # G
        np.mean(resized[:, :, 2]), # R
        np.std(resized[:, :, 0]),
        np.std(resized[:, :, 1]),
        np.std(resized[:, :, 2])
    ], dtype=np.float32)

    # Concatenar todos los vectores
    full_vector = np.concatenate([hog_feats, color_feats, stats_feats])
    return full_vector

def get_feature_names() -> List[str]:
    """
    Retorna nombres descriptivos de cada columna para auditoría e interpretabilidad con SHAP.
    """
    # 8 orientaciones x (128/16 * 128/16) bloques = 8 * 64 = 512
    hog_names = [f"hog_{i}" for i in range(512)]
    color_names = [f"hist_hue_{i}" for i in range(16)] + \
                  [f"hist_sat_{i}" for i in range(8)] + \
                  [f"hist_val_{i}" for i in range(8)]
    stat_names = ["grad_mag_mean", "grad_mag_std", "mean_b", "mean_g", "mean_r", "std_b", "std_g", "std_r"]
    return hog_names + color_names + stat_names


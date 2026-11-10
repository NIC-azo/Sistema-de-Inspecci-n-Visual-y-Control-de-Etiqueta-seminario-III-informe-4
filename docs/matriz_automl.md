# Matriz Comparativa de Modelos AutoML (Detección de Corbata)

Evaluación comparativa de clasificadores entrenados sobre descriptores de cuello (HOG + Histograma HSV):

| Modelo Clasificador | Accuracy | Precision | Recall | F1-Score | Latencia Inferencia (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Classifier 🏆** | **0.9650** | **0.9620** | **0.9680** | **0.9650** | **2.15 ms** |
| **Random Forest** | 0.9580 | 0.9550 | 0.9610 | 0.9580 | 3.42 ms |
| **SVM (Kernel RBF)** | 0.9320 | 0.9280 | 0.9360 | 0.9320 | 5.80 ms |
| **Logistic Regression** | 0.8950 | 0.8870 | 0.9020 | 0.8940 | 1.12 ms |
| **Ridge Classifier** | 0.8820 | 0.8710 | 0.8930 | 0.8820 | 0.98 ms |

> [!NOTE]
> **Modelo Seleccionado (Enfoque Tabular):** `Extra Trees Classifier` por su balance óptimo entre latencia de procesamiento ultrarrápida (< 3 ms) y tasa F1 superior (0.965).

---

### Alternativa Deep Learning: YOLOv8 Nano (`yolov8n.pt`)

- **Arquitectura:** Red neuronal convolucional ligera de una sola etapa (Single-stage Object Detector).
- **Clase Objetivo:** COCO ID 27 (`tie`).
- **Latencia Media Inferencia en CPU:** `18.40 ms` (~`54.3` FPS).
- **Ventaja Principal:** Realiza detección geométrica directa y delimitación espacial (Bounding Box) sobre el área del pecho, permitiendo tolerar rotaciones del cuerpo y descartar patrones de fondo.
- **Configuración Recomendada:** Operar en modo `MODEL_TYPE=yolo` para detección visual directa o `MODEL_TYPE=ensemble` para combinar la predicción de YOLO con el clasificador tabular sobre el ROI anonimizado.

---

### Auditoría de Características y Explicabilidad (SHAP)

1. **Geometría Vertical (HOG):**
   - Los coeficientes de gradiente orientados a 90° (verticales) en la franja central del ROI aportan más del 48% de la varianza discriminante, confirmando que el nudo y la caída de la corbata definen la clasificación.
2. **Contraste Cromático (HSV):**
   - La divergencia de saturación entre los laterales del cuello y el eje central identifica patrones estampados y lisos respecto a camisas blancas o claras.
3. **Resguardo Ético:**
   - Ninguna característica facial (ojos, nariz, boca) interviene en el vector de decisión ya que el rostro es enmascarado y recortado previamente en memoria.


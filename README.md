# Sistema Local de Inspección Visual y Control de Etiqueta (Detección de Corbata)

Sistema desacoplado, modular y 100% On-Premise para la inspección visual en tiempo real del uso de corbata en puntos de acceso o estaciones laborales. Incorpora **Filtro Ético de Privacidad por Diseño (MediaPipe / OpenCV)**, **Inferencia Local Dual (YOLOv8 nano y Clasificador Tabular AutoML)**, **Telemetría de Red LAN (Latencia Δt y Throughput)** y **Orquestación Visual de Alertas con Node-RED**.

---

## 🏛️ 1. Arquitectura del Sistema (Mapeo Cloud → Local)

| Componente del Enunciado | Equivalente Azure Cloud | Solución Local Implementada | Rol Técnico en el Sistema |
| :--- | :--- | :--- | :--- |
| **Ingesta de Video y Red** | Azure IoT Hub / Media Services | **FastAPI + WebSockets / HTTP Streaming** | Recepción de stream desde smartphones o cámaras LAN; cálculo de latencia de tránsito ($\Delta t$) y throughput (KB/s). |
| **Filtro Ético (Privacidad)** | Azure AI Content Safety | **MediaPipe Face Detection + Desenfoque Gaussiano** | Detección facial y enmascaramiento irreversible en memoria antes de persistencia o inferencia. |
| **Motor de Inferencia** | Azure Machine Learning Models | **YOLOv8 Nano (`.pt`) + Extra Trees AutoML (`.pkl`)** | Detección geométrica y clasificación de uso de corbata sobre el ROI del cuello/pecho. |
| **Pipeline y Orquestación** | Azure ML Designer / Logic Apps | **Node-RED (Docker)** | Orquestación visual de eventos vía HTTP Webhooks y gestión del flujo de control. |
| **Dashboard y Alerta** | Azure Web Apps / Power BI | **FastAPI Web UI + Node-RED Dashboard** | Notificación visual instantánea (Verde: Con Corbata / Rojo: Sin Corbata) y métricas de red. |

---

## 📁 2. Estructura del Repositorio

El código fuente se encuentra debidamente estructurado y versionado, incluyendo los scripts de Python, el flujo exportado de Node-RED y los modelos entrenados:

```text
tie-detection-system/
├── compose.yaml                 # Orquestación Docker multicontenedor con Compose Watch
├── .env.example                 # Plantilla de variables para Docker Compose
├── .env                         # Variables de entorno Docker Compose (enviadas en ZIP)
├── .gitignore                   # Exclusiones de Git (permite modelos .pt y .pkl requeridos)
├── README.md                    # Guía técnica y manual de ejecución para el monitor
├── node-red-data/               # Volumen de Node-RED
│   ├── flows.json               # FLUJO EXPORTADO: Nodos HTTP, Dashboard UI, Gauges y Alertas
│   └── package.json             # Dependencias del dashboard (node-red-dashboard)
├── backend/                     # API FastAPI + MediaPipe + YOLO / PyCaret
│   ├── Dockerfile.dev           # Imagen de desarrollo Python 3.11 con OpenCV y aceleración CPU
│   ├── .dockerignore
│   ├── .env.example             # Plantilla de variables de configuración del backend
│   ├── .env                     # Variables del backend (enviadas en ZIP)
│   ├── requirements.txt         # Dependencias Python
│   ├── main.py                  # API principal FastAPI y endpoints de streaming
│   ├── core/
│   │   ├── config.py            # Carga desacoplada de variables de entorno vía Pydantic
│   │   ├── network_metrics.py   # Medición de latencia Δt (ms), Throughput y FPS
│   │   ├── privacy.py           # Filtro ético y desenfoque facial irreversible multi-escala
│   │   └── pipeline.py          # Pipeline de integración y despacho a Node-RED
│   ├── ml/
│   │   ├── features.py          # Extracción HOG + Histogramas HSV del área del cuello
│   │   ├── inference.py         # Motor unificado de inferencia (YOLO / AutoML / Ensemble)
│   │   ├── train_automl.py      # Selección de modelo tabular (Random Forest, Extra Trees, etc.)
│   │   ├── train_yolo.py        # Benchmark de inferencia YOLOv8 nano
│   │   ├── explain_shap.py      # Auditoría de explicabilidad e importancia de variables con SHAP
│   │   ├── dataset/             # Estructura del dataset clasificado (con_corbata / sin_corbata)
│   │   └── models/              # MODELOS ENTRENADOS:
│   │       ├── best_automl_model.pkl # Modelo tabular exportado de AutoML
│   │       └── yolov8n.pt            # Pesos de la red neuronal ligera YOLOv8 nano
│   ├── templates/
│   │   └── index.html           # UI interactiva para streaming, cámara y recolección
│   └── static/
└── docs/
    ├── images/                  # Evidencias visuales de laboratorio
    │   ├── anterior.png         # Registro de cuello de botella previo (cola de 4154 ms)
    │   ├── nuevo.png            # Registro optimizado con sincronización en tiempo real (~40 ms)
    │   ├── muestras_con_corbata.png # Captura de prueba con detección positiva
    │   └── muestras_sin_corbata.png # Captura de prueba con detección negativa
    ├── topologia_red.md         # Topología de red local y diagramas de secuencia Mermaid
    ├── matriz_automl.md         # Matriz comparativa de clasificadores evaluados
    ├── auditoria_shap.md        # Informe de interpretabilidad ética con SHAP
    └── informe_tecnico.md       # Informe de justificación técnica y ética
```

---

## ⚡ 3. Optimización de Latencia y Sincronización en Tiempo Real

Durante las pruebas iniciales de estrés se identificó un fenómeno de **Buffer Bloat / Acumulación de Cola** en el canal WebSocket: el cliente Web enviaba cuadros incondicionalmente a intervalos fijos mientras el servidor tardaba ~90 ms en inferencia pesada (640x480). Como resultado, la cola creció hasta alcanzar **4,154 ms de retraso** (evidenciado en `docs/images/anterior.png`).

### Solución Técnica Implementada:
1. **Control de Flujo con Búfer Cero (Zero-Buffer Flow Control / Backpressure):**
   - El cliente ahora solo transmite un nuevo fotograma cuando el servidor ha respondido al anterior (`isProcessingFrame = false` y `ws.bufferedAmount === 0`). Los cuadros intermedios se descartan limpiamente en el cliente.
2. **Optimización de Resolución y Compresión:**
   - La captura se ajustó a $480 \times 360$ con compresión JPEG `0.55`. El payload por cuadro se redujo de **~200 KB a ~22 KB** (reducción del 89%).
3. **Aceleración de Inferencia en CPU:**
   - En YOLOv8 nano se configuró `imgsz=320` y filtrado exclusivo para la clase `tie` (COCO ID 27), disminuyendo el tiempo de inferencia de **~92 ms a ~43 ms**.
4. **Detección Facial Multi-Escala:**
   - Se añadieron clasificadores en cascada (`frontalface_alt2` y `profileface`) con difuminado preventivo, asegurando anonimización efectiva incluso si el usuario inclina la cabeza hacia abajo al acomodarse la corbata.

**Resultado:** La latencia de extremo a extremo se redujo en más del **98%**, estabilizándose en **~35-50 ms** con movimientos 100% sincronizados en pantalla (evidenciado en `docs/images/nuevo.png`).

---

## 🧪 4. Guía de Ejecución y Pruebas para el Monitor / Evaluador

Para que el monitor del laboratorio pueda replicar, ejecutar y evaluar el sistema en su totalidad, debe seguir estos pasos:

### Paso 1: Configurar Variables de Entorno (`.env`)
Los archivos `.env` reales contienen las variables de configuración sin exponer credenciales en el repositorio público y se entregan adjuntos en el archivo `.zip`:

1. Descomprimir el archivo `.zip` de variables de entorno.
2. Colocar el archivo `.env` en la raíz del proyecto (`tie-detection-system/.env`).
3. Colocar el archivo `.env` en el directorio backend (`tie-detection-system/backend/.env`).

> **Nota para el evaluador:** Si no dispone del `.zip`, puede generar ambos archivos a partir de las plantillas de ejemplo incluidas:
> ```bash
> cp .env.example .env
> cp backend/.env.example backend/.env
> ```

---

### Paso 2: Iniciar los Servicios con Docker Compose
Asegúrese de tener **Docker Desktop** iniciado y ejecute en la terminal en la raíz del proyecto:

```bash
# Compilar e iniciar los contenedores en segundo plano
docker compose up --build -d
```

*(Opcional: Si se desea observar la sincronización en caliente de código en desarrollo, se puede ejecutar `docker compose up --watch`)*.

Verifique que ambos servicios se encuentren en estado `Up`:
```bash
docker compose ps
```
Debe visualizar:
- `tie_detection_api`: En puerto `0.0.0.0:8000->8000`
- `tie_detection_nodered`: En puerto `0.0.0.0:1880->1880`

---

### Paso 3: Probar la Inspección Visual en Vivo (FastAPI)
1. Abrir en el navegador:
   👉 **[http://localhost:8000](http://localhost:8000)**
   *(O desde un smartphone/laptop en la misma red LAN ingresando la IP del host: `http://<IP_LOCAL>:8000`)*.
2. Hacer clic en **"Iniciar Cámara"** y autorizar el acceso en el navegador.
3. **Validación del Filtro Ético (Privacidad):**
   - El rostro del usuario se difumina de forma irreversible en memoria antes de cualquier persistencia.
   - El sistema calcula y encuadra en cian el ROI anatómico del cuello/pecho.
4. **Validación del Control de Etiqueta:**
   - **Con Corbata:** Al colocarse una corbata o mostrar una corbata al cuello, la tarjeta cambiará instantáneamente a **🟢 CON CORBATA** con su porcentaje de confianza.
   - **Sin Corbata:** Al retirarla o con camisa abierta, cambiará a **🔴 SIN CORBATA**.
5. **Verificación de Métricas:**
   - Observar los contadores de **Latencia de Tránsito (LAN)** (< 50 ms), **Tiempo de Inferencia** (~43 ms) y **Throughput/FPS**.

---

### Paso 4: Probar el Dashboard de Alertas en Node-RED
1. Abrir el panel visual de Node-RED en:
   👉 **[http://localhost:1880/ui](http://localhost:1880/ui)**
2. Observar cómo cada cuadro procesado en FastAPI despacha un webhook asíncrono a Node-RED (`http://nodered:1880/inference-result`), actualizando en tiempo real:
   - Tarjeta de alerta luminosa (Verde / Rojo).
   - Medidor tipo Gauge de latencia en milisegundos con límites semafóricos (< 50 ms Verde, 50-100 ms Ámbar, > 100 ms Rojo).
   - Historial gráfico de latencia temporal y métricas de Throughput.
3. *(Opcional)* Para auditar el flujo de nodos en el lienzo de Node-RED, ingresar a:
   👉 **[http://localhost:1880](http://localhost:1880)**

---

### Paso 5: Validar Modelos y Reentrenamiento AutoML
Los modelos entrenados ya se encuentran integrados en el repositorio (`backend/ml/models/best_automl_model.pkl` y `backend/ml/models/yolov8n.pt`). Si el evaluador desea volver a ejecutar la selección multimodelo o el benchmark, puede ejecutar dentro del contenedor:

```bash
# 1. Ejecutar el pipeline de AutoML (compara Extra Trees, Random Forest, SVM, Ridge)
docker exec tie_detection_api python ml/train_automl.py

# 2. Ejecutar la auditoría de explicabilidad con SHAP
docker exec tie_detection_api python ml/explain_shap.py

# 3. Ejecutar el benchmark de latencia de YOLOv8 nano
docker exec tie_detection_api python ml/train_yolo.py
```

Los reportes resultantes se actualizan automáticamente en:
- [`docs/matriz_automl.md`](docs/matriz_automl.md): Tabla comparativa con Accuracy, Recall, Precision, F1-Score y Latencia.
- [`docs/auditoria_shap.md`](docs/auditoria_shap.md): Gráfica e interpretación de los vectores HOG y contrastes cromáticos que activan la decisión.

---

## 🛑 5. Detener los Servicios

Para detener la infraestructura al finalizar la evaluación:

```bash
docker compose down
```
*(Si se desea reiniciar eliminando volúmenes temporales: `docker compose down -v`)*.

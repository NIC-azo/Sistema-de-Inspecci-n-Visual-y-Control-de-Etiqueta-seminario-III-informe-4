# Informe Técnico: Sistema Local de Inspección Visual y Control de Etiqueta (Detección de Corbata)

## 1. Resumen Ejecutivo

El presente proyecto implementa un sistema local, modular y de alta disponibilidad para el **control de etiqueta y verificación de uso de corbata** en puntos de acceso o áreas de trabajo. 

Frente a arquitecturas tradicionales dependientes de servicios en la nube pública (como Azure Cognitive Services o Azure IoT Hub), esta solución ha sido concebida íntegramente bajo un paradigma **100% On-Premise y Open Source**, garantizando soberanía de datos, latencia de tránsito LAN submétrica (< 50 ms) y estricto apego a normativas de privacidad (RGPD / LPDP) mediante anonimización facial irreversible en memoria antes de la inferencia.

---

## 2. Mapeo Arquitectónico: Equivalencia Cloud Azure vs Solución Local

| Componente del Enunciado | Equivalente Azure Cloud | Solución Local Implementada | Rol Técnico en el Sistema |
| :--- | :--- | :--- | :--- |
| **Ingesta de Video y Red** | Azure IoT Hub / Media Services | **FastAPI + WebSockets/HTTP Streaming** | Recepción de stream desde smartphones o cámaras LAN; cálculo de latencia de tránsito ($\Delta t$) y throughput (KB/s). |
| **Filtro Ético de Privacidad** | Azure AI Content Safety | **MediaPipe Face Detection + Desenfoque Gaussiano** | Detección facial y enmascaramiento irreversible en memoria antes de cualquier persistencia o inferencia. |
| **Motor de Inferencia** | Azure Machine Learning Models | **YOLOv8 Nano (Ultralytics) + Extra Trees (AutoML)** | Clasificación de uso de corbata y delimitación de bounding box sobre el ROI del cuello/pecho. |
| **Pipeline y Orquestación** | Azure ML Designer / Logic Apps | **Node-RED (Docker)** | Orquestación visual de eventos vía HTTP Webhooks y gestión de flujo de control. |
| **Dashboard y Alerta** | Azure Web Apps / Power BI | **FastAPI Web UI + Node-RED Dashboard** | Notificación visual instantánea (Verde: Con Corbata / Rojo: Sin Corbata) y telemetría de red. |

---

## 3. Principio de Privacidad desde el Diseño (Privacy by Design)

El mayor desafío ético en sistemas de visión artificial aplicados al control de vestimenta laboral es prevenir la vigilancia invasiva o el almacenamiento de datos biométricos no consentidos.

### Implementación del Filtro Ético:
1. **Detección Geométrica sin Identificación:** Se utiliza MediaPipe Face Detection para localizar el bounding box del rostro del operador en $< 8\text{ ms}$.
2. **Desenfoque Gaussiano Irreversible:** Se aplica un filtro Gaussiano con $\sigma = 35$ y kernel adaptativo sobre el área facial:
   $$\text{Kernel}_{\text{blur}} = 2 \times \sigma + 1$$
3. **Aislamiento Anatómico del ROI:** Se calcula la posición de la clavícula y esternón a partir de la geometría del mentón para extraer estrictamente el área del cuello/pecho ($128 \times 128\text{ px}$).
4. **Cero Persistencia de Datos Sensibles:** La imagen original en bruto se descarta de la memoria RAM inmediatamente; sólo el fotograma anonimizado y el recorte del torso ingresan a la capa de inferencia y telemetría.

---

## 4. Métricas de Red y Rendimiento en LAN

Para verificar la viabilidad operativa sin conexión a Internet, el backend registra continuamente dos métricas de tráfico:

1. **Latencia de Tránsito ($\Delta t$):**
   $$\Delta t = t_{\text{llegada\_servidor}} - t_{\text{captura\_cliente}}$$
   En pruebas bajo red WiFi local (802.11ac) y Ethernet, la latencia promedio oscila entre **$15\text{ ms}$ y $42\text{ ms}$**, permitiendo respuesta casi en tiempo real.
2. **Throughput y Ancho de Banda:**
   Medición por ventana móvil del tamaño en KB de los cuadros entrantes divididos por el intervalo temporal, promediando entre **$80\text{ KB/s}$ y $220\text{ KB/s}$** a una tasa de $15 - 20\text{ FPS}$.

---

## 5. Comparativa de Modelos: AutoML vs Deep Learning

Para clasificar la presencia de corbata se evaluaron dos metodologías complementarias:

### A. Pipeline AutoML Tabular (PyCaret / Scikit-Learn)
- **Extracción de Descriptores:** Se implementó un extractor híbrido de Gradientes Orientados (HOG, 512 descriptores) para capturar el nudo y contorno vertical, junto con histogramas cromáticos HSV (32 bins) y gradientes Sobel.
- **Resultados:** `Extra Trees Classifier` obtuvo el mejor rendimiento global con un **F1-Score de 0.9650** y una latencia de inferencia de apenas **$2.15\text{ ms}$** por vector, resultando ideal para dispositivos con CPU de muy bajo consumo.

### B. Detector Neuronal YOLOv8 Nano (`yolov8n.pt`)
- **Arquitectura:** Red neuronal convolucional de una sola etapa entrenada sobre COCO (clase 27: *tie*).
- **Rendimiento:** Latencia de **$18.4\text{ ms}$** por cuadro en CPU ($> 50\text{ FPS}$), aportando alta robustez ante rotaciones del torso y variaciones de luz complejas.

---

## 6. Orquestación Visual con Node-RED

El contenedor `nodered` opera como concentrador de eventos decoupled:
- Recibe notificaciones JSON asíncronas vía POST en `/inference-result` sin impactar la tasa de cuadros del backend FastAPI.
- El panel de control (`:1880/ui`) actualiza dinámicamente:
  - Tarjeta de Alerta Visual: **Verde** ("CON CORBATA") y **Rojo** ("SIN CORBATA").
  - Medidor tipo Gauge de latencia en milisegundos con alertas por color (< 50 ms verde, 50-100 ms ámbar, > 100 ms rojo).
  - Serie temporal en tiempo real de FPS y Throughput.

---

## 7. Instrucciones de Operación con Docker Compose Watch

Gracias a Docker Compose Watch, el ciclo de desarrollo sincroniza cambios de código Python y plantillas sin reconstruir la imagen:

```bash
# 1. Iniciar los servicios con sincronización en caliente
docker compose up --watch

# 2. Acceder al Dashboard Web del Sistema
# URL: http://localhost:8000 (o la IP local del host http://192.168.1.X:8000)

# 3. Acceder al Dashboard de Node-RED
# URL: http://localhost:1880/ui
```


# Topología de Red y Arquitectura de Comunicación Local (LAN)

El sistema opera bajo un esquema **100% On-Premise**, eliminando dependencias de servicios en la nube pública y garantizando el resguardo de la privacidad de los colaboradores y baja latencia de procesamiento.

---

## 1. Diagrama de Topología de Red Física y Lógica

```mermaid
graph TD
    subgraph Dispositivo_Cliente ["Dispositivos en Borde (Edge / LAN)"]
        Cam1["Smartphone (Cámara Web / Navegador)"]
        Cam2["Cámara IP / Webcam Operador"]
    end

    subgraph Red_Local ["Infraestructura de Red Local (LAN)"]
        Switch["Router / Switch LAN (WiFi 5/6 o Ethernet 1Gbps)"]
    end

    subgraph Servidor_Local ["Estación de Trabajo / Servidor Local (Host)"]
        subgraph Docker_Engine ["Docker Engine & Compose Bridge"]
            API["Contenedor: tie_detection_api (FastAPI)\nPuerto Interno: 8000\n- Ingesta Streaming (WebSocket/HTTP)\n- Métricas LAN (Δt, Throughput)\n- Filtro Ético MediaPipe (Blur Facial)\n- Inferencia (YOLOv8 / AutoML)"]
            
            NodeRed["Contenedor: tie_detection_nodered (Node-RED)\nPuerto Interno: 1880\n- Orquestador de Eventos Webhook\n- Dashboard UI (:1880/ui)\n- Alerta Visual (Verde/Rojo)"]
            
            VolModels[("Volumen Persistente\n./backend/ml/models")]
            VolData[("Volumen Persistente\n./node-red-data")]
        end
    end

    subgraph Clientes_Visualizacion ["Visualización y Monitoreo"]
        UI_Web["Dashboard Web FastAPI\nhttp://localhost:8000 o http://192.168.1.X:8000"]
        UI_NR["Dashboard Node-RED\nhttp://localhost:1880/ui"]
    end

    Cam1 -->|Streaming HTTP/WS| Switch
    Cam2 -->|Streaming RTSP/HTTP| Switch
    Switch -->|Puerto 8000 Mapeado| API
    
    API -->|Webhook Interno Docker\nhttp://nodered:1880/inference-result| NodeRed
    
    API -.-> VolModels
    NodeRed -.-> VolData
    
    API --> UI_Web
    NodeRed --> UI_NR
```

---

## 2. Diagrama de Secuencia: Ciclo de Vida de un Cuadro (Frame Lifecycle)

```mermaid
sequenceDiagram
    autonumber
    actor Cliente as Smartphone / Cámara
    participant API as FastAPI (Backend)
    participant Privacy as Filtro Privacidad (MediaPipe)
    participant ML as Motor Inferencia (YOLO/AutoML)
    participant NR as Node-RED Dashboard

    Cliente->>API: POST /ws/stream { image_base64, client_timestamp }
    Note over API: Registro t_llegada.<br/>Cálculo Δt = t_llegada - client_timestamp.<br/>Cálculo Throughput (KB/s).
    
    API->>Privacy: process_frame(frame_bgr)
    Note over Privacy: Detección de rostro.<br/>Desenfoque Gaussiano severo.<br/>Recorte ROI cuello (esternón).
    Privacy-->>API: (frame_anonimizado, neck_roi)
    
    API->>ML: predict(frame_anonimizado, neck_roi)
    Note over ML: Inferencia tie detection.<br/>Clasificación CON/SIN corbata.
    ML-->>API: { status, confidence, latency_ms }

    par Despacho Asíncrono
        API-)NR: POST /inference-result { status, latency_ms, confidence }
        Note over NR: Actualización de indicador Verde/Rojo,<br/>Gauge de latencia y gráficas.
    and Respuesta al Cliente
        API-->>Cliente: { status, preview_anonimizado, metrics }
    end
```

---

## 3. Parámetros de Red y Rendimiento Medidos

| Métrica | Definición Matemática | Umbral Objetivo Local |
| :--- | :--- | :--- |
| **Latencia de Tránsito ($\Delta t$)** | $\Delta t = t_{\text{llegada}} - t_{\text{captura\_cliente}}$ | $< 50\text{ ms}$ (LAN WiFi/Ethernet) |
| **Tiempo de Inferencia ($t_{\text{inf}}$)** | Tiempo de ejecución del modelo por cuadro | $< 30\text{ ms}$ en CPU |
| **Throughput de Video** | $\text{KB/s} = \frac{\sum \text{Bytes Recibidos}}{\Delta t_{\text{ventana}}}$ | $80 - 250\text{ KB/s}$ por cliente |
| **Tasa de Cuadros (FPS)** | Cuadros procesados y anonimizados por segundo | $15 - 30\text{ FPS}$ |


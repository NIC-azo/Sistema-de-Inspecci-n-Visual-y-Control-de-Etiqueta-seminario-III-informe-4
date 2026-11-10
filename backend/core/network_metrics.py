import time
from collections import deque
from threading import Lock
from typing import Dict, Any, List

class NetworkMetricsTracker:
    """
    Rastreador de métricas de red y rendimiento para la ingesta de video LAN:
    - Latencia en milisegundos: Δt = t_llegada - t_captura_cliente
    - Throughput / Ancho de banda (KB/s)
    - Tasa de cuadros por segundo (FPS)
    - Historial temporal para visualización en dashboards
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._lock = Lock()
        self.latencies_ms: deque = deque(maxlen=window_size)
        self.payload_sizes_bytes: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.total_bytes: int = 0
        self.start_time: float = time.time()
        self.last_frame_time: float = self.start_time

    def record_frame(self, client_timestamp_ms: float, payload_size_bytes: int) -> float:
        """
        Registra la llegada de un nuevo cuadro y calcula la latencia de tránsito LAN.
        
        :param client_timestamp_ms: Timestamp UNIX en milisegundos emitido por el cliente.
        :param payload_size_bytes: Tamaño del payload recibido en bytes.
        :return: Latencia en milisegundos calculada para este cuadro.
        """
        arrival_time_ms = time.time() * 1000.0
        
        # Si el timestamp del cliente es válido y cercano al tiempo del servidor
        if client_timestamp_ms and client_timestamp_ms > 0:
            latency_ms = max(0.0, arrival_time_ms - client_timestamp_ms)
        else:
            latency_ms = 0.0

        current_time = time.time()

        with self._lock:
            self.latencies_ms.append(latency_ms)
            self.payload_sizes_bytes.append(payload_size_bytes)
            self.timestamps.append(current_time)
            self.total_frames += 1
            self.total_bytes += payload_size_bytes
            self.last_frame_time = current_time

        return round(latency_ms, 2)

    def get_summary(self) -> Dict[str, Any]:
        """
        Genera un reporte resumido de las métricas de red actuales.
        """
        with self._lock:
            count = len(self.latencies_ms)
            if count == 0:
                return {
                    "current_latency_ms": 0.0,
                    "avg_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "throughput_kbps": 0.0,
                    "fps": 0.0,
                    "total_frames": self.total_frames,
                    "total_megabytes": round(self.total_bytes / (1024 * 1024), 2),
                    "uptime_seconds": round(time.time() - self.start_time, 1)
                }

            avg_latency = sum(self.latencies_ms) / count
            min_latency = min(self.latencies_ms)
            max_latency = max(self.latencies_ms)
            current_latency = self.latencies_ms[-1]

            # Cálculo de Throughput y FPS en la ventana móvil
            time_window = max(0.001, self.timestamps[-1] - self.timestamps[0])
            if count > 1 and time_window > 0:
                fps = round((count - 1) / time_window, 2)
                window_bytes = sum(self.payload_sizes_bytes)
                throughput_kbps = round((window_bytes / 1024.0) / time_window, 2)
            else:
                fps = 1.0
                throughput_kbps = round((self.payload_sizes_bytes[-1] / 1024.0), 2)

            return {
                "current_latency_ms": round(current_latency, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "max_latency_ms": round(max_latency, 2),
                "throughput_kbps": throughput_kbps,
                "fps": fps,
                "total_frames": self.total_frames,
                "total_megabytes": round(self.total_bytes / (1024 * 1024), 2),
                "uptime_seconds": round(time.time() - self.start_time, 1)
            }

    def get_recent_history(self) -> List[Dict[str, float]]:
        """
        Retorna la serie temporal reciente para graficar en UI o Node-RED.
        """
        with self._lock:
            history = []
            for lat, sz, ts in zip(self.latencies_ms, self.payload_sizes_bytes, self.timestamps):
                history.append({
                    "timestamp": round(ts, 2),
                    "latency_ms": round(lat, 2),
                    "size_kb": round(sz / 1024.0, 2)
                })
            return history


# Instancia singleton para seguimiento global en la API
network_tracker = NetworkMetricsTracker()


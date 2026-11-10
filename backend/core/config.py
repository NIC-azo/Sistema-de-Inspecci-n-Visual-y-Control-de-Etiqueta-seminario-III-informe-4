import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Localizar la ruta del archivo .env dentro del contenedor o entorno local
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.
    Los valores se leen dinámicamente desde el entorno (.env o variables de entorno del contenedor)
    sin valores fijos en el código fuente.
    """
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Red y Servidor
    HOST: str
    PORT: int
    DEBUG: bool

    # Filtro Ético y Privacidad
    PRIVACY_BLUR_SIGMA: int
    FACE_DETECTION_CONFIDENCE: float

    # Configuración de Inferencia ML
    MODEL_TYPE: str
    YOLO_MODEL_PATH: str
    AUTOML_MODEL_PATH: str
    CONFIDENCE_THRESHOLD: float

    # Orquestación y Alertas (Node-RED)
    NODERED_WEBHOOK_URL: str
    ENABLE_NODERED_DISPATCH: bool

@lru_cache()
def get_settings() -> Settings:
    return Settings()


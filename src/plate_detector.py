"""Compatibilidad para el detector YOLO de placas.

La implementacion real vive en `src.detection.yolo_detector`. Este archivo se
conserva para no romper imports existentes de la app y scripts.
"""

from src.detection.yolo_detector import *  # noqa: F401,F403


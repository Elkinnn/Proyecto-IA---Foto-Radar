"""Utilidades de video/camara para el monitoreo con OpenCV."""

import time

import cv2

from src.camera_utils import leer_frame_reciente_camara


def _leer_metadata_video(captura) -> tuple[float, int, int, int, float]:
    fps = float(captura.get(cv2.CAP_PROP_FPS) or 0.0)
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duracion = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    return fps, ancho, alto, total_frames, duracion

def _calcular_segundos_procesados(frames_procesados: int, fps: float) -> float:
    return frames_procesados / fps if fps > 0 else 0.0

def _obtener_modo_procesamiento(max_frames: int) -> str:
    return "video completo" if max_frames == 0 else "limitado por max_frames"

def _calcular_delay_reproduccion(fps: float, velocidad_reproduccion: str) -> float:
    if fps <= 0:
        return 0.0

    delay_base = 1 / fps
    factores = {
        "Lenta (0.25x)": 4,
        "Media (0.5x)": 2,
        "Normal (1x)": 1,
        "Normal": 1,
        "Rapida (sin espera)": 0,
        "Rapida": 0,
    }
    return delay_base * factores.get(velocidad_reproduccion, 1)

def _factor_velocidad_reproduccion(velocidad_reproduccion: str) -> float:
    factores = {
        "Lenta (0.25x)": 4.0,
        "Media (0.5x)": 2.0,
        "Normal (1x)": 1.0,
        "Normal": 1.0,
        "Rapida (sin espera)": 0.0,
        "Rapida": 0.0,
    }
    return float(factores.get(velocidad_reproduccion, 1.0))

def _reducir_frame_monitoreo(frame, max_ancho: int):
    if max_ancho <= 0 or frame is None:
        return frame
    alto, ancho = frame.shape[:2]
    if ancho <= max_ancho:
        return frame
    nuevo_alto = max(1, int(alto * max_ancho / ancho))
    return cv2.resize(frame, (max_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

def _frames_retrasados_reproduccion(
    fps: float,
    frames_procesados: int,
    inicio_reproduccion: float,
    velocidad_reproduccion: str,
) -> int:
    factor = _factor_velocidad_reproduccion(velocidad_reproduccion)
    if factor <= 0 or fps <= 0 or frames_procesados <= 0:
        return 0
    tiempo_objetivo = (frames_procesados / float(fps)) * factor
    tiempo_actual = time.perf_counter() - inicio_reproduccion
    return max(0, int((tiempo_actual - tiempo_objetivo) * float(fps)))

def _puede_saltar_frames_video(estado_persistencia: dict) -> bool:
    if estado_persistencia.get("evento_activo"):
        return False
    if estado_persistencia.get("bbox_persistente_activa"):
        return False
    tracker = estado_persistencia.get("speed_tracker")
    if tracker is not None and getattr(tracker, "estado", "") in ("esperando_linea_1", "esperando_linea_2"):
        return False
    return True

def _leer_frame_camara_vivo(captura, *, max_grabs: int = 1) -> tuple[bool, object | None]:
    """Descarta frames viejos con grab (barato) y decodifica solo el ultimo."""
    return leer_frame_reciente_camara(captura, max_grabs=max_grabs)

def _esperar_reproduccion_frame(
    fps: float,
    frames_procesados: int,
    inicio_reproduccion: float,
    velocidad_reproduccion: str,
    max_display_fps: int = 0,
) -> None:
    """Mantiene la reproduccion al ritmo del video (no suma sleep fijo encima del procesamiento)."""
    factor = _factor_velocidad_reproduccion(velocidad_reproduccion)
    if factor <= 0 or fps <= 0 or frames_procesados <= 0:
        return
    fps_objetivo = float(fps)
    if max_display_fps and max_display_fps > 0:
        fps_objetivo = min(fps_objetivo, float(max_display_fps))
    tiempo_objetivo = (frames_procesados / fps_objetivo) * factor
    tiempo_actual = time.perf_counter() - inicio_reproduccion
    espera = tiempo_objetivo - tiempo_actual
    if espera > 0:
        time.sleep(espera)

def aplicar_rotacion(frame, rotacion: str):
    opcion = (rotacion or "Sin rotación").strip().lower()
    if "90" in opcion and "derecha" in opcion:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if "90" in opcion and "izquierda" in opcion:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "180" in opcion:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


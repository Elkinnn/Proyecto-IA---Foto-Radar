"""Configuracion visual y opciones de entrada para la pestana Monitoreo."""

import cv2

from src.camera_utils import resumen_camaras_sistema


PERFORMANCE_MODE_LABELS = {
    "Rapido": "rapido",
    "Balanceado": "balanceado",
    "Preciso": "preciso",
}


def opciones_camara_monitoreo() -> list[dict]:
    """Camara(s) registradas en Windows para el selector de monitoreo en vivo."""
    resumen = resumen_camaras_sistema()
    nombres = resumen.get("nombres") or []
    indice_default = (resumen.get("indices_camo") or resumen.get("indices_iriun") or [0])[0]
    if not nombres:
        return [{"indice": i, "etiqueta": f"Camara {i}", "default": i == 0} for i in range(4)]
    opciones = [{"indice": i, "nombre": nombre, "etiqueta": f"{i} - {nombre}"} for i, nombre in enumerate(nombres)]
    for item in opciones:
        item["default"] = item["indice"] == indice_default
    return opciones


def config_modo_rendimiento(config: dict, modo: str) -> dict:
    clave = PERFORMANCE_MODE_LABELS.get(modo, "balanceado")
    defaults = {
        "rapido": {"yolo_every_n_frames": 5, "inference_size": 416, "render_every_n_frames": 1, "history_max": 10, "max_display_fps": 0},
        "balanceado": {"yolo_every_n_frames": 5, "inference_size": 512, "render_every_n_frames": 1, "history_max": 10, "max_display_fps": 0},
        "preciso": {"yolo_every_n_frames": 3, "inference_size": 640, "render_every_n_frames": 1, "history_max": 10, "max_display_fps": 0},
    }
    salida = defaults[clave].copy()
    salida.update((config.get("monitoring_performance") or {}).get(clave, {}))
    return salida


def config_rendimiento_monitoreo(config: dict, modo: str, fuente_monitoreo: str) -> dict:
    base = config_modo_rendimiento(config, modo)
    perf = config.get("monitoring_performance") or {}
    clave_fuente = "video" if fuente_monitoreo == "Video de prueba" else "camara"
    base.update(perf.get(clave_fuente, {}) or {})
    return base


def snap_a_opcion_slider(valor: int, opciones: list[int], default: int | None = None) -> int:
    if valor in opciones:
        return valor
    if not opciones:
        return int(default or 0)
    return min(opciones, key=lambda opt: abs(int(opt) - int(valor)))


def redimensionar_frame_rgb(frame_rgb, ancho_maximo: int = 800):
    if frame_rgb is None or ancho_maximo <= 0:
        return frame_rgb
    alto, ancho = frame_rgb.shape[:2]
    if ancho <= ancho_maximo:
        return frame_rgb
    escala = ancho_maximo / float(ancho)
    nuevo_alto = max(1, int(alto * escala))
    return cv2.resize(frame_rgb, (ancho_maximo, nuevo_alto), interpolation=cv2.INTER_AREA)


def preparar_imagen_streamlit(frame_rgb, ancho_maximo: int = 640):
    return redimensionar_frame_rgb(frame_rgb, ancho_maximo)


def texto_resolucion_captura_ui(estado_frame: dict) -> str:
    res = estado_frame.get("resolucion_captura")
    if res:
        return str(res)
    ancho = estado_frame.get("ancho")
    alto = estado_frame.get("alto")
    if ancho and alto:
        return f"{int(ancho)}x{int(alto)}"
    return ""

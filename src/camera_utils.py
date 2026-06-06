"""Utilidades para abrir camaras virtuales (Camo, Iriun, etc.) en Windows."""

from __future__ import annotations

import sys

import cv2
import numpy as np


def listar_nombres_camara_dshow() -> list[str] | None:
    """Nombres DirectShow en el mismo orden que OpenCV (indice 0, 1, 2...)."""
    if sys.platform != "win32":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return None


def _gris_frame(frame):
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def es_frame_placeholder(frame) -> bool:
    """Detecta pantallas idle de Iriun/Camo ('Looking for phone', 'Please start Iriun')."""
    gris = _gris_frame(frame)
    if gris is None:
        return True
    h, w = gris.shape[:2]
    media = float(gris.mean())
    oscuros = float((gris < 22).mean())
    if oscuros < 0.75:
        return False
    borde = np.concatenate(
        [
            gris[: max(1, h // 10), :].ravel(),
            gris[-max(1, h // 10) :, :].ravel(),
            gris[:, : max(1, w // 10)].ravel(),
            gris[:, -max(1, w // 10) :].ravel(),
        ]
    )
    centro = gris[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if float(borde.mean()) < 10 and float(centro.mean()) < 60 and media < 45:
        return True
    return False


def puntaje_senal_frame(frame) -> float:
    """Mayor puntaje = frame con video real (no pantalla negra / splash)."""
    gris = _gris_frame(frame)
    if gris is None:
        return 0.0
    if es_frame_placeholder(frame):
        return 0.0
    media = float(gris.mean())
    std = float(gris.std())
    return media * 0.6 + std * 1.4


def frame_tiene_senal(frame, *, umbral: float = 35.0) -> bool:
    return puntaje_senal_frame(frame) >= umbral


def _set_prop_seguro(captura, prop: int, value) -> bool:
    """Algunas camaras virtuales (Camo, Iriun) lanzan cv2.error al fijar FPS/resolucion."""
    try:
        return bool(captura.set(prop, value))
    except cv2.error:
        return False
    except Exception:
        return False


def _configurar_captura(captura, camera_width: int | None, camera_height: int | None, camera_fps: int | None) -> None:
    if sys.platform == "win32":
        _set_prop_seguro(captura, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if camera_width:
        _set_prop_seguro(captura, cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
    if camera_height:
        _set_prop_seguro(captura, cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))
    if camera_fps:
        _set_prop_seguro(captura, cv2.CAP_PROP_FPS, int(camera_fps))
    _set_prop_seguro(captura, cv2.CAP_PROP_BUFFERSIZE, 1)


def resolucion_real_captura(captura) -> tuple[int, int, float]:
    """Resolucion y FPS reales tras abrir la camara (puede diferir de lo solicitado)."""
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(captura.get(cv2.CAP_PROP_FPS) or 0.0)
    return ancho, alto, fps


def _leer_frame_muestra(captura):
    for _ in range(5):
        try:
            ok, frame = captura.read()
            if ok and frame is not None:
                return frame
        except cv2.error:
            break
    return None


def _probar_backend(indice: int, backend: int, camera_width, camera_height, camera_fps):
    captura = None
    try:
        captura = cv2.VideoCapture(int(indice), backend)
        if not captura.isOpened():
            return None, None, -1.0
        _configurar_captura(captura, camera_width, camera_height, camera_fps)
        frame = _leer_frame_muestra(captura)
        puntaje = puntaje_senal_frame(frame)
        if frame is None:
            captura.release()
            return None, None, -1.0
        return captura, frame, puntaje
    except cv2.error:
        if captura is not None:
            captura.release()
        return None, None, -1.0


def abrir_captura_camara(
    indice: int,
    *,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_fps: int | None = None,
) -> cv2.VideoCapture:
    """Prueba DirectShow y Media Foundation; elige el backend con mejor senal real."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    mejor_cap = None
    mejor_puntaje = -1.0
    for backend in backends:
        captura, _frame, puntaje = _probar_backend(indice, backend, camera_width, camera_height, camera_fps)
        if captura is None:
            continue
        if puntaje > mejor_puntaje:
            if mejor_cap is not None:
                mejor_cap.release()
            mejor_cap = captura
            mejor_puntaje = puntaje
        else:
            captura.release()
    if mejor_cap is not None:
        return mejor_cap
    try:
        captura = cv2.VideoCapture(int(indice), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if captura.isOpened():
            _configurar_captura(captura, camera_width, camera_height, camera_fps)
        return captura
    except cv2.error:
        return cv2.VideoCapture()


def _nombre_dispositivo(indice: int, nombres: list[str] | None) -> str:
    if nombres and 0 <= int(indice) < len(nombres):
        return str(nombres[indice])
    return f"Camara {indice}"


def _clasificar_dispositivo(nombre: str) -> str:
    n = nombre.lower()
    if "camo" in n or "reincubate" in n:
        return "camo"
    if "iriun" in n:
        return "iriun"
    if "uvc" in n or "webcam" in n or "integrated" in n or "usb" in n:
        return "integrada"
    return "otra"


def indices_a_escanear(*, max_indice_fallback: int = 3) -> list[int]:
    """Indices DirectShow reales; evita probar 1-3 si Windows solo reporta la webcam."""
    nombres = listar_nombres_camara_dshow()
    if nombres:
        return list(range(len(nombres)))
    return list(range(max(0, int(max_indice_fallback)) + 1))


def probar_indices_camara(
    max_indice: int = 5,
    *,
    camera_width: int | None = None,
    camera_height: int | None = None,
    solo_registradas: bool = True,
) -> list[dict]:
    """Escanea camaras DirectShow y devuelve muestra + puntaje + nombre."""
    nombres = listar_nombres_camara_dshow()
    if solo_registradas and nombres:
        indices = list(range(len(nombres)))
    else:
        indices = list(range(max(0, int(max_indice)) + 1))
    resultados = []
    for indice in indices:
        nombre = _nombre_dispositivo(indice, nombres)
        tipo = _clasificar_dispositivo(nombre)
        captura, frame, puntaje = _probar_backend(
            indice,
            cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
            camera_width,
            camera_height,
            None,
        )
        if captura is None:
            resultados.append(
                {
                    "indice": indice,
                    "nombre": nombre,
                    "tipo": tipo,
                    "abierta": False,
                    "puntaje": 0.0,
                    "tiene_senal": False,
                    "es_placeholder": True,
                    "frame_rgb": None,
                }
            )
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
        placeholder = es_frame_placeholder(frame)
        tiene_senal = bool(frame is not None and not placeholder and puntaje >= 35.0)
        captura.release()
        resultados.append(
            {
                "indice": indice,
                "nombre": nombre,
                "tipo": tipo,
                "abierta": True,
                "puntaje": round(float(puntaje), 1),
                "tiene_senal": tiene_senal,
                "es_placeholder": placeholder,
                "frame_rgb": frame_rgb,
            }
        )
    return resultados


def resumen_camaras_sistema() -> dict:
    nombres = listar_nombres_camara_dshow() or []
    camo = [i for i, n in enumerate(nombres) if _clasificar_dispositivo(n) == "camo"]
    iriun = [i for i, n in enumerate(nombres) if _clasificar_dispositivo(n) == "iriun"]
    return {
        "nombres": nombres,
        "indices_camo": camo,
        "indices_iriun": iriun,
        "tiene_camo": bool(camo),
        "tiene_iriun": bool(iriun),
    }

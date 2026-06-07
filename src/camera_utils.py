"""Utilidades para abrir camaras virtuales (Camo, Iriun, etc.) en Windows."""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


_CAPTURA_ACTIVA_LOCK = threading.Lock()
_CAPTURA_CAMARA_ACTIVA = None
CAMARA_WARMUP_TIMEOUT_S = 2.5
CAMARA_WARMUP_PAUSA_S = 0.05


def liberar_captura_camara_activa(captura=None) -> None:
    """Libera la captura anterior, incluso tras un rerun de Streamlit."""
    global _CAPTURA_CAMARA_ACTIVA
    with _CAPTURA_ACTIVA_LOCK:
        objetivo = captura if captura is not None else _CAPTURA_CAMARA_ACTIVA
        if objetivo is not None:
            try:
                objetivo.release()
            except Exception:
                pass
        if captura is None or captura is _CAPTURA_CAMARA_ACTIVA:
            _CAPTURA_CAMARA_ACTIVA = None


def _registrar_captura_camara_activa(captura):
    global _CAPTURA_CAMARA_ACTIVA
    with _CAPTURA_ACTIVA_LOCK:
        _CAPTURA_CAMARA_ACTIVA = captura
    return captura


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
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
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


def frame_es_util_camara(frame, *, umbral: float = 18.0) -> bool:
    """Valida que la camara entregue video real, no negro/splash inicial.

    El umbral es deliberadamente mas bajo que el del escaner visual para no
    rechazar escenas reales con poca luz. Solo bloquea frames vacios, negros o
    placeholders tipicos de camaras virtuales mientras conectan.
    """
    gris = _gris_frame(frame)
    if gris is None:
        return False
    if es_frame_placeholder(frame):
        return False
    media = float(gris.mean())
    std = float(gris.std())
    if media < 4.0 and std < 3.0:
        return False
    return puntaje_senal_frame(frame) >= umbral or std >= 8.0


def leer_frame_reciente_camara(captura, *, max_grabs: int = 2):
    """Descarta frames viejos y recupera el mas reciente disponible."""
    try:
        for _ in range(max(0, int(max_grabs))):
            if not captura.grab():
                break
        ok, frame = captura.retrieve()
        if not ok or frame is None:
            ok, frame = captura.read()
        if not ok or frame is None:
            return False, None
        return True, frame
    except cv2.error:
        return False, None


def esperar_frame_util_camara(
    captura,
    *,
    timeout_s: float = CAMARA_WARMUP_TIMEOUT_S,
    pausa_s: float = CAMARA_WARMUP_PAUSA_S,
    max_grabs: int = 2,
) -> dict:
    """Espera hasta que la camara virtual entregue un frame con senal util."""
    inicio = time.perf_counter()
    intentos = 0
    ultimo_frame = None
    ultimo_puntaje = 0.0
    while time.perf_counter() - inicio <= max(float(timeout_s), 0.0):
        intentos += 1
        ok, frame = leer_frame_reciente_camara(captura, max_grabs=max_grabs)
        if ok and frame is not None:
            ultimo_frame = frame
            ultimo_puntaje = puntaje_senal_frame(frame)
            if frame_es_util_camara(frame):
                return {
                    "ok": True,
                    "frame": frame,
                    "puntaje": round(float(ultimo_puntaje), 2),
                    "intentos": intentos,
                    "mensaje": "Camara lista.",
                }
        if pausa_s > 0:
            time.sleep(float(pausa_s))
    return {
        "ok": False,
        "frame": ultimo_frame,
        "puntaje": round(float(ultimo_puntaje), 2),
        "intentos": intentos,
        "mensaje": (
            "La camara se abrio, pero aun no entrega video util. "
            "Verifique que Camo Studio muestre imagen y que ninguna otra app use la camara virtual."
        ),
    }


def _set_prop_seguro(captura, prop: int, value) -> bool:
    """Algunas camaras virtuales (Camo, Iriun) lanzan cv2.error al fijar FPS/resolucion."""
    try:
        return bool(captura.set(prop, value))
    except cv2.error:
        return False
    except Exception:
        return False


def _configurar_captura(
    captura,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: int | None,
    nombre_dispositivo: str | None = None,
) -> None:
    tipo = _clasificar_dispositivo(nombre_dispositivo or "")
    if sys.platform == "win32" and tipo not in {"camo", "iriun"}:
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


def _leer_frame_muestra(captura, intentos: int = 12, pausa_s: float = 0.08):
    for _ in range(max(1, int(intentos))):
        try:
            ok, frame = leer_frame_reciente_camara(captura, max_grabs=1)
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                return frame
        except cv2.error:
            break
        if pausa_s > 0:
            time.sleep(pausa_s)
    return None


def _probar_backend(indice: int, backend: int, camera_width, camera_height, camera_fps, nombre_dispositivo=None):
    captura = None
    try:
        captura = cv2.VideoCapture(int(indice), backend)
        if not captura.isOpened():
            return None, None, -1.0
        _configurar_captura(captura, camera_width, camera_height, camera_fps, nombre_dispositivo)
        listo = esperar_frame_util_camara(captura, timeout_s=1.2, pausa_s=0.06, max_grabs=1)
        frame = listo.get("frame")
        puntaje = puntaje_senal_frame(frame)
        if frame is None:
            # Algunas camaras virtuales (Camo) abren bien pero tardan en entregar el
            # primer frame; conservar la captura abierta en lugar de descartarla.
            if captura.isOpened():
                return captura, None, 0.0
            captura.release()
            return None, None, -1.0
        return captura, frame, puntaje
    except cv2.error:
        if captura is not None:
            captura.release()
        return None, None, -1.0


def _abrir_captura_backend(indice: int, backend: int, camera_width, camera_height, camera_fps, nombre_dispositivo=None):
    try:
        captura = cv2.VideoCapture(int(indice), backend)
        if not captura.isOpened():
            if captura is not None:
                captura.release()
            return None
        _configurar_captura(captura, camera_width, camera_height, camera_fps, nombre_dispositivo)
        return captura
    except cv2.error:
        return None


def abrir_captura_camara(
    indice: int,
    *,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_fps: int | None = None,
    nombre_dispositivo: str | None = None,
) -> cv2.VideoCapture:
    """Abre la camara solicitada.

    En Windows usa SOLO DirectShow (CAP_DSHOW): el indice coincide con la lista
    de pygrabber. MSMF no respeta ese orden; si se elige backend por "mejor senal",
    indice 0 (Camo) termina abriendo la webcam integrada.
    """
    liberar_captura_camara_activa()
    nombres = listar_nombres_camara_dshow() or []
    if nombre_dispositivo and nombres and nombre_dispositivo not in nombres:
        # El nombre ayuda a diagnosticar; el indice sigue siendo la fuente de verdad.
        nombre_dispositivo = _nombre_dispositivo(int(indice), nombres)
    elif not nombre_dispositivo and nombres and 0 <= int(indice) < len(nombres):
        nombre_dispositivo = nombres[int(indice)]

    if sys.platform == "win32":
        for intento in range(3):
            captura = _abrir_captura_backend(
                int(indice),
                cv2.CAP_DSHOW,
                camera_width,
                camera_height,
                camera_fps,
                nombre_dispositivo,
            )
            if captura is not None:
                return _registrar_captura_camara_activa(captura)
            if intento < 2:
                time.sleep(0.35)
        return cv2.VideoCapture()

    for intento in range(3):
        captura = _abrir_captura_backend(
            int(indice),
            cv2.CAP_ANY,
            camera_width,
            camera_height,
            camera_fps,
            nombre_dispositivo,
        )
        if captura is not None:
            return _registrar_captura_camara_activa(captura)
        if intento < 2:
            time.sleep(0.35)
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
            nombre,
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


def opciones_camara_para_ui() -> list[dict]:
    """Lista de camaras registradas en Windows para el selector de Streamlit."""
    nombres = listar_nombres_camara_dshow() or []
    resumen = resumen_camaras_sistema()
    indice_default = (resumen.get("indices_camo") or resumen.get("indices_iriun") or [0])[0]
    if not nombres:
        return [{"indice": i, "etiqueta": f"Camara {i}"} for i in range(4)]
    opciones = []
    for i, nombre in enumerate(nombres):
        opciones.append({"indice": i, "etiqueta": f"{i} — {nombre}"})
    for item in opciones:
        item["default"] = item["indice"] == indice_default
    return opciones


def mensaje_error_apertura_camara(indice: int) -> str:
    nombres = listar_nombres_camara_dshow() or []
    base = (
        "No se pudo abrir la camara. Cierre otras apps que la usen (Zoom, Camo Studio preview duplicado, OBS) "
        "y vuelva a intentar."
    )
    if not nombres:
        return f"{base} Indice solicitado: {indice}."
    lista = " · ".join(f"{i}={nombre}" for i, nombre in enumerate(nombres))
    if indice >= len(nombres):
        return (
            f"{base} Windows solo reporta {len(nombres)} camara(s): {lista}. "
            f"El indice {indice} no existe; elija 0–{len(nombres) - 1}."
        )
    nombre = nombres[indice]
    tipo = _clasificar_dispositivo(nombre)
    ayuda_tipo = ""
    if tipo == "camo":
        ayuda_tipo = (
            " Abra Camo Studio, conecte el iPhone y confirme que la imagen se vea dentro de Camo "
            "antes de iniciar Monitoreo."
        )
    elif tipo == "iriun":
        ayuda_tipo = (
            " Abra Iriun Webcam tanto en el telefono como en Windows y espere a que muestre video "
            "antes de iniciar Monitoreo."
        )
    return f"{base}{ayuda_tipo} Camaras disponibles: {lista}. Indice solicitado: {indice} ({nombre})."

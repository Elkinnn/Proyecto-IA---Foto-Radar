"""Simulacion academica de velocidad con dos lineas virtuales."""

from pathlib import Path
import json

import cv2
import numpy as np

from src.fuzzy_system import clasificar_velocidad
from src.plate_reader import asegurar_rgb
from src.speed_estimator import SpeedTracker


def _centro_y_simulado(
    numero_frame: int,
    frame_inicial: int,
    frame_linea_1: int,
    frame_linea_2: int,
    total_frames: int,
    linea_1_y: int,
    linea_2_y: int,
    alto_frame: int,
) -> float:
    margen_superior = 90
    margen_inferior = alto_frame - 90
    inicio_y = max(20, linea_1_y - 180)
    fin_y = min(margen_inferior, linea_2_y + 180)

    if numero_frame <= frame_linea_1:
        denom = max(frame_linea_1 - frame_inicial, 1)
        avance = max(numero_frame - frame_inicial, 0) / denom
        return inicio_y + avance * (linea_1_y - inicio_y)

    if numero_frame <= frame_linea_2:
        denom = max(frame_linea_2 - frame_linea_1, 1)
        avance = (numero_frame - frame_linea_1) / denom
        return linea_1_y + avance * (linea_2_y - linea_1_y)

    denom = max(total_frames - frame_linea_2, 1)
    avance = min((numero_frame - frame_linea_2) / denom, 1.0)
    return linea_2_y + avance * (fin_y - linea_2_y)

def _crear_frame_simulacion_velocidad(
    numero_frame: int,
    bbox: list[int],
    linea_1_y: int,
    linea_2_y: int,
    distancia_metros: float,
    velocidad: dict,
) -> np.ndarray:
    ancho_frame = 640
    alto_frame = 900
    frame = np.full((alto_frame, ancho_frame, 3), 38, dtype=np.uint8)

    cv2.line(frame, (0, linea_1_y), (ancho_frame, linea_1_y), (255, 170, 0), 3)
    cv2.line(frame, (0, linea_2_y), (ancho_frame, linea_2_y), (0, 80, 255), 3)
    cv2.putText(frame, "Linea 1", (20, linea_1_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 0), 2)
    cv2.putText(frame, "Linea 2", (20, linea_2_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
    centro_x = int((x1 + x2) / 2)
    centro_y = int((y1 + y2) / 2)
    cv2.circle(frame, (centro_x, centro_y), 7, (255, 255, 255), -1)
    cv2.circle(frame, (centro_x, centro_y), 9, (0, 0, 0), 1)

    velocidad_kmh = velocidad.get("velocidad_kmh")
    texto_velocidad = "Pendiente" if velocidad_kmh is None else f"{velocidad_kmh:.2f} km/h"
    cv2.putText(frame, f"Frame: {numero_frame}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    cv2.putText(frame, f"Estado: {velocidad.get('estado')}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    cv2.putText(frame, f"Distancia: {distancia_metros:.1f} m", (20, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    cv2.putText(frame, f"Velocidad: {texto_velocidad}", (20, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return frame

def ejecutar_simulacion_velocidad(
    fps: float,
    distancia_metros: float,
    limite_kmh: float,
    placa_manual: str,
    posicion_linea_1: float,
    posicion_linea_2: float,
    frame_inicial: int,
    frame_linea_1: int,
    frame_linea_2: int,
    total_frames: int,
) -> dict:
    ancho_frame = 640
    alto_frame = 900
    linea_1_y = int(alto_frame * posicion_linea_1)
    linea_2_y = int(alto_frame * posicion_linea_2)
    tracker = SpeedTracker(linea_1_y, linea_2_y, distancia_metros, fps)

    evidencia_dir = Path("reports") / "evidencias" / "simulacion_velocidad"
    evidencia_dir.mkdir(parents=True, exist_ok=True)
    rutas = {
        "linea_1": evidencia_dir / "frame_cruce_linea_1.jpg",
        "linea_2": evidencia_dir / "frame_cruce_linea_2.jpg",
        "calculada": evidencia_dir / "frame_velocidad_calculada.jpg",
        "json": evidencia_dir / "simulacion_velocidad.json",
    }

    for ruta in rutas.values():
        if ruta.exists():
            try:
                ruta.unlink()
            except PermissionError:
                pass

    ultimo_frame = None
    resumen = tracker.resumen()
    guardo_linea_1 = False
    guardo_linea_2 = False
    guardo_calculada = False

    for numero_frame in range(max(frame_inicial, 0), total_frames + 1):
        centro_y = _centro_y_simulado(
            numero_frame,
            frame_inicial,
            frame_linea_1,
            frame_linea_2,
            total_frames,
            linea_1_y,
            linea_2_y,
            alto_frame,
        )
        centro_x = ancho_frame // 2
        bbox = [
            int(centro_x - 95),
            int(centro_y - 28),
            int(centro_x + 95),
            int(centro_y + 28),
        ]
        resumen = tracker.actualizar(bbox, numero_frame)
        frame = _crear_frame_simulacion_velocidad(numero_frame, bbox, linea_1_y, linea_2_y, distancia_metros, resumen)

        if resumen.get("frame_cruce_linea_1") == numero_frame and not guardo_linea_1:
            cv2.imwrite(str(rutas["linea_1"]), frame)
            guardo_linea_1 = True
        if resumen.get("frame_cruce_linea_2") == numero_frame and not guardo_linea_2:
            cv2.imwrite(str(rutas["linea_2"]), frame)
            guardo_linea_2 = True
        if resumen.get("velocidad_kmh") is not None and not guardo_calculada:
            cv2.imwrite(str(rutas["calculada"]), frame)
            guardo_calculada = True

        ultimo_frame = frame

    datos = {
        "placa": placa_manual,
        "fps": fps,
        "distancia_metros": distancia_metros,
        "limite_kmh": limite_kmh,
        "posicion_linea_1": posicion_linea_1,
        "posicion_linea_2": posicion_linea_2,
        "frame_cruce_linea_1": resumen.get("frame_cruce_linea_1"),
        "frame_cruce_linea_2": resumen.get("frame_cruce_linea_2"),
        "frame_cruce_linea_1_exacto": resumen.get("frame_cruce_linea_1_exacto"),
        "frame_cruce_linea_2_exacto": resumen.get("frame_cruce_linea_2_exacto"),
        "tiempo_cruce_linea_1": resumen.get("tiempo_cruce_linea_1"),
        "tiempo_cruce_linea_2": resumen.get("tiempo_cruce_linea_2"),
        "tiempo_segundos": resumen.get("tiempo_entre_lineas"),
        "velocidad_kmh": resumen.get("velocidad_kmh"),
        "estado": resumen.get("estado"),
        "metodo_medicion": resumen.get("metodo_medicion"),
        "formula_medicion": resumen.get("formula_medicion"),
        "ruta_frame_cruce_linea_1": str(rutas["linea_1"]) if guardo_linea_1 else None,
        "ruta_frame_cruce_linea_2": str(rutas["linea_2"]) if guardo_linea_2 else None,
        "ruta_frame_velocidad_calculada": str(rutas["calculada"]) if guardo_calculada else None,
    }
    if resumen.get("velocidad_kmh") is not None:
        difuso = clasificar_velocidad(resumen["velocidad_kmh"], limite_kmh)
        datos.update(
            {
                "velocidad": resumen["velocidad_kmh"],
                "resultado_difuso": difuso,
                "estado_difuso": difuso["estado"],
                "nivel_infraccion": difuso["nivel_infraccion"],
                "sancion": difuso["sancion"],
                "horas_suspension": difuso["horas_suspension"],
                "multa_usd": difuso.get("multa_usd"),
                "multa_texto": difuso.get("multa_texto"),
                "mensaje_difuso": difuso["mensaje"],
                "grados_pertenencia": difuso["grados"],
                "metodo_difuso": difuso.get("metodo"),
            }
        )
    else:
        difuso = None

    with open(rutas["json"], "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)

    return {
        "resumen": resumen,
        "difuso": difuso,
        "datos": datos,
        "frame_final_rgb": asegurar_rgb(ultimo_frame) if ultimo_frame is not None else None,
        "ruta_json": str(rutas["json"]),
    }


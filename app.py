from datetime import datetime
from pathlib import Path
import csv
import json
import threading
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import (
    generar_video_demo_anotado,
    procesar_camara_monitoreo,
    procesar_frame_video_monitoreo,
    procesar_imagen_prueba,
    procesar_video_monitoreo,
)
from src.camera_utils import (
    frame_tiene_senal,
    indices_a_escanear,
    probar_indices_camara,
    resumen_camaras_sistema,
)
from src.fuzzy_system import clasificar_velocidad
from src.notifier import evaluar_calidad_evento, notificar_evento_placa, validar_correo
from src.plate_detector import PlateDetector
from src.plate_reader import (
    asegurar_rgb,
    consolidar_lecturas_evento_placa,
    guardar_debug_votacion_evento,
    leer_placa_cnn_seguro_desde_monitoreo,
    leer_placa_desde_recorte,
    preprocesar_placa,
    registrar_diagnostico_recorte,
    registrar_reporte_ocr,
    segmentar_caracteres_v2,
)
from src.speed_estimator import SpeedTracker
from src.utils import cargar_config, cargar_variables_entorno, guardar_archivo_subido

cargar_variables_entorno()
from scripts.generar_caracteres_desde_placas_ecuador import (
    CARACTERES_DIR as CARACTERES_ECUADOR_DIR,
    CLASES as CLASES_CARACTERES_ECUADOR,
    listar_imagenes_placas,
    normalizar_placa_esperada,
    preparar_recorte_placa,
    guardar_caracteres as guardar_caracteres_ecuador,
    registrar_auditoria_guardado as registrar_auditoria_guardado_caracteres,
    validar_placa_para_guardado,
    leer_labels as leer_labels_caracteres_ecuador,
    escribir_labels as escribir_labels_caracteres_ecuador,
)


PERFORMANCE_MODE_LABELS = {
    "Rapido": "rapido",
    "Balanceado": "balanceado",
    "Preciso": "preciso",
}

_OCR_ASYNC_LOCK = threading.Lock()
_OCR_ASYNC_PENDIENTES: set[str] = set()
_OCR_ASYNC_RESULTADOS: dict[str, dict] = {}


def _config_modo_rendimiento(config: dict, modo: str) -> dict:
    clave = PERFORMANCE_MODE_LABELS.get(modo, "balanceado")
    defaults = {
        "rapido": {"yolo_every_n_frames": 5, "inference_size": 416, "render_every_n_frames": 1, "history_max": 10, "max_display_fps": 0},
        "balanceado": {"yolo_every_n_frames": 5, "inference_size": 512, "render_every_n_frames": 1, "history_max": 10, "max_display_fps": 0},
        "preciso": {"yolo_every_n_frames": 3, "inference_size": 640, "render_every_n_frames": 1, "history_max": 10, "max_display_fps": 0},
    }
    salida = defaults[clave].copy()
    salida.update((config.get("monitoring_performance") or {}).get(clave, {}))
    return salida


def _config_rendimiento_monitoreo(config: dict, modo: str, fuente_monitoreo: str) -> dict:
    base = _config_modo_rendimiento(config, modo)
    perf = config.get("monitoring_performance") or {}
    clave_fuente = "video" if fuente_monitoreo == "Video de prueba" else "camara"
    base.update(perf.get(clave_fuente, {}) or {})
    return base


def _snap_a_opcion_slider(valor: int, opciones: list[int], default: int | None = None) -> int:
    if valor in opciones:
        return valor
    if not opciones:
        return int(default or 0)
    return min(opciones, key=lambda opt: abs(int(opt) - int(valor)))


def _redimensionar_frame_rgb(frame_rgb, ancho_maximo: int = 800):
    if frame_rgb is None or ancho_maximo <= 0:
        return frame_rgb
    alto, ancho = frame_rgb.shape[:2]
    if ancho <= ancho_maximo:
        return frame_rgb
    escala = ancho_maximo / float(ancho)
    nuevo_alto = max(1, int(alto * escala))
    return cv2.resize(frame_rgb, (ancho_maximo, nuevo_alto), interpolation=cv2.INTER_LINEAR)


def _preparar_imagen_streamlit(frame_rgb, ancho_maximo: int = 640):
    return _redimensionar_frame_rgb(frame_rgb, ancho_maximo)


st.set_page_config(
    page_title="Fotorradar Ecuador IA",
    page_icon=":vertical_traffic_light:",
    layout="wide",
)


def panel_resultados(evento: dict | None) -> None:
    st.subheader("Panel de resultados")

    if not evento:
        col1, col2, col3 = st.columns(3)
        col1.metric("Placa detectada", "Pendiente")
        col2.metric("Velocidad", "Pendiente")
        col3.metric("Estado", "Sin evento")
        st.info("La deteccion de placa, reconocimiento de caracteres CNN, velocidad real y sanciones se integraran despues del flujo de video.")
        return

    clasificacion = evento.get("clasificacion_difusa") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Placa detectada", "Si" if evento.get("placa_detectada") else "No")
    col2.metric("Texto reconocido", evento.get("texto_placa") or "Pendiente")
    col3.metric("Velocidad km/h", f"{evento.get('velocidad_kmh', 0):.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Clasificacion", clasificacion.get("estado", "Pendiente"))
    col5.metric("Nivel infraccion", clasificacion.get("nivel_infraccion", "Pendiente"))
    col6.metric("Sancion", "Si" if evento.get("sancion_generada") else "No")

    evidencia = evento.get("ruta_reporte")
    if evidencia:
        st.caption(f"Evidencia guardada: {evidencia}")


def mostrar_resultado_prueba(resultado: dict) -> None:
    col_media, col_info = st.columns([1.2, 1])

    with col_media:
        st.subheader("Archivo procesado")
        ruta_mostrar = resultado.get("ruta_archivo_procesado") or resultado.get("ruta_archivo")
        if ruta_mostrar and Path(ruta_mostrar).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            st.image(ruta_mostrar, use_container_width=True)
        elif resultado.get("ruta_archivo"):
            st.video(resultado["ruta_archivo"])

    with col_info:
        panel_resultados(resultado)


def _etiqueta_estado_flujo(estado: str) -> str:
    etiquetas = {
        "completado": "Completado",
        "pendiente": "Pendiente",
        "no_aplica": "No aplica",
        "advertencia": "Error / advertencia",
    }
    return etiquetas.get(estado, estado)


def _mostrar_estado_flujo(resumen: dict, velocidad: dict, velocidad_kmh: float | None) -> None:
    st.subheader("Estado del flujo")

    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    placa_detectada = bool(ultima_deteccion) or resumen.get("estado_placa") in {"Detectada", "Mantenida"}
    frame_linea_1 = velocidad.get("frame_cruce_linea_1")
    frame_linea_2 = velocidad.get("frame_cruce_linea_2")
    difuso = resumen.get("clasificacion_difusa")
    notificacion = resumen.get("notificacion_correo")

    if notificacion and notificacion.get("enviado"):
        estado_correo = "completado"
    elif notificacion and notificacion.get("modo") in {"simulado", "error"}:
        estado_correo = "advertencia"
    elif notificacion and notificacion.get("modo") == "descartado":
        estado_correo = "no_aplica"
    else:
        estado_correo = "pendiente"

    estados = [
        ("DetecciÃ³n de placa", "completado" if placa_detectada else "pendiente"),
        ("Cruce LÃ­nea 1", "completado" if frame_linea_1 else "pendiente"),
        ("Cruce LÃ­nea 2", "completado" if frame_linea_2 else "pendiente"),
        ("Velocidad calculada", "completado" if velocidad_kmh is not None else "pendiente"),
        ("ClasificaciÃ³n difusa", "completado" if difuso else ("pendiente" if velocidad_kmh is not None else "no_aplica")),
        ("NotificaciÃ³n por correo", estado_correo),
    ]

    columnas = st.columns(4)
    for idx, (nombre, estado) in enumerate(estados):
        columnas[idx % 4].metric(nombre, _etiqueta_estado_flujo(estado))

    col_det1, col_det2, col_det3 = st.columns(3)
    col_det1.metric("Placa detectada", "SÃ­" if placa_detectada else "No")
    confianza = ultima_deteccion.get("confianza")
    col_det2.metric("Confianza", f"{float(confianza):.2f}" if confianza is not None else "Pendiente")
    mejor_confianza = resumen.get("mejor_confianza_evento")
    col_det3.metric("Mejor confianza evento", f"{mejor_confianza:.2f}" if mejor_confianza is not None else "Pendiente")

    ruta_mejor_recorte = resumen.get("ruta_mejor_recorte_evento") or resumen.get("ultimo_recorte_placa")
    if ruta_mejor_recorte:
        st.caption(f"Mejor recorte de placa: {ruta_mejor_recorte}")
        st.image(ruta_mejor_recorte, use_container_width=False)

    col_cruce1, col_cruce2 = st.columns(2)
    col_cruce1.metric("Cruce LÃ­nea 1", "SÃ­" if frame_linea_1 else "No")
    col_cruce1.caption(f"Frame LÃ­nea 1: {frame_linea_1 or 'Pendiente'}")
    col_cruce2.metric("Cruce LÃ­nea 2", "SÃ­" if frame_linea_2 else "No")
    col_cruce2.caption(f"Frame LÃ­nea 2: {frame_linea_2 or 'Pendiente'}")

    if frame_linea_1 and not frame_linea_2:
        st.warning("La placa cruzÃ³ la LÃ­nea 1, pero no cruzÃ³ la LÃ­nea 2. No se puede calcular velocidad hasta completar el cruce entre ambas lÃ­neas.")
        st.info("Esperando cruce de LÃ­nea 2.")
    elif not frame_linea_1:
        st.info("Esperando que el centro de la placa cruce la LÃ­nea 1.")


def _mostrar_diagnostico_monitoreo(resumen: dict, velocidad: dict, velocidad_kmh: float | None) -> None:
    st.subheader("DiagnÃ³stico del monitoreo")

    placa_detectada = bool(resumen.get("ultima_deteccion")) or resumen.get("estado_placa") in {"Detectada", "Mantenida"}
    frame_linea_1 = velocidad.get("frame_cruce_linea_1")
    frame_linea_2 = velocidad.get("frame_cruce_linea_2")

    if velocidad_kmh is not None:
        motivo = "La mediciÃ³n de velocidad se completÃ³ correctamente."
        recomendacion = "Revise el resultado difuso, la consulta en base de datos y la notificaciÃ³n si corresponde."
    elif not placa_detectada:
        motivo = "No se detectÃ³ placa vÃ¡lida."
        recomendacion = "Ajuste confianza mÃ­nima, iluminaciÃ³n, enfoque, rotaciÃ³n, zona de cÃ¡mara o posiciÃ³n del vehÃ­culo."
    elif not frame_linea_1:
        motivo = "La placa fue detectada, pero el centro de la placa todavÃ­a no cruzÃ³ la LÃ­nea 1."
        recomendacion = "Ubique la LÃ­nea 1 sobre la trayectoria real de la placa o use un video donde el vehÃ­culo avance hacia ambas lÃ­neas."
    elif frame_linea_1 and not frame_linea_2:
        motivo = "La placa no cruzÃ³ LÃ­nea 2."
        recomendacion = "Use un video donde el vehÃ­culo pase completamente entre ambas lÃ­neas o ajuste la posiciÃ³n de LÃ­nea 2."
    else:
        motivo = "La detecciÃ³n fue vÃ¡lida, pero no hubo movimiento suficiente para medir velocidad."
        recomendacion = "Verifique que la cÃ¡mara estÃ© fija, que la placa se desplace de arriba hacia abajo y que las lÃ­neas estÃ©n separadas correctamente."

    col_diag1, col_diag2 = st.columns(2)
    col_diag1.info(f"Motivo: {motivo}")
    col_diag2.info(f"RecomendaciÃ³n: {recomendacion}")

    mensaje_detector = resumen.get("mensaje_detector")
    if mensaje_detector:
        st.caption(f"Detector: {mensaje_detector}")


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


def _buscar_recortes_ocr() -> tuple[list[Path], str]:
    eventos_dir = Path("reports") / "evidencias" / "eventos_placa"
    placas_dir = Path("reports") / "evidencias" / "placas_detectadas"
    extensiones = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    recortes_eventos = []
    for patron in extensiones:
        recortes_eventos.extend(eventos_dir.glob(patron) if eventos_dir.exists() else [])
    recortes_eventos = [ruta for ruta in recortes_eventos if "recorte" in ruta.name.lower()]
    if recortes_eventos:
        return sorted(recortes_eventos, key=lambda ruta: ruta.stat().st_mtime, reverse=True), "eventos_placa"

    recortes_placas = []
    for patron in extensiones:
        recortes_placas.extend(placas_dir.glob(patron) if placas_dir.exists() else [])
    return sorted(recortes_placas, key=lambda ruta: ruta.stat().st_mtime, reverse=True), "placas_detectadas"


def mostrar_resumen_monitoreo(resumen: dict) -> None:
    st.subheader("Resumen del monitoreo")
    col1, col2, col3 = st.columns(3)
    col1.metric("FPS original", f"{resumen.get('fps', 0):.2f}")
    col2.metric("Frames totales", resumen.get("total_frames", 0) or "No disponible")
    col3.metric("Duracion total", f"{resumen.get('duracion_segundos', 0):.1f} s")

    col4, col5, col6 = st.columns(3)
    col4.metric("Frames procesados", resumen.get("frames_procesados", 0))
    col5.metric("Segundos procesados", f"{resumen.get('segundos_procesados', 0):.1f} s")
    col6.metric("Modo procesamiento", resumen.get("modo_procesamiento", "Pendiente"))

    col7, col8, col9 = st.columns(3)
    col7.metric("Resolucion procesada", f"{resumen.get('ancho', 0)} x {resumen.get('alto', 0)}")
    col8.metric("Fuente", resumen.get("fuente", "Pendiente"))
    col9.metric("Velocidad", resumen.get("velocidad_reproduccion", "Pendiente"))

    col10, _, _ = st.columns(3)
    col10.metric("Rotacion aplicada", resumen.get("rotacion", "Sin rotaciÃ³n"))

    st.info(resumen.get("mensaje_estado", "Sin estado disponible."))
    if not resumen.get("modelo_detector_disponible"):
        st.warning(resumen.get("mensaje_detector", "Modelo de placa no encontrado. Entrene primero el detector."))

    st.subheader("Deteccion de placas")
    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    confianza = float(ultima_deteccion.get("confianza", 0.0))
    col_det1, col_det2, col_det3 = st.columns(3)
    col_det1.metric("Estado de placa", resumen.get("estado_placa", "Pendiente"))
    col_det2.metric("Confianza", f"{confianza:.2f}" if ultima_deteccion else "Pendiente")
    col_det3.metric("Detecciones instantaneas", resumen.get("detecciones_frame", 0))

    col_det4, col_det5 = st.columns(2)
    col_det4.metric("Eventos de placa", resumen.get("eventos_placa", resumen.get("placas_detectadas", 0)))
    col_det5.metric("Frames desde ultima deteccion", resumen.get("frames_desde_ultima_deteccion", 0))

    col_det6, col_det7, col_det8 = st.columns(3)
    col_det6.metric("Detecciones brutas YOLO", resumen.get("detecciones_brutas", 0))
    col_det7.metric("Detecciones validas", resumen.get("detecciones_validas", 0))
    col_det8.metric("Motivo rechazo", resumen.get("motivos_rechazo") or "Ninguno")

    st.subheader("Evento de placa")
    mejor_confianza = resumen.get("mejor_confianza_evento")
    col_evt1, col_evt2, col_evt3 = st.columns(3)
    col_evt1.metric("Evento activo", "Si" if resumen.get("evento_activo") else "No")
    col_evt2.metric("ID evento actual", resumen.get("evento_id", "Pendiente"))
    col_evt3.metric("Confianza actual", f"{confianza:.2f}" if ultima_deteccion else "Pendiente")

    col_evt4, col_evt5, col_evt6 = st.columns(3)
    col_evt4.metric("Mejor confianza evento", f"{mejor_confianza:.2f}" if mejor_confianza is not None else "Pendiente")
    col_evt5.metric("Frame mejor deteccion", resumen.get("frame_mejor_evento") or "Pendiente")
    col_evt6.metric("Frames sin deteccion", resumen.get("frames_sin_deteccion", 0))

    ruta_mejor_recorte = resumen.get("ruta_mejor_recorte_evento")
    if ruta_mejor_recorte:
        st.caption(f"Mejor recorte del evento: {ruta_mejor_recorte}")
        st.image(ruta_mejor_recorte, use_container_width=False)

    st.subheader("Velocidad")
    velocidad = resumen.get("velocidad") or {}
    estado_velocidad = velocidad.get("estado", "esperando_linea_1")
    etiquetas_estado = {
        "esperando_linea_1": "Esperando linea 1",
        "esperando_linea_2": "Esperando linea 2",
        "velocidad_calculada": "Calculada",
        "medicion_invalida": "Invalida",
    }
    col_vel1, col_vel2, col_vel3 = st.columns(3)
    col_vel1.metric("Estado de velocidad", etiquetas_estado.get(estado_velocidad, estado_velocidad))
    frame_l1_x = velocidad.get("frame_cruce_linea_1_exacto")
    frame_l2_x = velocidad.get("frame_cruce_linea_2_exacto")
    col_vel2.metric(
        "TIC (Linea 1)",
        f"frame {frame_l1_x:.2f}" if frame_l1_x is not None else (velocidad.get("frame_cruce_linea_1") or "Pendiente"),
    )
    col_vel3.metric(
        "TOC (Linea 2)",
        f"frame {frame_l2_x:.2f}" if frame_l2_x is not None else (velocidad.get("frame_cruce_linea_2") or "Pendiente"),
    )

    col_vel4, col_vel5, col_vel6 = st.columns(3)
    tiempo_entre = velocidad.get("tiempo_entre_lineas")
    velocidad_kmh = velocidad.get("velocidad_kmh")
    col_vel4.metric("Delta t (tic-toc)", f"{tiempo_entre:.4f} s" if tiempo_entre is not None else "Pendiente")
    col_vel5.metric("Distancia calibrada", f"{velocidad.get('distancia_metros', resumen.get('distancia_lineas_m', 0)):.1f} m")
    col_vel6.metric("Velocidad estimada", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "Pendiente")

    if velocidad.get("formula_medicion"):
        st.caption(f"Formula: {velocidad['formula_medicion']}")
    elif velocidad.get("motivo_invalido"):
        st.warning(velocidad["motivo_invalido"])

    col_vel7, col_vel8, col_vel9 = st.columns(3)
    col_vel7.metric("Posicion Linea 1", f"{resumen.get('posicion_linea_1', 0.45):.2f}")
    col_vel8.metric("Posicion Linea 2", f"{resumen.get('posicion_linea_2', 0.65):.2f}")
    col_vel9.metric("Estado de cruce", etiquetas_estado.get(estado_velocidad, estado_velocidad))

    _mostrar_estado_flujo(resumen, velocidad, velocidad_kmh)

    st.subheader("ClasificaciÃ³n difusa")
    if velocidad_kmh is not None:
        try:
            from src.utils import cargar_config

            cfg_fuzzy = cargar_config()
        except OSError:
            cfg_fuzzy = None
        difuso = resumen.get("clasificacion_difusa") or clasificar_velocidad(
            float(velocidad_kmh),
            _obtener_limite_kmh_resumen(resumen, cfg_fuzzy),
            cfg_fuzzy,
        )
        col_dif1, col_dif2, col_dif3 = st.columns(3)
        col_dif1.metric("Estado difuso", difuso["estado"])
        col_dif2.metric("Multa USD", f"${difuso.get('multa_usd', 0):.2f}")
        col_dif3.metric("Horas de suspensiÃ³n", difuso["horas_suspension"])

        col_dif4, col_dif5, col_dif6 = st.columns(3)
        col_dif4.metric("Nivel de infracciÃ³n", difuso["nivel_infraccion"])
        col_dif5.metric("SanciÃ³n", difuso["sancion"])
        col_dif6.metric("LÃ­mite evaluado", f"{difuso['limite_kmh']:.1f} km/h")
        st.info(difuso["mensaje"])
        st.caption(f"Metodo: {difuso.get('metodo', '—')} · {difuso.get('multa_texto', '')}")
    else:
        col_dif1, col_dif2, col_dif3 = st.columns(3)
        col_dif1.metric("Estado difuso", "Pendiente")
        col_dif2.metric("Nivel de infracciÃ³n", "Pendiente")
        col_dif3.metric("SanciÃ³n", "Pendiente")

    notificacion = resumen.get("notificacion_correo")
    if resumen.get("placa_controlada") or notificacion:
        st.subheader("Resultado del evento y notificacion")
        difuso_evento = resumen.get("clasificacion_difusa") or {}
        evaluacion = resumen.get("evaluacion_calidad_evento") or {}

        col_evt_res1, col_evt_res2, col_evt_res3 = st.columns(3)
        col_evt_res1.metric("Placa reconocida", resumen.get("placa_controlada", "Pendiente"))
        col_evt_res2.metric("Apta para envio", "Si" if evaluacion.get("apto") else "No")
        col_evt_res3.metric("Velocidad (demo)", f"{difuso_evento.get('velocidad_kmh', 0):.1f} km/h" if difuso_evento else "Pendiente")

        col_evt_res4, col_evt_res5, col_evt_res6 = st.columns(3)
        col_evt_res4.metric("Estado difuso", difuso_evento.get("estado", "Pendiente"))
        col_evt_res5.metric("Nivel de infraccion", difuso_evento.get("nivel_infraccion", "Pendiente"))
        col_evt_res6.metric("Sancion", difuso_evento.get("sancion", "Pendiente"))

        if evaluacion and not evaluacion.get("apto") and evaluacion.get("razones"):
            st.warning("No se envio correo. Motivos del control de calidad:\n- " + "\n- ".join(evaluacion["razones"]))

        if notificacion:
            estado_envio = notificacion.get("estado")
            modo = notificacion.get("modo")
            col_not1, col_not2, col_not3 = st.columns(3)
            col_not1.metric("Destinatario", notificacion.get("destinatario") or "Sin correo")
            col_not2.metric("Envio", "Enviado" if notificacion.get("enviado") else (modo or "Pendiente"))
            col_not3.metric("Estado", estado_envio or "Pendiente")
            if notificacion.get("enviado"):
                st.success(notificacion.get("mensaje_estado", "Correo enviado."))
            elif modo == "simulado":
                st.info(notificacion.get("mensaje_estado", "Correo generado en modo simulado."))
            elif modo == "sin_destino":
                st.info(notificacion.get("mensaje_estado", "Ingrese un correo destino."))
            elif modo in {"error", "descartado"}:
                st.warning(notificacion.get("mensaje_estado", notificacion.get("error", "No se envio el correo.")))
            for adjunto in notificacion.get("adjuntos", []) or []:
                if Path(str(adjunto)).exists():
                    st.image(str(adjunto), caption=Path(str(adjunto)).name, use_container_width=False)

    _mostrar_diagnostico_monitoreo(resumen, velocidad, velocidad_kmh)

    ultimo_recorte = resumen.get("ultimo_recorte_placa")
    if ultimo_recorte:
        st.caption(f"Ultimo recorte de placa: {ultimo_recorte}")
        st.image(ultimo_recorte, use_container_width=False)

    primer_frame = resumen.get("primer_frame_evidencia")
    ultimo_frame = resumen.get("ultimo_frame_evidencia")
    if primer_frame or ultimo_frame:
        ev1, ev2 = st.columns(2)
        if primer_frame:
            ev1.caption(f"Primer frame: {primer_frame}")
            ev1.image(primer_frame, use_container_width=True)
        if ultimo_frame:
            ev2.caption(f"Ultimo frame: {ultimo_frame}")
            ev2.image(ultimo_frame, use_container_width=True)


def _metricas_calidad_desde_resumen(resumen: dict, placa: str) -> dict:
    mejor_info = resumen.get("mejor_recorte_placa_info") or {}
    cantidad = resumen.get("cantidad_caracteres_segmentados_ocr")
    return {
        "placa": placa,
        "formato_valido": bool(resumen.get("formato_consolidado_valido") or resumen.get("formato_ocr_valido")),
        "confianza_final": resumen.get("confianza_final_evento") or resumen.get("confianza_ocr"),
        "aspect_ratio": mejor_info.get("aspect_ratio"),
        "nitidez": mejor_info.get("nitidez"),
        "cerca_borde": mejor_info.get("cerca_borde"),
        "cantidad_caracteres": cantidad if cantidad else len(placa or ""),
    }


def _adjuntos_evento_desde_resumen(resumen: dict) -> list[str]:
    candidatos_frame = [
        resumen.get("ruta_mejor_frame_evento"),
        resumen.get("mejor_frame_bbox_placa"),
        resumen.get("ruta_frame_evento_en_vivo"),
        resumen.get("mejor_frame_recorte_placa"),
        resumen.get("ultimo_frame_deteccion"),
    ]
    candidatos_recorte = [
        resumen.get("ruta_mejor_recorte_evento"),
        resumen.get("mejor_recorte_placa"),
        resumen.get("ruta_recorte_evento_en_vivo"),
        resumen.get("ultimo_recorte_placa"),
    ]

    def _primera_ruta_valida(rutas: list) -> str | None:
        for ruta in rutas:
            if not ruta:
                continue
            ruta_str = str(ruta)
            if Path(ruta_str).exists():
                return ruta_str
        return None

    frame = _primera_ruta_valida(candidatos_frame)
    recorte = _primera_ruta_valida(candidatos_recorte)

    adjuntos: list[str] = []
    if frame:
        adjuntos.append(frame)
    if recorte and recorte not in adjuntos:
        adjuntos.append(recorte)
    return adjuntos


def _tiene_recorte_placa_valido(resumen: dict) -> bool:
    ruta = (
        resumen.get("mejor_recorte_placa")
        or resumen.get("ruta_mejor_recorte_evento")
        or resumen.get("ultimo_recorte_placa")
    )
    return bool(ruta and Path(str(ruta)).exists())


def preparar_estado_notificacion_evento(resumen: dict, placa_evento: str, config: dict) -> dict:
    """Evalua calidad y deja listo el resumen para envio (manual o automatico)."""
    placa = (placa_evento or "").strip().upper()
    metricas = _metricas_calidad_desde_resumen(resumen, placa)
    evaluacion = evaluar_calidad_evento(metricas, config)
    resumen["evaluacion_calidad_evento"] = evaluacion
    resumen["placa_controlada"] = placa
    resumen["puede_enviar_notificacion"] = bool(
        evaluacion.get("apto") and _tiene_recorte_placa_valido(resumen)
    )
    return resumen


def enviar_notificacion_evento_monitoreo(
    resumen: dict,
    placa_evento: str,
    correo_destino: str,
    config: dict,
) -> dict:
    """Envia correo para un evento (usado por el envio automatico)."""
    placa = (placa_evento or "").strip().upper()
    resumen = preparar_estado_notificacion_evento(resumen, placa, config)
    resumen = _aplicar_multa_difusa_resumen(resumen, config)
    evaluacion = resumen.get("evaluacion_calidad_evento") or {}

    if not validar_correo(correo_destino):
        resumen["notificacion_correo"] = {
            "enviado": False,
            "modo": "sin_destino",
            "estado": "correo_no_ingresado",
            "mensaje_estado": "Ingrese un correo destino valido para enviar la notificacion.",
        }
        return resumen

    if not resumen.get("puede_enviar_notificacion"):
        razones = evaluacion.get("razones") or ["No hay recorte de placa valido o la lectura no es confiable."]
        resumen["notificacion_correo"] = {
            "enviado": False,
            "modo": "bloqueado",
            "estado": "no_apto",
            "mensaje_estado": "No se puede enviar: " + "; ".join(razones),
            "evaluacion_calidad": evaluacion,
        }
        return resumen

    metricas = _metricas_calidad_desde_resumen(resumen, placa)
    adjuntos = _adjuntos_evento_desde_resumen(resumen)
    velocidad_kmh = _obtener_velocidad_kmh_desde_resumen(resumen)
    limite_kmh = _obtener_limite_kmh_resumen(resumen, config)
    notificacion = notificar_evento_placa(
        destinatario=correo_destino,
        placa=placa,
        metricas_calidad=metricas,
        adjuntos=adjuntos,
        config=config,
        evento_id=resumen.get("evento_id") or resumen.get("eventos_placa") or "monitoreo",
        contexto={
            "fecha_hora": datetime.now().isoformat(timespec="seconds"),
            "velocidad_kmh": velocidad_kmh,
            "limite_kmh": limite_kmh,
        },
        velocidad_kmh=velocidad_kmh,
        limite_kmh=limite_kmh,
    )

    resumen["clasificacion_difusa"] = notificacion.get("clasificacion_difusa")
    resumen["notificacion_correo"] = notificacion
    st.session_state.evento_monitoreo_actual = {
        "placa_controlada": placa,
        "clasificacion_difusa": notificacion.get("clasificacion_difusa"),
        "notificacion_correo": notificacion,
        "evaluacion_calidad_evento": evaluacion,
    }
    return resumen


def _extraer_placa_desde_resumen(resumen: dict) -> str:
    """Mejor lectura OCR disponible (consolidada o individual)."""
    candidatos = (
        resumen.get("placa_consolidada_evento"),
        resumen.get("placa_individual"),
        resumen.get("texto_ocr_corregido"),
        resumen.get("texto_ocr_crudo"),
    )
    for candidato in candidatos:
        if not candidato:
            continue
        texto = str(candidato).strip()
        for prefijo in ("Lectura parcial: ", "Formato dudoso: ", "Sin lectura: "):
            if texto.startswith(prefijo):
                texto = texto[len(prefijo) :].strip()
        normalizado = texto.upper().replace("-", "").replace(" ", "")
        if normalizado and normalizado not in {"PENDIENTE", "ANALIZANDOPLACA...", "SINLECTURA", "SINPLACA"}:
            return normalizado
    return ""


def _obtener_texto_placa_ui(resumen: dict) -> str:
    evento_id = resumen.get("evento_id")
    if evento_id is not None:
        entry = (st.session_state.get("ocr_eventos_cache") or {}).get(int(evento_id))
        if entry and entry.get("estado") == "listo":
            tmp = dict(resumen)
            tmp.update(entry.get("datos") or {})
            if entry.get("resumen_completo"):
                tmp.update(entry["resumen_completo"])
            placa_cache = _extraer_placa_desde_resumen(tmp)
            if placa_cache:
                return placa_cache
    placa = _extraer_placa_desde_resumen(resumen)
    if placa:
        return placa
    if resumen.get("capturando_vehiculo_activo"):
        evento_id = int(resumen.get("evento_id") or 0)
        return f"Capturando #{evento_id:03d}..." if evento_id else "Capturando..."
    if resumen.get("estado_ocr") == "pendiente":
        return "Leyendo..."
    texto_pendiente = str(resumen.get("texto_ocr_corregido") or resumen.get("texto_ocr_crudo") or "")
    if "Analizando placa" in texto_pendiente:
        return "Leyendo..."
    if resumen.get("evento_activo") and resumen.get("estado_placa") in ("Detectada", "Mantenida"):
        evento_id = int(resumen.get("evento_id") or 0)
        return f"Capturando #{evento_id:03d}..." if evento_id else "Capturando..."
    return "—"


def _obtener_resumen_panel_vivo(resumen_live: dict) -> dict:
    """Placa bloqueada por evento_id; el recorte en vivo se actualiza con mejores frames."""
    cache = st.session_state.get("ocr_eventos_cache") or {}
    evento_activo_id = int(resumen_live.get("evento_id") or 0) if resumen_live.get("evento_activo") else None

    if evento_activo_id and evento_activo_id in cache:
        entry = cache[evento_activo_id]
        if entry.get("estado") == "listo":
            return _resumen_panel_desde_cache(entry, resumen_live)
        out = _resumen_panel_desde_cache(entry, resumen_live)
        ruta_viva = resumen_live.get("ruta_recorte_evento_en_vivo")
        if ruta_viva and Path(str(ruta_viva)).exists():
            out["ruta_recorte_evento_en_vivo"] = str(ruta_viva)
            out["mejor_recorte_placa"] = str(ruta_viva)
            out["ruta_mejor_recorte_evento"] = str(ruta_viva)
        if entry.get("estado") == "pendiente":
            out["estado_ocr"] = "pendiente"
            out["texto_ocr_corregido"] = "Analizando placa..."
        return out

    for eid in sorted((int(k) for k in cache.keys()), reverse=True):
        if evento_activo_id and eid == evento_activo_id:
            continue
        entry = cache[eid]
        if entry.get("estado") != "listo":
            continue
        out = _resumen_panel_desde_cache(entry, resumen_live)
        if evento_activo_id:
            out["capturando_otro_vehiculo"] = evento_activo_id
        return out

    cola = st.session_state.get("eventos_monitoreo_cola") or []
    if cola:
        item = cola[0]
        res_cerrado = dict(item.get("resumen") or {})
        if item.get("estado_ocr") == "listo":
            if evento_activo_id and evento_activo_id != int(item.get("evento_id") or 0):
                res_cerrado["capturando_otro_vehiculo"] = evento_activo_id
            return res_cerrado
        if item.get("estado_ocr") == "pendiente":
            res_cerrado.setdefault("estado_ocr", "pendiente")
            res_cerrado.setdefault("texto_ocr_corregido", "Analizando placa...")
            if evento_activo_id and evento_activo_id != int(item.get("evento_id") or 0):
                res_cerrado["capturando_otro_vehiculo"] = evento_activo_id
            return res_cerrado

    if resumen_live.get("evento_activo"):
        out = dict(resumen_live)
        for campo in (
            "texto_ocr_crudo",
            "texto_ocr_corregido",
            "confianza_ocr",
            "formato_ocr_valido",
            "estado_ocr",
            "mensaje_ocr",
        ):
            out.pop(campo, None)
        ruta_snap = resumen_live.get("ruta_snapshot_ocr_evento") or resumen_live.get("ruta_recorte_evento_en_vivo")
        if ruta_snap:
            out["ruta_recorte_evento_en_vivo"] = ruta_snap
            out["mejor_recorte_placa"] = ruta_snap
        out["capturando_vehiculo_activo"] = True
        return out
    return resumen_live


def _resumen_panel_desde_cache(entry: dict, resumen_live: dict) -> dict:
    out = dict(entry.get("resumen_base") or resumen_live)
    ruta = entry.get("ruta_recorte")
    if ruta:
        out["ruta_recorte_evento_en_vivo"] = ruta
        out["mejor_recorte_placa"] = ruta
        out["ruta_mejor_recorte_evento"] = ruta
    if entry.get("estado") == "listo":
        out.update(entry.get("datos") or {})
        if entry.get("resumen_completo"):
            out.update(entry["resumen_completo"])
    return out


def _obtener_ruta_recorte_ui(resumen: dict) -> str | None:
    evento_activo = bool(resumen.get("evento_activo"))
    ocr_listo = resumen.get("estado_ocr") == "listo" or _evento_ocr_listo_en_cache(resumen.get("evento_id"))
    if evento_activo and not ocr_listo:
        claves = (
            "ruta_recorte_evento_en_vivo",
            "mejor_recorte_placa",
            "ruta_snapshot_ocr_evento",
            "ruta_mejor_recorte_evento",
            "ultimo_recorte_placa",
        )
    else:
        claves = (
            "ruta_snapshot_ocr_evento",
            "ruta_recorte_evento_en_vivo",
            "mejor_recorte_placa",
            "ruta_mejor_recorte_evento",
            "ultimo_recorte_placa",
        )
    for clave in claves:
        ruta = resumen.get(clave)
        if ruta and Path(str(ruta)).exists():
            return str(ruta)
    return None


def _cargar_recorte_ui_rgb(ruta: str):
    """Lee recorte desde disco; cache por mtime para no releer en cada frame de video."""
    path = Path(str(ruta))
    if not path.exists():
        return None
    mtime = path.stat().st_mtime_ns
    cache = st.session_state.setdefault("_cache_recorte_ui_rgb", {})
    prev = cache.get(str(path))
    if prev and prev.get("mtime") == mtime:
        return prev.get("rgb")
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cache[str(path)] = {"mtime": mtime, "rgb": rgb}
    st.session_state._cache_recorte_ui_rgb = cache
    return rgb


def _marcar_duplicados_placa_en_cola(cola: list) -> None:
    """Marca duplicados del mismo paso de vehiculo (ventana de frames)."""
    _consolidar_marcas_paso_vehiculo(cola, None)


def _puntaje_item_evento(item: dict) -> float:
    """Mayor puntaje = mejor candidato para enviar correo de ese paso."""
    res = item.get("resumen") or {}
    puntaje = 0.0
    if (res.get("evaluacion_calidad_evento") or {}).get("apto"):
        puntaje += 10.0
    if res.get("formato_consolidado_valido") or res.get("formato_ocr_valido"):
        puntaje += 5.0
    conf_ocr = res.get("confianza_final_evento") or res.get("confianza_ocr")
    if conf_ocr is not None:
        puntaje += float(conf_ocr) * 3.0
    conf_yolo = (res.get("ultima_deteccion") or {}).get("confianza") or res.get("mejor_confianza_evento")
    if conf_yolo is not None:
        puntaje += float(conf_yolo) * 1.0
    placa = _extraer_placa_desde_resumen(res)
    if placa and len(placa) == 7:
        puntaje += 1.5
    elif placa and len(placa) == 6:
        puntaje += 0.5
    if res.get("estado_ocr") == "error":
        puntaje -= 5.0
    return puntaje


def _agrupar_cola_por_paso_vehiculo(cola: list, ventana_frames: int) -> list[dict]:
    ordenados = sorted(
        [item for item in cola if item.get("frame_mejor") is not None or (item.get("resumen") or {}).get("frame_mejor_evento")],
        key=lambda item: int(item.get("frame_mejor") or (item.get("resumen") or {}).get("frame_mejor_evento") or 0),
    )
    grupos: list[dict] = []
    for item in ordenados:
        frame = int(item.get("frame_mejor") or (item.get("resumen") or {}).get("frame_mejor_evento") or 0)
        ubicado = False
        for grupo in grupos:
            if abs(frame - int(grupo["frame_fin"])) <= ventana_frames:
                grupo["items"].append(item)
                grupo["frame_fin"] = max(int(grupo["frame_fin"]), frame)
                grupo["frame_inicio"] = min(int(grupo["frame_inicio"]), frame)
                ubicado = True
                break
        if not ubicado:
            grupos.append({"frame_inicio": frame, "frame_fin": frame, "items": [item]})
    return grupos


def _grupo_listo_para_envio(grupo: dict, frame_actual: int, cooldown_frames: int) -> bool:
    items = grupo.get("items") or []
    if not items:
        return False
    if any(item.get("estado_ocr") == "pendiente" for item in items):
        return False
    ultimo_frame = max(
        int(item.get("frame_mejor") or (item.get("resumen") or {}).get("frame_mejor_evento") or 0)
        for item in items
    )
    return int(frame_actual or 0) - ultimo_frame >= int(cooldown_frames)


def _consolidar_marcas_paso_vehiculo(cola: list, config: dict | None) -> None:
    ventana = 150
    if config:
        ventana = int((config.get("notificaciones") or {}).get("ventana_frames_mismo_paso", ventana))
    for item in cola:
        item.pop("es_mejor_del_paso", None)
        item.pop("suprimido_duplicado", None)
    for grupo in _agrupar_cola_por_paso_vehiculo(cola, ventana):
        paso_id = f"paso_{grupo['frame_inicio']}"
        items_listos = [i for i in grupo["items"] if i.get("estado_ocr") == "listo"]
        if not items_listos:
            for item in grupo["items"]:
                item["paso_grupo_id"] = paso_id
            continue
        mejor = max(items_listos, key=_puntaje_item_evento)
        for item in grupo["items"]:
            item["paso_grupo_id"] = paso_id
            if item is mejor:
                item["es_mejor_del_paso"] = True
            else:
                item["suprimido_duplicado"] = True


def _estado_correo_item(item: dict) -> str:
    if item.get("notificacion_enviada"):
        return "Enviado"
    if item.get("suprimido_duplicado"):
        return "Duplicado"
    if item.get("estado_ocr") == "pendiente":
        return "OCR..."
    if item.get("estado_ocr") == "error":
        return "Error OCR"
    res = item.get("resumen") or {}
    if item.get("es_mejor_del_paso") and res.get("puede_enviar_notificacion"):
        return "Pendiente envio"
    if not res.get("puede_enviar_notificacion"):
        return "No apto"
    return "Duplicado"


def _procesar_envio_automatico_cola(
    config: dict,
    placa_controlada: str,
    frame_actual: int | None = None,
    *,
    forzar: bool = False,
) -> bool:
    """Agrupa pasos, elige la mejor lectura y envia un solo correo por vehiculo."""
    notif_cfg = (config.get("notificaciones") or {})
    if not notif_cfg.get("envio_automatico", True):
        return False
    correo = (st.session_state.get("correo_destino_monitoreo") or "").strip()
    if not validar_correo(correo):
        return False
    cola = st.session_state.get("eventos_monitoreo_cola") or []
    if not cola:
        return False

    ventana = int(notif_cfg.get("ventana_frames_mismo_paso", 150))
    cooldown = int(notif_cfg.get("cooldown_frames_envio", 60))
    frame_ref = int(frame_actual if frame_actual is not None else st.session_state.get("frame_actual") or 0)

    _consolidar_marcas_paso_vehiculo(cola, config)
    enviados: set[str] = set(st.session_state.get("pasos_grupo_notificados") or [])
    descartados: set[str] = set(st.session_state.get("pasos_grupo_descartados") or [])
    hubo_cambio = False

    for grupo in _agrupar_cola_por_paso_vehiculo(cola, ventana):
        paso_id = f"paso_{grupo['frame_inicio']}"
        if paso_id in enviados or paso_id in descartados:
            continue
        if not forzar and not _grupo_listo_para_envio(grupo, frame_ref, cooldown):
            continue
        if forzar and any(item.get("estado_ocr") == "pendiente" for item in grupo.get("items") or []):
            continue

        items_listos = [i for i in grupo["items"] if i.get("estado_ocr") == "listo"]
        if not items_listos:
            continue

        mejor = max(items_listos, key=_puntaje_item_evento)
        for item in grupo["items"]:
            if item is mejor:
                item["es_mejor_del_paso"] = True
                item.pop("suprimido_duplicado", None)
            else:
                item["es_mejor_del_paso"] = False
                item["suprimido_duplicado"] = True

        if mejor.get("notificacion_enviada"):
            enviados.add(paso_id)
            continue

        candidatos_apto = [
            i for i in items_listos if (i.get("resumen") or {}).get("puede_enviar_notificacion")
        ]
        if not candidatos_apto:
            descartados.add(paso_id)
            hubo_cambio = True
            continue

        mejor = max(candidatos_apto, key=_puntaje_item_evento)
        resumen = dict(mejor.get("resumen") or {})
        placa = _obtener_placa_para_evento(resumen, placa_controlada)
        resumen = enviar_notificacion_evento_monitoreo(resumen, placa, correo, config)
        mejor["resumen"] = resumen
        notif = resumen.get("notificacion_correo") or {}
        enviado = bool(notif.get("enviado") or notif.get("modo") == "simulado")
        mejor["notificacion_enviada"] = enviado
        if enviado:
            enviados.add(paso_id)
        hubo_cambio = True

    st.session_state.pasos_grupo_notificados = list(enviados)
    st.session_state.pasos_grupo_descartados = list(descartados)
    st.session_state.eventos_monitoreo_cola = cola
    return hubo_cambio


def _render_tabla_vehiculos_detectados(cola: list, config: dict | None = None, max_filas: int = 12) -> None:
    """Tabla con miniatura del recorte usado para cada prediccion."""
    if not cola:
        return
    _consolidar_marcas_paso_vehiculo(cola, config)
    st.markdown("**Vehiculos detectados**")
    st.caption(
        "Eventos del mismo paso se agrupan por proximidad de frames. "
        "Solo la mejor lectura apta se envia automaticamente al correo."
    )
    encabezado = st.columns([0.45, 1.15, 0.95, 0.75, 0.65, 0.55, 0.65])
    encabezado[0].markdown("**#**")
    encabezado[1].markdown("**Recorte**")
    encabezado[2].markdown("**Placa**")
    encabezado[3].markdown("**Hora**")
    encabezado[4].markdown("**Frame**")
    encabezado[5].markdown("**YOLO**")
    encabezado[6].markdown("**Correo**")
    for item in cola[:max_filas]:
        res_item = item.get("resumen") or {}
        ultima = res_item.get("ultima_deteccion") or {}
        conf_yolo = ultima.get("confianza") or res_item.get("mejor_confianza_evento")
        ruta = item.get("ruta_recorte") or _obtener_ruta_recorte_ui(res_item)
        placa_txt = _obtener_texto_placa_ui(res_item)
        if item.get("es_mejor_del_paso"):
            placa_txt = f"{placa_txt} ✓"
        elif item.get("suprimido_duplicado"):
            placa_txt = f"{placa_txt} · dup"
        fila = st.columns([0.45, 1.15, 0.95, 0.75, 0.65, 0.55, 0.65])
        fila[0].write(f"{int(item.get('evento_id') or 0)}")
        with fila[1]:
            if ruta and Path(str(ruta)).exists():
                st.image(str(ruta), width=112)
            else:
                st.caption("Sin recorte")
        fila[2].write(placa_txt)
        fila[3].write(item.get("hora", "—"))
        frame_ref = item.get("frame_mejor") or res_item.get("frame_mejor_evento") or res_item.get("frame_actual")
        fila[4].write(str(frame_ref) if frame_ref is not None else "—")
        fila[5].write(f"{float(conf_yolo):.0%}" if conf_yolo is not None else "—")
        fila[6].write(_estado_correo_item(item))
    enviados = sum(1 for item in cola if item.get("notificacion_enviada"))
    if enviados:
        st.caption(f"{enviados} correo(s) enviado(s) automaticamente en esta sesion.")


def _obtener_velocidad_kmh_desde_resumen(resumen: dict) -> float | None:
    if resumen.get("velocidad_kmh") is not None:
        return float(resumen["velocidad_kmh"])
    velocidad = resumen.get("velocidad") or {}
    if velocidad.get("velocidad_kmh") is not None:
        return float(velocidad["velocidad_kmh"])
    return None


def _obtener_limite_kmh_resumen(resumen: dict, config: dict | None) -> float:
    if resumen.get("limite_velocidad_kmh") is not None:
        return float(resumen["limite_velocidad_kmh"])
    if config:
        return float((config.get("speed") or {}).get("campus_speed_limit_kmh", 30.0))
    return 30.0


def _aplicar_multa_difusa_resumen(resumen: dict, config: dict | None) -> dict:
    kmh = _obtener_velocidad_kmh_desde_resumen(resumen)
    if kmh is None:
        return resumen
    salida = dict(resumen)
    limite = _obtener_limite_kmh_resumen(salida, config)
    salida["clasificacion_difusa"] = clasificar_velocidad(kmh, limite, config)
    salida["velocidad_kmh"] = kmh
    salida.setdefault("limite_velocidad_kmh", limite)
    return salida


def _formatear_metrica_velocidad(velocidad: dict, estado_placa: str = "", difuso: dict | None = None) -> tuple[str, str | None]:
    """Texto principal y detalle del tracker de velocidad para la UI."""
    velocidad_kmh = velocidad.get("velocidad_kmh")
    if velocidad_kmh is not None:
        detalle = "Tic-toc L1→L2 (cruce geometrico + sub-frame)"
        if difuso and difuso.get("multa_texto"):
            detalle = f"{difuso.get('multa_texto')} · {difuso.get('estado', '')}"
        return f"{float(velocidad_kmh):.1f} km/h", detalle

    estado = str(velocidad.get("estado") or "esperando_linea_1")
    if estado == "medicion_invalida":
        motivo = velocidad.get("motivo_invalido") or "Medicion rechazada"
        return "Medicion invalida", motivo

    etiquetas = {
        "esperando_linea_1": "Esperando Linea 1",
        "esperando_linea_2": "Esperando Linea 2",
        "velocidad_calculada": "Calculando...",
    }
    principal = etiquetas.get(estado, "Sin medicion")

    frame_l1 = velocidad.get("frame_cruce_linea_1")
    frame_l2 = velocidad.get("frame_cruce_linea_2")
    frame_l1_x = velocidad.get("frame_cruce_linea_1_exacto")
    if estado == "esperando_linea_1":
        if estado_placa in {"Detectada", "Mantenida"}:
            detalle = "Placa detectada · aun no cruza Linea 1 (cian)"
        else:
            detalle = "Esperando vehiculo sobre Linea 1"
    elif estado == "esperando_linea_2":
        tic = f"{frame_l1_x:.2f}" if frame_l1_x is not None else str(frame_l1)
        detalle = f"TIC en frame {tic} · esperando TOC en Linea 2 (azul)"
    elif frame_l1 is not None and frame_l2 is not None:
        detalle = f"Cruces: L1 frame {frame_l1} · L2 frame {frame_l2}"
    else:
        detalle = "Monitoreo tic-toc entre lineas activo"

    return principal, detalle


def _detalle_lineas_velocidad(velocidad: dict, resumen: dict) -> str:
    """Linea de contexto con distancia, limites y posicion del centro de la placa."""
    partes = []
    distancia = velocidad.get("distancia_metros") or resumen.get("distancia_lineas_m")
    if distancia is not None:
        partes.append(f"Distancia calibrada: {float(distancia):.1f} m")
    limite = resumen.get("limite_velocidad_kmh")
    if limite is not None:
        partes.append(f"Limite: {float(limite):.0f} km/h")
    fps = velocidad.get("fps") or resumen.get("fps")
    if fps:
        partes.append(f"FPS video: {float(fps):.2f}")
    l1 = resumen.get("posicion_linea_1")
    l2 = resumen.get("posicion_linea_2")
    if l1 is not None and l2 is not None:
        partes.append(f"Lineas UI: {float(l1):.0%} / {float(l2):.0%}")
    centro_y = velocidad.get("centro_y")
    if centro_y is not None:
        partes.append(f"Centro placa Y: {int(centro_y)} px")
    t1 = velocidad.get("tiempo_cruce_linea_1")
    t2 = velocidad.get("tiempo_cruce_linea_2")
    tiempo = velocidad.get("tiempo_entre_lineas")
    if t1 is not None and t2 is not None:
        partes.append(f"TIC {float(t1):.4f} s · TOC {float(t2):.4f} s")
    if tiempo is not None:
        partes.append(f"Delta t: {float(tiempo):.4f} s")
    return " · ".join(partes)


def _mostrar_formula_velocidad(velocidad: dict) -> None:
    formula = velocidad.get("formula_medicion")
    if formula:
        st.caption(f"Formula: {formula}")
    elif velocidad.get("estado") == "medicion_invalida" and velocidad.get("motivo_invalido"):
        st.warning(velocidad["motivo_invalido"])


def _render_panel_deteccion_esencial(
    resumen: dict,
    config: dict | None = None,
    placa_controlada: str = "",
    mostrar_historial: bool = False,
) -> dict:
    """Vista principal del operador: placa, velocidad, estado y recorte."""
    placa = _obtener_texto_placa_ui(resumen)
    ruta_recorte = _obtener_ruta_recorte_ui(resumen)
    velocidad = resumen.get("velocidad") or {}
    velocidad_kmh = velocidad.get("velocidad_kmh")
    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    conf_yolo = ultima_deteccion.get("confianza") or resumen.get("mejor_confianza_evento")
    conf_ocr = resumen.get("confianza_ocr")
    estado_placa = resumen.get("estado_placa") or "Esperando"
    difuso = resumen.get("clasificacion_difusa") or {}
    if velocidad_kmh is not None and not difuso:
        difuso = clasificar_velocidad(
            float(velocidad_kmh),
            _obtener_limite_kmh_resumen(resumen, config),
            config,
        )
        resumen["clasificacion_difusa"] = difuso

    texto_velocidad, detalle_velocidad = _formatear_metrica_velocidad(velocidad, estado_placa, difuso)
    texto_sancion = (
        difuso.get("multa_texto")
        or difuso.get("estado")
        or ("—" if velocidad_kmh is None else "Pendiente")
    )

    col_img, col_main = st.columns([0.3, 0.7], gap="medium")
    with col_img:
        if ruta_recorte:
            imagen_recorte = _cargar_recorte_ui_rgb(ruta_recorte)
            if imagen_recorte is not None:
                st.image(imagen_recorte, channels="RGB", use_container_width=True)
            else:
                st.image(ruta_recorte, use_container_width=True)
        else:
            st.info("Esperando placa...")
    with col_main:
        st.markdown(f"## {placa}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Velocidad", texto_velocidad, delta=detalle_velocidad)
        m2.metric("Deteccion", estado_placa)
        m3.metric("Multa / sancion", texto_sancion)

        detalle_lineas = _detalle_lineas_velocidad(velocidad, resumen)
        if detalle_lineas:
            st.caption(detalle_lineas)
        _mostrar_formula_velocidad(velocidad)
        if difuso.get("metodo"):
            st.caption(f"Logica difusa: {difuso['metodo']} · categoria {difuso.get('categoria_fuzzy', '—')}")

        partes_conf = []
        if conf_yolo is not None:
            partes_conf.append(f"YOLO {float(conf_yolo):.0%}")
        if conf_ocr is not None:
            partes_conf.append(f"CNN {float(conf_ocr):.0%}")
        if partes_conf:
            st.caption(" · ".join(partes_conf))

    if config is not None:
        resumen = preparar_estado_notificacion_evento(
            resumen,
            _obtener_placa_para_evento(resumen, placa_controlada),
            config,
        )
        if resumen.get("puede_enviar_notificacion"):
            st.success("Captura apta. El correo se enviara automaticamente al cerrar el paso del vehiculo.")
        elif ruta_recorte:
            razones = (resumen.get("evaluacion_calidad_evento") or {}).get("razones") or []
            if razones:
                st.caption(f"Correo: {razones[0]}")

    if mostrar_historial:
        cola = st.session_state.get("eventos_monitoreo_cola") or []
        if cola:
            _render_tabla_vehiculos_detectados(cola, config=config)
        historial = st.session_state.get("historial_detecciones_monitoreo", [])
        if historial and not cola:
            st.markdown("**Ultimas detecciones**")
            filas = [
                {
                    "Hora": item.get("hora"),
                    "Placa": item.get("placa"),
                    "Velocidad": item.get("velocidad"),
                    "Confianza": item.get("confianza"),
                }
                for item in historial[:5]
            ]
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    return resumen


def _render_debug_monitoreo(resumen: dict, config: dict) -> None:
    """Informacion tecnica opcional (desarrollo / ajuste fino)."""
    velocidad = resumen.get("velocidad") or {}
    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    confianza_yolo = ultima_deteccion.get("confianza") or resumen.get("mejor_confianza_evento")
    confianza_ocr = resumen.get("confianza_ocr")
    velocidad_kmh = velocidad.get("velocidad_kmh")
    mejor_info = resumen.get("mejor_recorte_placa_info") or {}
    ruta_recorte = _obtener_ruta_recorte_ui(resumen)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Texto crudo OCR", resumen.get("texto_ocr_crudo", "—"))
    c2.metric("Conf. YOLO", f"{float(confianza_yolo):.2f}" if confianza_yolo is not None else "—")
    c3.metric("Conf. CNN", f"{float(confianza_ocr):.2f}" if confianza_ocr is not None else "—")
    c4.metric("Velocidad", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "—")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Puntaje recorte", f"{mejor_info.get('puntaje_total'):.3f}" if mejor_info.get("puntaje_total") is not None else "—")
    c6.metric("Nitidez", f"{mejor_info.get('nitidez'):.1f}" if mejor_info.get("nitidez") is not None else "—")
    c7.metric("Segmentacion", resumen.get("estrategia_segmentacion") or "—")
    c8.metric("Frame", resumen.get("frame_actual") or resumen.get("frames_procesados") or "—")

    candidatos = resumen.get("ultimos_candidatos_recorte") or []
    if mejor_info or candidatos:
        st.caption("Seleccion de mejor recorte")
        if mejor_info:
            st.json({k: mejor_info.get(k) for k in ("frame_index", "puntaje_total", "conf_yolo", "nitidez", "ruta_recorte") if mejor_info.get(k) is not None})
        if candidatos:
            st.dataframe(pd.DataFrame(candidatos), use_container_width=True, hide_index=True)

    tiempos = resumen.get("tiempos_etapa") or {}
    if tiempos:
        st.caption("Tiempos por etapa (ms)")
        st.json(tiempos)

    st.json(
        {
            "modelo_yolo": config["models"].get("plate_detector_model", config["models"].get("plate_detector_path")),
            "modelo_cnn": config["models"].get("character_reader_path"),
            "fps_procesamiento": resumen.get("fps_procesamiento"),
            "tiempo_yolo_ms": resumen.get("tiempo_yolo_ms"),
            "tiempo_lector_cnn_ms": resumen.get("tiempo_lector_cnn_ms"),
            "bbox": ultima_deteccion.get("bbox"),
            "ruta_recorte": ruta_recorte,
            "estado_ocr": resumen.get("estado_ocr"),
            "motivo_ocr": resumen.get("motivo_lector_cnn") or resumen.get("causa_probable_ocr"),
            "votacion": resumen.get("votos_por_posicion_evento"),
        }
    )


def _render_sidebar_envio_correo(config: dict, placa_controlada: str) -> None:
    """Estado del envio automatico de correos."""
    st.divider()
    st.markdown("**Notificacion por correo**")
    notif_cfg = (config.get("notificaciones") or {})
    correo = (st.session_state.get("correo_destino_monitoreo") or "").strip()
    cola = st.session_state.get("eventos_monitoreo_cola") or []

    if notif_cfg.get("envio_automatico", True):
        st.caption("Envio automatico activo: un correo por paso de vehiculo (mejor lectura).")
    else:
        st.caption("Envio automatico desactivado en config.")

    if not validar_correo(correo):
        st.warning("Ingrese un correo destino valido arriba.")
        return

    enviados = [item for item in cola if item.get("notificacion_enviada")]
    pendientes = [
        item
        for item in cola
        if item.get("es_mejor_del_paso") and not item.get("notificacion_enviada") and not item.get("suprimido_duplicado")
    ]

    st.metric("Correos enviados", len(enviados))
    if enviados:
        ultimo = enviados[0]
        res_u = ultimo.get("resumen") or {}
        notif = res_u.get("notificacion_correo") or {}
        placa_u = _obtener_texto_placa_ui(res_u)
        st.caption(f"Ultimo: **{placa_u}** → {notif.get('destinatario') or correo}")
    elif pendientes:
        st.caption("Esperando cierre del paso del vehiculo para enviar la mejor lectura.")
    else:
        st.caption("Sin envios en esta sesion.")

    if enviados:
        with st.expander("Historial de envios", expanded=False):
            for item in enviados[:6]:
                res_item = item.get("resumen") or {}
                st.caption(
                    f"#{int(item.get('evento_id') or 0):03d} · {_obtener_texto_placa_ui(res_item)} · {item.get('hora', '—')}"
                )


def _cache_key_ocr_monitoreo(resumen: dict, ruta_recorte: str) -> str:
    evento_id = resumen.get("evento_id") or resumen.get("eventos_placa") or 0
    frame_ref = resumen.get("frame_mejor_evento") or resumen.get("frame_actual") or resumen.get("frames_procesados")
    return f"evt{int(evento_id)}|{ruta_recorte}|{frame_ref}"


_CAMPOS_OCR_RESUMEN = (
    "texto_ocr_crudo",
    "texto_ocr_corregido",
    "confianza_ocr",
    "formato_ocr_valido",
    "estado_ocr",
    "mensaje_ocr",
    "causa_probable_ocr",
    "motivo_lector_cnn",
    "placa_individual",
    "placa_consolidada_evento",
    "estado_consolidado_evento",
    "confianza_final_evento",
    "formato_consolidado_valido",
    "votos_por_posicion_evento",
    "correcciones_evento",
    "lecturas_usadas_evento",
    "lecturas_descartadas_evento",
    "lecturas_descartadas_detalle",
    "lectura_base_usada",
    "recortes_usados_evento",
    "ruta_debug_votacion_evento",
    "lector_cnn_ok",
    "etapa_error_lector_cnn",
    "tiempo_lector_cnn_ms",
)


def _limpiar_campos_ocr_resumen(resumen: dict, *, pendiente: bool = True) -> dict:
    limpio = dict(resumen)
    for campo in _CAMPOS_OCR_RESUMEN:
        limpio.pop(campo, None)
    if pendiente:
        limpio["texto_ocr_crudo"] = "Analizando placa..."
        limpio["texto_ocr_corregido"] = "Analizando placa..."
        limpio["confianza_ocr"] = None
        limpio["formato_ocr_valido"] = False
        limpio["estado_ocr"] = "pendiente"
    return limpio


def _reiniciar_ocr_vivo_si_cambio_evento(resumen: dict) -> dict:
    evento_id = resumen.get("evento_id")
    if evento_id is None:
        return resumen
    evento_id = int(evento_id)
    prev = st.session_state.get("ocr_evento_vivo_id")
    if prev is not None and evento_id != int(prev):
        st.session_state.ocr_evento_vivo_id = evento_id
        st.session_state.lecturas_evento_id = evento_id
        st.session_state.lecturas_evento_placa_monitoreo = []
    elif prev is None:
        st.session_state.ocr_evento_vivo_id = evento_id
        st.session_state.lecturas_evento_id = evento_id
    limpio = dict(resumen)
    for campo in _CAMPOS_OCR_RESUMEN:
        limpio.pop(campo, None)
    return limpio


def _cache_key_ocr_evento_id(evento_id: int) -> str:
    return f"evento_{int(evento_id)}"


def _cache_key_ocr_evento_cerrado(evento_id: int, ruta_recorte: str = "") -> str:
    return _cache_key_ocr_evento_id(evento_id)


def _evento_ocr_listo_en_cache(evento_id: int | None) -> bool:
    if evento_id is None:
        return False
    entry = (st.session_state.get("ocr_eventos_cache") or {}).get(int(evento_id))
    return bool(entry and entry.get("estado") == "listo")


def _hay_ocr_pendiente_en_cache() -> bool:
    return any(
        entry.get("estado") == "pendiente"
        for entry in (st.session_state.get("ocr_eventos_cache") or {}).values()
    )


def _fusionar_resumen_live_monitoreo(estado_frame: dict, fuente: str, placa_controlada: str) -> dict:
    prev = st.session_state.get("ultimo_resultado_parcial") or {}
    if estado_frame.get("_solo_video"):
        resumen_live = {
            clave: prev[clave]
            for clave in (
                "evento_id",
                "evento_activo",
                "mejor_confianza_evento",
                "frame_mejor_evento",
                "ruta_recorte_evento_en_vivo",
                "ruta_snapshot_ocr_evento",
                "ruta_mejor_recorte_evento",
                "velocidad",
                "distancia_lineas_m",
                "limite_velocidad_kmh",
                "modo_rendimiento",
            )
            if clave in prev
        }
    else:
        resumen_live = dict(prev)
        for campo in (
            "texto_ocr_crudo",
            "texto_ocr_corregido",
            "confianza_ocr",
            "formato_ocr_valido",
            "estado_ocr",
            "mensaje_ocr",
            "placa_consolidada_evento",
            "placa_individual",
        ):
            resumen_live.pop(campo, None)
    resumen_live.update(estado_frame)
    resumen_live.setdefault("frames_procesados", estado_frame.get("frame_actual", 0))
    resumen_live.setdefault("fuente", fuente)
    resumen_live.setdefault("placa_controlada", placa_controlada)
    return _reiniciar_ocr_vivo_si_cambio_evento(resumen_live)


def _persistir_ocr_evento_en_cache(
    evento_id: int,
    resumen_base: dict,
    ruta_recorte: str,
    config: dict,
    placa_controlada: str,
    resultado_ocr: dict | None,
    tiempo_ms: float,
    *,
    error: str | None = None,
) -> None:
    eid = int(evento_id)
    cache_key = _cache_key_ocr_evento_id(eid)
    cache = st.session_state.setdefault("ocr_eventos_cache", {})
    resumen = dict(resumen_base)
    resumen["evento_id"] = eid
    if error:
        entry = {
            "estado": "error",
            "ruta_recorte": str(ruta_recorte),
            "resumen_base": dict(resumen_base),
            "datos": {
                "texto_ocr_crudo": "Sin lectura",
                "texto_ocr_corregido": "Sin lectura",
                "confianza_ocr": None,
                "formato_ocr_valido": False,
                "estado_ocr": "error",
                "mensaje_ocr": error,
                "lector_cnn_ok": False,
            },
        }
    else:
        resumen = _aplicar_resultado_ocr_monitoreo(
            resumen,
            resultado_ocr or {},
            float(tiempo_ms),
            str(ruta_recorte),
            cache_key,
            registrar_metricas=False,
        )
        placa = _obtener_placa_para_evento(resumen, placa_controlada)
        resumen_completo = preparar_estado_notificacion_evento(resumen, placa, config)
        entry = {
            "estado": "listo",
            "ruta_recorte": str(ruta_recorte),
            "resumen_base": dict(resumen_base),
            "datos": {campo: resumen.get(campo) for campo in _CAMPOS_OCR_RESUMEN if campo in resumen},
            "resumen_completo": resumen_completo,
        }
    cache[eid] = entry
    st.session_state.ocr_eventos_cache = cache
    _sync_cola_item_desde_cache(eid, entry, config, placa_controlada)


def _solicitar_ocr_desde_estado_frame(
    estado_frame: dict,
    fuente: str,
    placa_controlada: str,
    config: dict,
) -> None:
    """Encola OCR en segundo plano; no bloquea el video."""
    ocr_pendiente = estado_frame.get("ocr_evento_pendiente")
    if ocr_pendiente and ocr_pendiente.get("evento_id") and ocr_pendiente.get("ruta_recorte"):
        eid = int(ocr_pendiente["evento_id"])
        if _evento_ocr_listo_en_cache(eid):
            return
        base_ocr = _fusionar_resumen_live_monitoreo(estado_frame, fuente, placa_controlada)
        base_ocr["evento_id"] = eid
        base_ocr["ruta_snapshot_ocr_evento"] = ocr_pendiente["ruta_recorte"]
        _encolar_ocr_evento_vivo_asincrono(eid, str(ocr_pendiente["ruta_recorte"]), base_ocr, config, placa_controlada)
        return

    evento_id = estado_frame.get("evento_id")
    ruta = estado_frame.get("ruta_snapshot_ocr_evento") or estado_frame.get("ruta_recorte_evento_en_vivo")
    if not evento_id or not ruta:
        return
    eid = int(evento_id)
    if _evento_ocr_listo_en_cache(eid):
        return
    if not Path(str(ruta)).exists():
        return
    if int(estado_frame.get("detecciones_validas") or 0) <= 0 and not estado_frame.get("evento_activo"):
        return
    base_ocr = _fusionar_resumen_live_monitoreo(estado_frame, fuente, placa_controlada)
    _encolar_ocr_evento_vivo_asincrono(eid, str(ruta), base_ocr, config, placa_controlada)


def _poll_ocr_monitoreo_instantaneo(
    config: dict,
    placa_controlada: str,
    resumen_live: dict | None,
    *,
    frame_actual: int | None = None,
    forzar_cola: bool = False,
) -> tuple[dict, bool]:
    """Consulta resultados OCR listos sin bloquear el video."""
    resumen = dict(resumen_live or {})
    listos_antes = set(st.session_state.get("_ocr_eventos_listos") or [])
    hubo = _actualizar_cache_ocr_eventos(config, placa_controlada)
    cache = st.session_state.get("ocr_eventos_cache") or {}
    listos_ahora = {int(eid) for eid, entry in cache.items() if entry.get("estado") == "listo"}
    if listos_ahora - listos_antes:
        hubo = True
    st.session_state._ocr_eventos_listos = list(listos_ahora)
    if forzar_cola or _hay_ocr_pendiente_en_cache():
        _, hubo_cola = _procesar_cola_ocr_monitoreo(
            config,
            placa_controlada,
            resumen,
            forzar=True,
            frame_actual=frame_actual,
        )
        hubo = hubo or hubo_cola
    return resumen, hubo


def _encolar_ocr_evento_vivo_asincrono(
    evento_id: int,
    ruta_recorte: str,
    resumen_base: dict,
    config: dict,
    placa_controlada: str,
) -> None:
    eid = int(evento_id)
    cache = st.session_state.setdefault("ocr_eventos_cache", {})
    if cache.get(eid, {}).get("estado") == "listo":
        return
    cache_key = _cache_key_ocr_evento_id(eid)
    ruta_nueva = str(ruta_recorte)
    with _OCR_ASYNC_LOCK:
        en_vuelo = cache_key in _OCR_ASYNC_PENDIENTES
    if en_vuelo:
        entry = cache.get(eid) or {}
        entry["ruta_recorte"] = ruta_nueva
        entry["resumen_base"] = dict(resumen_base)
        entry["_ocr_snapshot_mas_reciente"] = ruta_nueva
        entry.setdefault("estado", "pendiente")
        cache[eid] = entry
        st.session_state.ocr_eventos_cache = cache
        return
    with _OCR_ASYNC_LOCK:
        if cache_key in _OCR_ASYNC_PENDIENTES:
            return
    cache[eid] = {
        "estado": "pendiente",
        "ruta_recorte": ruta_nueva,
        "resumen_base": dict(resumen_base),
    }
    st.session_state.ocr_eventos_cache = cache
    with _OCR_ASYNC_LOCK:
        if cache_key in _OCR_ASYNC_PENDIENTES:
            return
        _OCR_ASYNC_PENDIENTES.add(cache_key)
    contexto = {
        "funcion": "_encolar_ocr_evento_vivo_asincrono",
        "evento_id": eid,
        "frame_actual": resumen_base.get("frame_actual"),
        "fuente": resumen_base.get("fuente"),
    }
    threading.Thread(
        target=_tarea_ocr_monitoreo_asincrona,
        args=(cache_key, ruta_nueva, contexto, dict(resumen_base)),
        daemon=True,
    ).start()


def _solicitar_ocr_evento_tiempo_real(
    evento_id: int,
    ruta_recorte: str,
    resumen_base: dict,
    config: dict | None = None,
    placa_controlada: str = "",
) -> None:
    if config is None:
        config = {}
    _encolar_ocr_evento_vivo_asincrono(int(evento_id), str(ruta_recorte), resumen_base, config, placa_controlada)


def _sync_cola_item_desde_cache(evento_id: int, entry: dict, config: dict, placa_controlada: str) -> None:
    cola = st.session_state.get("eventos_monitoreo_cola") or []
    for item in cola:
        if int(item.get("evento_id") or 0) != int(evento_id):
            continue
        if entry.get("estado") == "listo" and entry.get("resumen_completo"):
            item["estado_ocr"] = "listo"
            item["resumen"] = dict(entry["resumen_completo"])
            _actualizar_historial_monitoreo(item["resumen"])
        elif entry.get("estado") == "error":
            item["estado_ocr"] = "error"
            item["resumen"].update(entry.get("datos") or {})
        break
    st.session_state.eventos_monitoreo_cola = cola


def _actualizar_cache_ocr_eventos(config: dict, placa_controlada: str) -> bool:
    cache = st.session_state.setdefault("ocr_eventos_cache", {})
    if not cache:
        return False
    hubo_cambio = False
    for eid, entry in list(cache.items()):
        if entry.get("estado") != "pendiente":
            continue
        cache_key = _cache_key_ocr_evento_id(int(eid))
        with _OCR_ASYNC_LOCK:
            payload = _OCR_ASYNC_RESULTADOS.pop(cache_key, None)
        if not payload:
            continue
        hubo_cambio = True
        resumen = dict(entry.get("resumen_base") or {})
        resumen["evento_id"] = int(eid)
        ruta_recorte = str(entry.get("ruta_recorte") or payload.get("ruta_recorte") or "")
        if payload.get("error"):
            entry["estado"] = "error"
            entry["datos"] = {
                "texto_ocr_crudo": "Sin lectura",
                "texto_ocr_corregido": "Sin lectura",
                "confianza_ocr": None,
                "formato_ocr_valido": False,
                "estado_ocr": "error",
                "mensaje_ocr": payload["error"],
                "lector_cnn_ok": False,
            }
        else:
            ruta_procesada = ruta_recorte
            snapshot_mas_reciente = entry.pop("_ocr_snapshot_mas_reciente", None)
            if (
                snapshot_mas_reciente
                and str(snapshot_mas_reciente) != str(ruta_procesada)
                and Path(str(snapshot_mas_reciente)).exists()
            ):
                entry["estado"] = "pendiente"
                entry["ruta_recorte"] = str(snapshot_mas_reciente)
                base_reocr = dict(entry.get("resumen_base") or resumen)
                base_reocr["evento_id"] = int(eid)
                cache[int(eid)] = entry
                st.session_state.ocr_eventos_cache = cache
                _encolar_ocr_evento_vivo_asincrono(
                    int(eid),
                    str(snapshot_mas_reciente),
                    base_reocr,
                    config,
                    placa_controlada,
                )
                continue
            resumen = _aplicar_resultado_ocr_monitoreo(
                resumen,
                payload["resultado_ocr"],
                float(payload.get("tiempo_lector_ms") or 0.0),
                ruta_recorte,
                cache_key,
                registrar_metricas=False,
            )
            entry["estado"] = "listo"
            entry["datos"] = {campo: resumen.get(campo) for campo in _CAMPOS_OCR_RESUMEN if campo in resumen}
            placa = _obtener_placa_para_evento(resumen, placa_controlada)
            entry["resumen_completo"] = preparar_estado_notificacion_evento(resumen, placa, config)
        _sync_cola_item_desde_cache(int(eid), entry, config, placa_controlada)
    if hubo_cambio:
        st.session_state.ocr_eventos_cache = cache
    return hubo_cambio


def _construir_resumen_evento_cerrado(evento: dict, fuente: str, config: dict | None = None) -> dict:
    resumen = {
        "evento_id": evento.get("evento_id"),
        "eventos_placa": evento.get("evento_id"),
        "frame_mejor_evento": evento.get("frame_mejor"),
        "frame_actual": evento.get("frame_fin"),
        "frames_procesados": evento.get("frame_fin"),
        "mejor_confianza_evento": evento.get("mejor_confianza"),
        "ruta_mejor_recorte_evento": evento.get("ruta_mejor_recorte"),
        "ruta_mejor_frame_evento": evento.get("ruta_mejor_frame"),
        "ruta_frame_evento_en_vivo": evento.get("ruta_frame_evento_en_vivo"),
        "mejor_recorte_placa": evento.get("ruta_mejor_recorte"),
        "ultima_deteccion": {
            "confianza": evento.get("mejor_confianza"),
            "bbox": evento.get("mejor_bbox"),
        },
        "velocidad": evento.get("velocidad") or {},
        "limite_velocidad_kmh": evento.get("limite_velocidad_kmh"),
        "fuente": fuente,
        "estado_placa": "Cerrado",
        "evento_activo": False,
    }
    return _aplicar_multa_difusa_resumen(resumen, config)


def _registrar_evento_cerrado_monitoreo(
    evento: dict,
    config: dict,
    placa_controlada: str,
    fuente: str,
) -> None:
    if not evento or not evento.get("evento_id"):
        return
    evento_id = int(evento["evento_id"])
    cola = st.session_state.setdefault("eventos_monitoreo_cola", [])
    if any(item.get("evento_id") == evento_id for item in cola):
        return

    resumen = _construir_resumen_evento_cerrado(evento, fuente, config)
    ruta_recorte = evento.get("ruta_mejor_recorte")
    cache = st.session_state.get("ocr_eventos_cache") or {}
    cache_entry = cache.get(evento_id)
    estado_ocr_inicial = "pendiente"
    if cache_entry and cache_entry.get("estado") == "listo" and cache_entry.get("resumen_completo"):
        resumen = dict(cache_entry["resumen_completo"])
        estado_ocr_inicial = "listo"
    elif cache_entry and cache_entry.get("estado") == "error":
        resumen.update(cache_entry.get("datos") or {})
        estado_ocr_inicial = "error"

    item = {
        "evento_id": evento_id,
        "hora": datetime.now().strftime("%H:%M:%S"),
        "resumen": resumen,
        "ruta_recorte": str(ruta_recorte) if ruta_recorte else None,
        "frame_mejor": evento.get("frame_mejor"),
        "estado_ocr": estado_ocr_inicial,
        "notificacion_enviada": False,
    }
    cola.insert(0, item)
    max_eventos = int(st.session_state.get("historial_maximo_monitoreo", 15) or 15)
    st.session_state.eventos_monitoreo_cola = cola[:max_eventos]
    st.session_state.ultimo_evento_cerrado_registrado = max(
        int(st.session_state.get("ultimo_evento_cerrado_registrado") or 0),
        evento_id,
    )

    ruta_recorte = evento.get("ruta_mejor_recorte")
    if estado_ocr_inicial == "pendiente" and ruta_recorte and Path(str(ruta_recorte)).exists():
        if not _evento_ocr_listo_en_cache(evento_id):
            _encolar_ocr_evento_cerrado(evento_id, resumen, str(ruta_recorte))

    _actualizar_historial_monitoreo(resumen)


def _encolar_ocr_evento_cerrado(evento_id: int, resumen: dict, ruta_recorte: str) -> None:
    cache = st.session_state.get("ocr_eventos_cache") or {}
    if cache.get(int(evento_id), {}).get("estado") == "listo":
        return
    cache_key = _cache_key_ocr_evento_id(evento_id)
    with _OCR_ASYNC_LOCK:
        if cache_key in _OCR_ASYNC_PENDIENTES or cache_key in _OCR_ASYNC_RESULTADOS:
            return
        _OCR_ASYNC_PENDIENTES.add(cache_key)
    if int(evento_id) not in (st.session_state.get("ocr_eventos_cache") or {}):
        st.session_state.setdefault("ocr_eventos_cache", {})[int(evento_id)] = {
            "estado": "pendiente",
            "ruta_recorte": str(ruta_recorte),
            "resumen_base": dict(resumen),
        }
    contexto = {
        "funcion": "_encolar_ocr_evento_cerrado",
        "evento_id": evento_id,
        "frame_actual": resumen.get("frame_actual"),
        "fuente": resumen.get("fuente"),
    }
    threading.Thread(
        target=_tarea_ocr_monitoreo_asincrona,
        args=(cache_key, ruta_recorte, contexto, dict(resumen)),
        daemon=True,
    ).start()


def _procesar_ocr_cola_eventos(config: dict, placa_controlada: str) -> bool:
    """Fusiona OCR listo para cada vehiculo cerrado. Retorna True si hubo cambios."""
    cola = st.session_state.get("eventos_monitoreo_cola") or []
    if not cola:
        return False
    hubo_cambio = False
    for item in cola:
        if item.get("estado_ocr") != "pendiente":
            continue
        evento_id = int(item.get("evento_id") or 0)
        resumen = dict(item.get("resumen") or {})
        ruta_recorte = resumen.get("ruta_mejor_recorte_evento")
        if not ruta_recorte:
            continue
        cache_key = _cache_key_ocr_evento_id(evento_id)
        cache = st.session_state.get("ocr_eventos_cache") or {}
        cache_entry = cache.get(evento_id)
        if cache_entry and cache_entry.get("estado") == "listo" and cache_entry.get("resumen_completo"):
            item["estado_ocr"] = "listo"
            item["resumen"] = dict(cache_entry["resumen_completo"])
            placa = _obtener_placa_para_evento(item["resumen"], placa_controlada)
            item["resumen"] = preparar_estado_notificacion_evento(item["resumen"], placa, config)
            _actualizar_historial_monitoreo(item["resumen"])
            hubo_cambio = True
            continue
        with _OCR_ASYNC_LOCK:
            payload = _OCR_ASYNC_RESULTADOS.pop(cache_key, None)
        if not payload:
            continue
        if payload.get("error"):
            resumen.update(
                {
                    "texto_ocr_crudo": "Sin lectura",
                    "texto_ocr_corregido": "Sin lectura",
                    "confianza_ocr": None,
                    "formato_ocr_valido": False,
                    "estado_ocr": "error",
                    "mensaje_ocr": payload["error"],
                    "lector_cnn_ok": False,
                }
            )
            item["estado_ocr"] = "error"
        else:
            resumen = _aplicar_resultado_ocr_monitoreo(
                resumen,
                payload["resultado_ocr"],
                float(payload.get("tiempo_lector_ms") or 0.0),
                str(payload.get("ruta_recorte") or ruta_recorte),
                cache_key,
                registrar_metricas=False,
            )
            item["estado_ocr"] = "listo"
        placa = _obtener_placa_para_evento(resumen, placa_controlada)
        item["resumen"] = preparar_estado_notificacion_evento(resumen, placa, config)
        cache = st.session_state.setdefault("ocr_eventos_cache", {})
        cache[evento_id] = {
            "estado": item["estado_ocr"],
            "ruta_recorte": str(ruta_recorte),
            "resumen_base": dict(item["resumen"]),
            "datos": {campo: resumen.get(campo) for campo in _CAMPOS_OCR_RESUMEN if campo in resumen},
            "resumen_completo": dict(item["resumen"]) if item["estado_ocr"] == "listo" else None,
        }
        st.session_state.ocr_eventos_cache = cache
        _actualizar_historial_monitoreo(item["resumen"])
        hubo_cambio = True
    if hubo_cambio:
        st.session_state.eventos_monitoreo_cola = cola
        _procesar_envio_automatico_cola(config, placa_controlada)
    return hubo_cambio


def _procesar_cola_ocr_monitoreo(
    config: dict,
    placa_controlada: str,
    resumen_live: dict | None = None,
    *,
    forzar: bool = False,
    frame_actual: int | None = None,
) -> tuple[dict, bool]:
    """Procesa OCR de vehiculos cerrados con throttle corto; no reinicia OCR en vivo."""
    resumen = dict(resumen_live or {})
    ahora = time.perf_counter()
    ultimo = float(st.session_state.get("ultimo_cola_ocr_ts") or 0.0)
    cache = st.session_state.get("ocr_eventos_cache") or {}
    tiene_pendiente = any(entry.get("estado") == "pendiente" for entry in cache.values())
    intervalo = 0.12 if (tiene_pendiente or forzar) else 0.25
    if not forzar and (ahora - ultimo) < intervalo:
        return resumen, False
    st.session_state.ultimo_cola_ocr_ts = ahora
    hubo_cache = _actualizar_cache_ocr_eventos(config, placa_controlada)
    hubo_cola = _procesar_ocr_cola_eventos(config, placa_controlada)
    hubo = hubo_cache or hubo_cola
    if hubo and frame_actual is not None:
        _procesar_envio_automatico_cola(config, placa_controlada, int(frame_actual))
    elif hubo:
        _procesar_envio_automatico_cola(config, placa_controlada)
    return resumen, hubo


def _obtener_resumen_evento_seleccionado() -> tuple[dict, int | None]:
    cola = st.session_state.get("eventos_monitoreo_cola") or []
    if cola:
        idx = int(st.session_state.get("monitoreo_evento_correo_sel") or 0)
        idx = max(0, min(idx, len(cola) - 1))
        item = cola[idx]
        return dict(item.get("resumen") or {}), int(item.get("evento_id") or 0)
    resumen = st.session_state.get("ultimo_resultado") or st.session_state.get("ultimo_resultado_parcial") or {}
    return dict(resumen), resumen.get("evento_id")


def _limpiar_ocr_asincrono_monitoreo() -> None:
    with _OCR_ASYNC_LOCK:
        _OCR_ASYNC_PENDIENTES.clear()
        _OCR_ASYNC_RESULTADOS.clear()


def _aplicar_resultado_ocr_monitoreo(
    resumen: dict,
    resultado_ocr: dict,
    tiempo_lector_ms: float,
    ruta_recorte: str,
    cache_key: str,
    *,
    registrar_metricas: bool = True,
) -> dict:
    lector_ok = bool(resultado_ocr.get("ok", True))
    texto_crudo = resultado_ocr.get("texto_detectado_crudo") or resultado_ocr.get("texto_crudo") or ""
    texto_corregido = resultado_ocr.get("texto_postprocesado") or resultado_ocr.get("texto_corregido_formato") or ""
    estado_lectura = resultado_ocr.get("estado_lectura") or resultado_ocr.get("estado") or "pendiente"
    motivo_lectura = resultado_ocr.get("motivo") or resultado_ocr.get("motivo_sin_lectura") or resultado_ocr.get("causa_probable") or ""
    if texto_corregido and estado_lectura == "lectura_parcial":
        texto_panel = f"Lectura parcial: {texto_corregido}"
    elif texto_corregido and estado_lectura == "formato_dudoso":
        texto_panel = f"Formato dudoso: {texto_corregido}"
    elif texto_corregido:
        texto_panel = texto_corregido
    elif texto_crudo:
        texto_panel = f"Lectura parcial: {texto_crudo}"
    else:
        texto_panel = f"Sin lectura: {motivo_lectura or estado_lectura}"
    datos_ocr = {
        "texto_ocr_crudo": texto_crudo or texto_panel,
        "texto_ocr_corregido": texto_panel,
        "confianza_ocr": resultado_ocr.get("confianza_promedio") or resultado_ocr.get("confianza_cnn_caracteres"),
        "formato_ocr_valido": bool(resultado_ocr.get("formato_valido") or (resultado_ocr.get("formato") or {}).get("valido")),
        "estado_ocr": estado_lectura,
        "mensaje_ocr": resultado_ocr.get("mensaje") or resultado_ocr.get("error"),
        "causa_probable_ocr": resultado_ocr.get("causa_probable"),
        "motivo_lector_cnn": motivo_lectura,
        "ruta_placa_preprocesada_ocr": (resultado_ocr.get("preprocesamiento") or {}).get("ruta_imagen_procesada"),
        "metodo_rectificacion": resultado_ocr.get("metodo_rectificacion") or (resultado_ocr.get("rectificacion") or {}).get("metodo_rectificacion"),
        "puntaje_rectificacion": resultado_ocr.get("puntaje_rectificacion") or (resultado_ocr.get("rectificacion") or {}).get("confianza_rectificacion"),
        "ruta_rectificacion_ocr": (resultado_ocr.get("rectificacion") or {}).get("ruta_seleccionada"),
        "ruta_debug_segmentacion_ocr": resultado_ocr.get("ruta_debug_segmentacion"),
        "ruta_banda_ocr": resultado_ocr.get("ruta_banda"),
        "ruta_debug_sin_lectura": resultado_ocr.get("ruta_debug_sin_lectura"),
        "estrategia_segmentacion": resultado_ocr.get("estrategia_segmentacion") or (resultado_ocr.get("segmentacion") or {}).get("estrategia_segmentacion"),
        "puntaje_segmentacion": resultado_ocr.get("puntaje_segmentacion") if resultado_ocr.get("puntaje_segmentacion") is not None else (resultado_ocr.get("segmentacion") or {}).get("puntaje_segmentacion"),
        "caracteres_segmentados_ocr": resultado_ocr.get("caracteres_segmentados") or resultado_ocr.get("caracteres", []),
        "cantidad_caracteres_segmentados_ocr": resultado_ocr.get("cantidad_caracteres_segmentados", 0),
        "tiempo_lector_cnn_ms": round(tiempo_lector_ms, 3),
        "lector_cnn_ok": lector_ok,
        "etapa_error_lector_cnn": resultado_ocr.get("etapa_error"),
    }
    lectura_evento = {
        "ruta_recorte": str(ruta_recorte),
        "frame_index": resumen.get("frame_actual") or resumen.get("frames_procesados"),
        "confianza_yolo": (resumen.get("ultima_deteccion") or {}).get("confianza") or (resumen.get("mejor_recorte_placa_info") or {}).get("conf_yolo"),
        "puntaje_recorte": (resumen.get("mejor_recorte_placa_info") or {}).get("puntaje_total"),
        "nitidez": (resumen.get("mejor_recorte_placa_info") or {}).get("nitidez"),
        "area_relativa": (resumen.get("mejor_recorte_placa_info") or {}).get("area_relativa"),
        "aspect_ratio": (resumen.get("mejor_recorte_placa_info") or {}).get("aspect_ratio"),
        "metodo_rectificacion": datos_ocr.get("metodo_rectificacion"),
        "puntaje_rectificacion": datos_ocr.get("puntaje_rectificacion"),
        "estrategia_segmentacion": datos_ocr.get("estrategia_segmentacion"),
        "puntaje_segmentacion": datos_ocr.get("puntaje_segmentacion"),
        "segmentacion_guiada_formato": bool((resultado_ocr.get("segmentacion") or {}).get("segmentacion_guiada_formato")),
        "guion_descartado": bool((resultado_ocr.get("segmentacion") or {}).get("guion_descartado")),
        "motivos_rechazo": resultado_ocr.get("motivos_rechazo") or (resultado_ocr.get("segmentacion") or {}).get("motivos_rechazo") or {},
        "cantidad_caracteres_segmentados": datos_ocr.get("cantidad_caracteres_segmentados_ocr"),
        "texto_crudo": texto_crudo,
        "texto_corregido_formato": texto_corregido,
        "confianza_cnn_caracteres": datos_ocr.get("confianza_ocr"),
        "formato_valido": datos_ocr.get("formato_ocr_valido"),
        "estado_lectura": estado_lectura,
        "predicciones_caracteres": resultado_ocr.get("predicciones_caracteres", []),
    }
    lecturas = st.session_state.get("lecturas_evento_placa_monitoreo", [])
    evento_id = resumen.get("evento_id")
    if evento_id is not None:
        if st.session_state.get("lecturas_evento_id") != int(evento_id):
            lecturas = []
            st.session_state.lecturas_evento_id = int(evento_id)
            st.session_state.lecturas_evento_placa_monitoreo = []
    if not any(item.get("ruta_recorte") == lectura_evento["ruta_recorte"] for item in lecturas):
        lecturas.insert(0, lectura_evento)
    lecturas = lecturas[:8]
    st.session_state.lecturas_evento_placa_monitoreo = lecturas
    consolidado = consolidar_lecturas_evento_placa(list(reversed(lecturas)), max_lecturas=5)
    try:
        ruta_debug_votacion = guardar_debug_votacion_evento(
            resumen.get("evento_id") or resumen.get("eventos_placa") or "monitoreo",
            list(reversed(lecturas)),
            consolidado,
        )
    except Exception:
        ruta_debug_votacion = ""
    datos_ocr.update(
        {
            "placa_individual": texto_corregido or texto_crudo,
            "placa_consolidada_evento": consolidado.get("texto_final") or texto_corregido or texto_crudo,
            "estado_consolidado_evento": consolidado.get("estado"),
            "confianza_final_evento": consolidado.get("confianza_final"),
            "formato_consolidado_valido": consolidado.get("formato_valido"),
            "votos_por_posicion_evento": consolidado.get("votos_por_posicion", []),
            "correcciones_evento": consolidado.get("correcciones_por_formato", []),
            "lecturas_usadas_evento": consolidado.get("cantidad_lecturas_usadas", 0),
            "lecturas_descartadas_evento": consolidado.get("cantidad_lecturas_descartadas", 0),
            "lecturas_descartadas_detalle": consolidado.get("lecturas_descartadas", []),
            "lectura_base_usada": consolidado.get("lectura_base_usada", {}),
            "recortes_usados_evento": len(lecturas),
            "ruta_debug_votacion_evento": ruta_debug_votacion,
        }
    )
    if registrar_metricas:
        try:
            _registrar_metricas_ocr_mejor_recorte(resumen, resultado_ocr)
        except Exception as exc:
            datos_ocr["mensaje_metricas_lector_cnn"] = f"No se pudieron guardar metricas: {exc}"
    st.session_state.ultimo_recorte_ocr_procesado = cache_key
    st.session_state.ultimo_resultado_ocr_monitoreo = datos_ocr
    st.session_state.contador_lector_cnn_monitoreo = int(st.session_state.get("contador_lector_cnn_monitoreo", 0) or 0) + 1
    resumen.update(datos_ocr)
    return resumen


def _tarea_ocr_monitoreo_asincrona(cache_key: str, ruta_recorte: str, contexto: dict, resumen_base: dict) -> None:
    try:
        inicio = time.perf_counter()
        resultado_ocr = leer_placa_cnn_seguro_desde_monitoreo(str(ruta_recorte), contexto=contexto)
        tiempo_ms = (time.perf_counter() - inicio) * 1000
        payload = {
            "resultado_ocr": resultado_ocr,
            "tiempo_lector_ms": tiempo_ms,
            "ruta_recorte": ruta_recorte,
            "resumen_base": dict(resumen_base),
        }
    except Exception as exc:
        payload = {
            "error": str(exc),
            "ruta_recorte": ruta_recorte,
            "resumen_base": dict(resumen_base),
        }
    with _OCR_ASYNC_LOCK:
        _OCR_ASYNC_RESULTADOS[cache_key] = payload
        _OCR_ASYNC_PENDIENTES.discard(cache_key)


def _encolar_ocr_monitoreo_asincrono(resumen: dict, ruta_recorte: str) -> None:
    cache_key = _cache_key_ocr_monitoreo(resumen, ruta_recorte)
    if st.session_state.get("ultimo_recorte_ocr_procesado") == cache_key:
        return
    with _OCR_ASYNC_LOCK:
        if cache_key in _OCR_ASYNC_PENDIENTES or cache_key in _OCR_ASYNC_RESULTADOS:
            return
        _OCR_ASYNC_PENDIENTES.add(cache_key)
    contexto = {
        "funcion": "_encolar_ocr_monitoreo_asincrono",
        "frame_actual": resumen.get("frame_actual"),
        "evento_id": resumen.get("evento_id"),
        "fuente": resumen.get("fuente"),
    }
    threading.Thread(
        target=_tarea_ocr_monitoreo_asincrona,
        args=(cache_key, ruta_recorte, contexto, resumen),
        daemon=True,
    ).start()


def _fusionar_ocr_asincrono_monitoreo(resumen: dict) -> tuple[dict, bool]:
    """Aplica resultados OCR listos sin bloquear el video. Retorna (resumen, hubo_cambio)."""
    ruta_recorte = _resolver_ruta_ocr_monitoreo(resumen)
    if not ruta_recorte:
        return resumen, False
    cache_key = _cache_key_ocr_monitoreo(resumen, ruta_recorte)
    ultimo_key = st.session_state.get("ultimo_recorte_ocr_procesado")
    if ultimo_key == cache_key:
        resumen.update(st.session_state.get("ultimo_resultado_ocr_monitoreo") or {})
        return resumen, False
    if ultimo_key and ultimo_key != cache_key:
        resumen = _limpiar_campos_ocr_resumen(resumen)
    with _OCR_ASYNC_LOCK:
        payload = _OCR_ASYNC_RESULTADOS.pop(cache_key, None)
        pendiente = cache_key in _OCR_ASYNC_PENDIENTES
    if not payload:
        if pendiente:
            return resumen, False
        return resumen, False
    if payload.get("error"):
        datos_ocr = {
            "texto_ocr_crudo": "Sin lectura",
            "texto_ocr_corregido": "Sin lectura",
            "confianza_ocr": None,
            "formato_ocr_valido": False,
            "estado_ocr": "error",
            "mensaje_ocr": payload["error"],
            "lector_cnn_ok": False,
            "etapa_error_lector_cnn": "salida_interfaz",
        }
        st.session_state.ultimo_recorte_ocr_procesado = cache_key
        st.session_state.ultimo_resultado_ocr_monitoreo = datos_ocr
        resumen.update(datos_ocr)
        return resumen, True
    base = dict(payload.get("resumen_base") or resumen)
    base.update({k: v for k, v in resumen.items() if k not in base or v is not None})
    resumen = _aplicar_resultado_ocr_monitoreo(
        base,
        payload["resultado_ocr"],
        float(payload.get("tiempo_lector_ms") or 0.0),
        str(payload.get("ruta_recorte") or ruta_recorte),
        cache_key,
        registrar_metricas=False,
    )
    return resumen, True


def _mostrar_panel_compacto_en_vivo(resumen: dict) -> None:
    """Panel minimo durante monitoreo activo."""
    resumen_panel = _obtener_resumen_panel_vivo(resumen)
    cola = st.session_state.get("eventos_monitoreo_cola") or []
    otro = resumen_panel.get("capturando_otro_vehiculo")
    if otro:
        st.caption(f"Leyendo vehiculo anterior · capturando #{int(otro):03d} en camara")
    elif cola:
        st.caption(f"{len(cola)} vehiculo(s) registrado(s) · evento actual #{int(resumen.get('evento_id') or 0):03d}")
    _render_panel_deteccion_esencial(resumen_panel)


def _enriquecer_resumen_monitoreo_con_ocr(resumen: dict) -> dict:
    ruta_recorte = resumen.get("_ruta_ocr_solicitada") or (
        resumen.get("ruta_recorte_evento_en_vivo")
        or resumen.get("mejor_recorte_placa")
        or resumen.get("ruta_mejor_recorte_evento")
        or resumen.get("ultimo_recorte_placa")
    )
    if not ruta_recorte:
        resumen.setdefault("texto_ocr_crudo", "Pendiente")
        resumen.setdefault("texto_ocr_corregido", "Pendiente")
        resumen.setdefault("confianza_ocr", None)
        resumen.setdefault("formato_ocr_valido", False)
        return resumen

    cache_key = _cache_key_ocr_monitoreo(resumen, str(ruta_recorte))
    if st.session_state.get("ultimo_recorte_ocr_procesado") == cache_key:
        resumen.update(st.session_state.get("ultimo_resultado_ocr_monitoreo") or {})
        return resumen

    try:
        inicio_lector = time.perf_counter()
        resultado_ocr = leer_placa_cnn_seguro_desde_monitoreo(
            str(ruta_recorte),
            contexto={
                "funcion": "_enriquecer_resumen_monitoreo_con_ocr",
                "frame_actual": resumen.get("frame_actual"),
                "evento_id": resumen.get("evento_id"),
                "fuente": resumen.get("fuente"),
            },
        )
        tiempo_lector_ms = (time.perf_counter() - inicio_lector) * 1000
        return _aplicar_resultado_ocr_monitoreo(
            resumen,
            resultado_ocr,
            tiempo_lector_ms,
            str(ruta_recorte),
            cache_key,
            registrar_metricas=True,
        )
    except Exception as exc:
        datos_ocr = {
            "texto_ocr_crudo": "Sin lectura",
            "texto_ocr_corregido": "Sin lectura",
            "confianza_ocr": None,
            "formato_ocr_valido": False,
            "estado_ocr": "error",
            "mensaje_ocr": str(exc),
            "lector_cnn_ok": False,
            "etapa_error_lector_cnn": "salida_interfaz",
        }
        st.session_state.ultimo_recorte_ocr_procesado = cache_key
        st.session_state.ultimo_resultado_ocr_monitoreo = datos_ocr
        resumen.update(datos_ocr)
        return resumen


def _clave_actualizacion_ocr_monitoreo(estado_frame: dict) -> tuple:
    return (
        estado_frame.get("evento_id"),
        estado_frame.get("frame_mejor_evento"),
        estado_frame.get("ruta_recorte_evento_en_vivo"),
        estado_frame.get("mejor_recorte_placa"),
        estado_frame.get("ultimo_recorte_placa"),
    )


def _resolver_ruta_ocr_monitoreo(estado_frame: dict) -> str | None:
    return (
        estado_frame.get("_ruta_ocr_solicitada")
        or estado_frame.get("ruta_recorte_evento_en_vivo")
        or estado_frame.get("mejor_recorte_placa")
        or estado_frame.get("ruta_mejor_recorte_evento")
        or estado_frame.get("ultimo_recorte_placa")
    )


def _registrar_metricas_ocr_mejor_recorte(resumen: dict, resultado_ocr: dict) -> None:
    mejor_info = resumen.get("mejor_recorte_placa_info") or {}
    if not mejor_info:
        return
    salida = Path("reports") / "evidencias" / "mejores_recortes_placa"
    salida.mkdir(parents=True, exist_ok=True)
    csv_path = salida / "metricas_reconocimiento_caracteres_mejores_recortes.csv"
    existe = csv_path.exists()
    fila = {
        "frame_index": mejor_info.get("frame_index"),
        "conf_yolo": mejor_info.get("conf_yolo"),
        "aspect_ratio": mejor_info.get("aspect_ratio"),
        "area_relativa": mejor_info.get("area_relativa"),
        "sharpness": mejor_info.get("nitidez"),
        "puntaje_total": mejor_info.get("puntaje_total"),
        "texto_crudo": resultado_ocr.get("texto_detectado_crudo"),
        "texto_formato": resultado_ocr.get("texto_postprocesado"),
        "formato_valido": (resultado_ocr.get("formato") or {}).get("valido"),
        "cantidad_caracteres_segmentados": resultado_ocr.get("cantidad_caracteres_segmentados"),
        "causa_probable": resultado_ocr.get("causa_probable"),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(fila.keys()))
        if not existe:
            writer.writeheader()
        writer.writerow(fila)


def _obtener_placa_para_evento(resumen: dict, placa_controlada: str) -> str:
    placa = _extraer_placa_desde_resumen(resumen)
    if placa:
        return placa
    respaldo = (placa_controlada or "").strip().upper().replace("-", "").replace(" ", "")
    if respaldo and respaldo not in {"PBC1234", "PENDIENTE"}:
        return respaldo
    return "PBC1234"


def _accion_monitoreo(resumen: dict) -> str:
    notificacion = resumen.get("notificacion_correo") or {}
    if notificacion.get("enviado"):
        return "Correo enviado"
    if notificacion.get("modo") == "simulado":
        return "Correo simulado"
    if notificacion.get("modo") == "descartado":
        return "Descartado (calidad)"
    if resumen.get("puede_enviar_notificacion"):
        return "Pendiente envio auto"
    if (resumen.get("velocidad") or {}).get("velocidad_kmh") is not None:
        return "Velocidad calculada"
    return resumen.get("estado_placa", "Monitoreando")


def _actualizar_historial_monitoreo(resumen: dict) -> None:
    velocidad = resumen.get("velocidad") or {}
    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    selector_activo = "ultimos_candidatos_recorte" in resumen or "mejor_recorte_placa_info" in resumen
    ruta_historial = resumen.get("mejor_recorte_placa") or resumen.get("ruta_mejor_recorte_evento")
    if not selector_activo:
        ruta_historial = ruta_historial or resumen.get("ultimo_recorte_placa")
    clave = (
        resumen.get("evento_id"),
        ruta_historial,
        velocidad.get("frame_cruce_linea_2"),
        bool(resumen.get("notificacion_correo")),
        resumen.get("frames_procesados") or resumen.get("frame_actual"),
    )
    if st.session_state.get("ultima_clave_historial_monitoreo") == clave:
        return
    if not clave[0] and velocidad.get("velocidad_kmh") is None and not resumen.get("notificacion_correo"):
        return

    fila = {
        "hora": datetime.now().strftime("%H:%M:%S"),
        "placa": _extraer_placa_desde_resumen(resumen) or "—",
        "velocidad": (
            f"{velocidad.get('velocidad_kmh'):.0f} km/h"
            if velocidad.get("velocidad_kmh") is not None
            else "—"
        ),
        "confianza": (
            f"CNN {float(resumen.get('confianza_ocr')):.0%}"
            if resumen.get("confianza_ocr") is not None
            else (
                f"YOLO {float(ultima_deteccion.get('confianza')):.0%}"
                if ultima_deteccion.get("confianza") is not None
                else "—"
            )
        ),
    }
    historial = st.session_state.get("historial_detecciones_monitoreo", [])
    historial.insert(0, fila)
    max_historial = int(st.session_state.get("historial_maximo_monitoreo", 15) or 15)
    st.session_state.historial_detecciones_monitoreo = historial[:max_historial]
    st.session_state.ultima_clave_historial_monitoreo = clave


def _actualizar_recortes_monitoreo(resumen: dict) -> None:
    selector_activo = "ultimos_candidatos_recorte" in resumen or "mejor_recorte_placa_info" in resumen
    ruta_recorte = resumen.get("mejor_recorte_placa") or resumen.get("ruta_mejor_recorte_evento")
    if not selector_activo:
        ruta_recorte = ruta_recorte or resumen.get("ultimo_recorte_placa")
    if not ruta_recorte:
        return
    clave = str(ruta_recorte)
    if st.session_state.get("ultima_clave_recorte_monitoreo") == clave:
        return

    item = {
        "ruta": clave,
        "hora": datetime.now().strftime("%H:%M:%S"),
        "lector CNN": resumen.get("placa_consolidada_evento") or resumen.get("texto_ocr_corregido") or resumen.get("texto_ocr_crudo") or "Pendiente",
        "individual": resumen.get("placa_individual") or resumen.get("texto_ocr_corregido") or resumen.get("texto_ocr_crudo") or "Pendiente",
        "confianza_yolo": (resumen.get("ultima_deteccion") or {}).get("confianza"),
        "confianza_ocr": resumen.get("confianza_ocr"),
        "confianza_final": resumen.get("confianza_final_evento"),
    }
    recortes = st.session_state.get("recortes_placas_monitoreo", [])
    if not any(actual.get("ruta") == clave for actual in recortes):
        recortes.insert(0, item)
    max_historial = int(st.session_state.get("historial_maximo_monitoreo", 15) or 15)
    st.session_state.recortes_placas_monitoreo = recortes[:max_historial]
    st.session_state.ultima_clave_recorte_monitoreo = clave


def mostrar_panel_monitoreo_limpio(
    resumen: dict,
    config: dict,
    ejecutar_lector_cnn: bool = True,
    placa_controlada: str = "",
    modo_compacto: bool = False,
) -> dict:
    if modo_compacto:
        _mostrar_panel_compacto_en_vivo(resumen)
        return resumen

    if ejecutar_lector_cnn:
        resumen = _enriquecer_resumen_monitoreo_con_ocr(resumen)
    else:
        resumen.setdefault("texto_ocr_crudo", "Analizando placa..." if resumen.get("mejor_recorte_placa") else "Pendiente")
        resumen.setdefault("texto_ocr_corregido", "Pendiente")
        resumen.setdefault("confianza_ocr", None)
        resumen.setdefault("formato_ocr_valido", False)
    _actualizar_historial_monitoreo(resumen)
    _actualizar_recortes_monitoreo(resumen)

    resumen = _render_panel_deteccion_esencial(
        resumen,
        config=config,
        placa_controlada=placa_controlada,
        mostrar_historial=True,
    )
    st.session_state.ultimo_resultado = resumen

    with st.expander("Detalles tecnicos (opcional)", expanded=False):
        _render_debug_monitoreo(resumen, config)

    return resumen


def _inicializar_estado_monitoreo() -> None:
    valores_iniciales = {
        "monitoreo_activo": False,
        "monitoreo_pausado": False,
        "frame_actual": 0,
        "ultimo_resultado": None,
        "ultima_imagen_procesada": None,
        "ruta_video_monitoreo": None,
        "placas_detectadas_acumuladas": 0,
        "estado_persistencia": None,
        "evento_velocidad_guardado_clave": None,
        "evento_monitoreo_actual": None,
        "ultimo_recorte_ocr_procesado": None,
        "ultimo_resultado_ocr_monitoreo": None,
        "historial_detecciones_monitoreo": [],
        "ultima_clave_historial_monitoreo": None,
        "recortes_placas_monitoreo": [],
        "ultima_clave_recorte_monitoreo": None,
        "contador_lector_cnn_monitoreo": 0,
        "ultima_clave_panel_monitoreo": None,
        "eventos_monitoreo_cola": [],
        "ultimo_evento_cerrado_registrado": 0,
        "ocr_evento_vivo_id": None,
        "lecturas_evento_id": None,
        "ocr_eventos_cache": {},
        "ocr_eventos_encolados": set(),
        "pasos_grupo_notificados": [],
        "pasos_grupo_descartados": [],
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def pestana_monitoreo(config: dict) -> None:
    st.subheader("Monitoreo en vivo")
    st.caption("Detecta placas, lee caracteres y envia notificaciones.")
    _inicializar_estado_monitoreo()

    fuente_col, visor_col = st.columns([0.22, 0.78])
    with fuente_col:
        st.markdown("**Fuente**")
        fuente_monitoreo = st.selectbox(
            "Fuente de monitoreo",
            ["Video de prueba", "Camara en vivo"],
            label_visibility="collapsed",
            key="monitoreo_fuente",
        )
        video = None
        indice_camara = 0

        if fuente_monitoreo == "Video de prueba":
            video = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"], key="video_monitoreo", label_visibility="collapsed")
        else:
            indice_camara = st.selectbox(
                "Camara",
                options=[0, 1, 2, 3],
                index=0,
                label_visibility="collapsed",
                help="Indice de camara. Con Camo suele ser 0 o 1; use 'Probar camaras' abajo.",
                key="monitoreo_indice_camara",
            )
            with st.expander("Probar camaras (Camo / Iriun)", expanded=False):
                resumen_sys = resumen_camaras_sistema()
                nombres_sys = resumen_sys["nombres"]
                if not nombres_sys:
                    st.warning(
                        "No se pudieron leer los nombres de camara en Windows. "
                        "En la terminal del proyecto ejecute: `pip install pygrabber` y reinicie Streamlit."
                    )
                else:
                    st.markdown(f"**{len(nombres_sys)} camara(s) registrada(s) en Windows:**")
                    for i, nombre in enumerate(nombres_sys):
                        st.caption(f"Indice **{i}** → `{nombre}`")
                if resumen_sys["tiene_camo"]:
                    st.success(
                        f"Camo detectado en indice(s) **{resumen_sys['indices_camo']}**. "
                        "Abra Camo Studio con el iPhone conectado antes de escanear o iniciar monitoreo."
                    )
                elif resumen_sys["tiene_iriun"]:
                    st.warning(
                        "Iriun esta instalado pero Camo no. Cierre Iriun Webcam en la PC, instale el driver "
                        "virtual de Camo desde **camo.com** (no Microsoft Store) y vuelva a escanear."
                    )
                else:
                    st.info(
                        "**Camo Studio funciona, pero Windows aun no ve la camara virtual.** "
                        "Ver video del iPhone dentro de Camo Studio es distinto a registrar una camara "
                        "para otras apps (esta app, OBS, Zoom, etc.).\n\n"
                        "En Camo Studio → **Galeria de complementos**:\n"
                        "1. **Compatibilidad con dispositivos Apple** → debe estar **Activo** (conexion iPhone).\n"
                        "2. **Complemento de camara** → debe estar **Activo** (camara virtual en Windows). "
                        "Si dice *Disponible*, abralo e instalelo (pide permisos de administrador).\n\n"
                        "Tambien revise el icono de **sobre/notificaciones** arriba a la derecha en Camo Studio "
                        "por si falta otro driver. Tras instalar, reinicie Camo Studio (o la PC) y vuelva a escanear."
                    )
                indices_scan = indices_a_escanear(max_indice_fallback=3)
                etiqueta_btn = (
                    f"Escanear {len(indices_scan)} camara(s) registrada(s)"
                    if nombres_sys
                    else "Escanear indices 0-3"
                )
                if st.button(etiqueta_btn, use_container_width=True, key="monitoreo_btn_probar_camaras"):
                    with st.spinner("Leyendo camaras..."):
                        pruebas = probar_indices_camara(3, solo_registradas=True)
                    if not pruebas:
                        st.error("No hay camaras registradas para escanear.")
                    for item in pruebas:
                        idx = item["indice"]
                        nombre = item.get("nombre") or f"Camara {idx}"
                        if not item["abierta"]:
                            st.warning(f"Indice {idx} (`{nombre}`): registrada en Windows pero no se pudo abrir")
                            continue
                        etiqueta = f"Indice {idx} · `{nombre}` · puntaje {item['puntaje']}"
                        if item.get("tipo") == "camo" and item.get("tiene_senal"):
                            etiqueta += " · Camo con video"
                            st.success(etiqueta)
                        elif item.get("tipo") == "iriun" and item.get("es_placeholder"):
                            etiqueta += " · Iriun sin iniciar (abra la app Iriun o use Camo)"
                            st.warning(etiqueta)
                        elif item.get("tipo") == "integrada" and item.get("tiene_senal"):
                            etiqueta += " · webcam del laptop (no es el iPhone)"
                            st.info(etiqueta)
                        elif item.get("tiene_senal"):
                            etiqueta += " · video OK"
                            st.success(etiqueta)
                        else:
                            etiqueta += " · sin video util"
                            st.warning(etiqueta)
                        if item.get("frame_rgb") is not None:
                            st.image(item["frame_rgb"], use_container_width=True)
                    camo_ok = [p for p in pruebas if p.get("tipo") == "camo" and p.get("tiene_senal")]
                    otros_ok = [p for p in pruebas if p.get("tipo") != "camo" and p.get("tiene_senal")]
                    if camo_ok:
                        item_sugerido = max(camo_ok, key=lambda p: p["puntaje"])
                        st.success(
                            f"Use indice **{item_sugerido['indice']}** (`{item_sugerido['nombre']}`) "
                            "en el selector de camara."
                        )
                    elif otros_ok and not resumen_sys["tiene_camo"]:
                        item_sugerido = max(otros_ok, key=lambda p: p["puntaje"])
                        st.warning(
                            f"La unica camara con video es indice **{item_sugerido['indice']}** "
                            f"(`{item_sugerido['nombre']}`): es la webcam del laptop, **no el iPhone**. "
                            "Cuando instale el driver de Camo, deberia aparecer otro dispositivo en la lista "
                            "de Windows (por ejemplo `Camo` o `Reincubate Camo`); escanee de nuevo y use ese indice."
                        )
                    elif not otros_ok and not camo_ok:
                        st.error(
                            "Ninguna camara devolvio video util. Revise Camo Studio, drivers virtuales "
                            "y que el iPhone este conectado antes de escanear."
                        )

        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            iniciar = st.button(
                "Reanudar" if st.session_state.get("monitoreo_pausado") else "Iniciar",
                type="primary",
                use_container_width=True,
                key="monitoreo_btn_iniciar",
            )
        with btn2:
            detener = st.button("Pausar", use_container_width=True, key="monitoreo_btn_pausar")
        with btn3:
            reiniciar = st.button("Reiniciar", use_container_width=True, key="monitoreo_btn_reiniciar")
        analizar_video = False
        if fuente_monitoreo == "Video de prueba":
            with st.expander("Analizar video (evidencia)", expanded=False):
                analizar_video = st.button("Generar video anotado", use_container_width=True, key="monitoreo_btn_analizar_video")

        st.markdown("**Correo destino**")
        correo_destino = st.text_input(
            "Correo para notificaciones",
            value=st.session_state.get("correo_destino_monitoreo", ""),
            placeholder="usuario@dominio.com",
            key="correo_destino_monitoreo",
            label_visibility="collapsed",
        )
        if correo_destino and not validar_correo(correo_destino):
            st.warning("Correo invalido.")

        rotacion_ui = "Sin rotacion"
        modo_rendimiento = "Balanceado"
        config_rendimiento = _config_rendimiento_monitoreo(config, modo_rendimiento, fuente_monitoreo)
        max_display_fps = int(config_rendimiento.get("max_display_fps", 0) or 0)

        distancia_metros = float(config["speed"].get("default_distance_meters", 10.0))
        limite_velocidad = float(config["speed"].get("campus_speed_limit_kmh", 30.0))
        posicion_linea_1 = 0.45
        posicion_linea_2 = 0.65
        frecuencia_deteccion = int(config_rendimiento.get("yolo_every_n_frames", 5))
        inference_size = int(config_rendimiento["inference_size"])
        render_every_n_frames = int(config_rendimiento["render_every_n_frames"])
        max_frame_width = int(config_rendimiento.get("max_frame_width", 960) or 0)
        st.session_state.historial_maximo_monitoreo = int(config_rendimiento["history_max"])
        conf_min = 0.45
        persistencia_frames = 4
        max_frames = 0
        velocidad_reproduccion = "Normal (1x)"
        placa_controlada = str(config.get("ocr", {}).get("manual_test_plate", "PBC1234"))
        _opciones_ancho_visual = [360, 420, 480, 640, 800, 960]
        ancho_visual_max = _snap_a_opcion_slider(
            int(config_rendimiento.get("preview_width", 640) or 640),
            _opciones_ancho_visual,
            640,
        )
        guardar_debug_monitoreo = False
        resolucion_camara = "1280x720"
        fps_camara_objetivo = 30
        cooldown_cnn_frames = 15

        with st.expander("Ajustes avanzados", expanded=False):
            rotacion_ui = st.selectbox(
                "Rotacion de imagen",
                ["Sin rotacion", "90 grados", "180 grados", "270 grados"],
                key="monitoreo_rotacion_ui",
            )
            distancia_metros = st.number_input(
                "Distancia real entre lineas (m)",
                min_value=0.1,
                value=distancia_metros,
                step=0.5,
                help="Distancia fisica medida en calle entre Linea 1 y Linea 2. v = distancia / delta_t (tic-toc).",
                key="distancia_monitoreo_avanzada",
            )
            limite_velocidad = st.number_input(
                "Limite de velocidad (km/h)",
                min_value=1.0,
                value=limite_velocidad,
                step=1.0,
                key="limite_monitoreo_avanzada",
            )
            posicion_linea_1 = st.slider("Posicion Linea 1", 0.05, 0.95, posicion_linea_1, 0.01, key="monitoreo_linea_1")
            posicion_linea_2 = st.slider("Posicion Linea 2", 0.05, 0.95, posicion_linea_2, 0.01, key="monitoreo_linea_2")
            frecuencia_deteccion = st.slider(
                "Detectar cada N frames",
                1,
                30,
                frecuencia_deteccion,
                1,
                help="YOLO se ejecuta solo cada N frames. En los frames intermedios se mantiene la ultima bbox visible.",
                key="monitoreo_frecuencia_deteccion",
            )
            inference_size = st.select_slider(
                "Resolucion inferencia YOLO",
                options=[320, 416, 512, 640, 768],
                value=inference_size,
                key="monitoreo_inference_size",
            )
            render_every_n_frames = st.slider(
                "Actualizar video cada N frames", 1, 10, render_every_n_frames, 1, key="monitoreo_render_every"
            )
            max_display_fps = st.slider(
                "FPS maximo visual (0 = sin limite)", 0, 30, max_display_fps, 1, key="monitoreo_max_display_fps"
            )
            if st.session_state.get("monitoreo_ancho_visual") not in (None, *_opciones_ancho_visual):
                st.session_state["monitoreo_ancho_visual"] = _snap_a_opcion_slider(
                    int(st.session_state["monitoreo_ancho_visual"]),
                    _opciones_ancho_visual,
                    640,
                )
            ancho_visual_max = st.select_slider(
                "Ancho maximo visual",
                options=_opciones_ancho_visual,
                value=ancho_visual_max,
                key="monitoreo_ancho_visual",
            )
            if fuente_monitoreo == "Video de prueba":
                max_frame_width = st.select_slider(
                    "Ancho maximo procesamiento video",
                    options=[0, 720, 960, 1280],
                    value=max_frame_width if max_frame_width in {0, 720, 960, 1280} else 960,
                    help="Reduce la resolucion interna del video para acelerar YOLO y la visualizacion. 0 = resolucion original.",
                    key="monitoreo_max_frame_width",
                )
            conf_min = st.slider("Confianza minima YOLO", 0.10, 0.90, conf_min, 0.05, key="monitoreo_conf_min")
            persistencia_frames = st.slider(
                "Persistencia de bbox",
                0,
                30,
                persistencia_frames,
                1,
                help="Mantiene la ultima caja visible aunque YOLO no se ejecute o falle temporalmente.",
                key="monitoreo_persistencia_bbox",
            )
            cooldown_cnn_frames = st.slider(
                "Cooldown lector CNN (frames)",
                5,
                90,
                cooldown_cnn_frames,
                5,
                help="Evita repetir la lectura CNN continuamente sobre la misma placa.",
                key="monitoreo_cooldown_cnn",
            )
            st.session_state.historial_maximo_monitoreo = st.slider(
                "Maximo historial reciente", 5, 30, 10, 1, key="monitoreo_historial_max"
            )
            guardar_debug_monitoreo = st.checkbox("Guardar debug pesado", value=False, key="monitoreo_guardar_debug")
            max_frames = st.number_input(
                "Frames maximos a procesar (0 = completo)",
                min_value=0,
                max_value=10000,
                value=max_frames,
                step=100,
                key="monitoreo_max_frames",
            )
            placa_controlada = st.text_input(
                "Placa de respaldo",
                value=placa_controlada,
                help="Solo si el lector CNN no entrega una placa valida.",
                key="monitoreo_placa_respaldo",
            )
            if fuente_monitoreo == "Camara en vivo":
                resolucion_camara = st.selectbox(
                    "Resolucion de camara",
                    ["640x480", "1280x720"],
                    index=1,
                    key="monitoreo_resolucion_camara",
                )
                fps_camara_objetivo = st.selectbox(
                    "FPS objetivo camara",
                    [15, 24, 30],
                    index=2,
                    key="monitoreo_fps_camara",
                )

        rotacion = {
            "Sin rotacion": "Sin rotacion",
            "90 grados": "Rotar 90 derecha",
            "180 grados": "Rotar 180",
            "270 grados": "Rotar 90 izquierda",
        }[rotacion_ui]

        ancho_camara, alto_camara = [int(valor) for valor in resolucion_camara.split("x")]
        plate_crop_cfg = dict(config.get("plate_crop_selection") or {})
        plate_crop_cfg["cooldown_frames"] = int(cooldown_cnn_frames)
        st.session_state.placa_controlada_monitoreo = placa_controlada
        sidebar_correo_placeholder = st.empty()

    with visor_col:
        frame_placeholder = st.empty()
        progreso = st.progress(0)
        estado_placeholder = st.empty()
        panel_placeholder = st.empty()

    if fuente_monitoreo == "Video de prueba" and video:
        if st.session_state.get("video_monitoreo_nombre") != video.name:
            st.session_state.ruta_video_monitoreo = guardar_archivo_subido(video, Path(config["paths"]["input_dir"]) / "videos")
            st.session_state.video_monitoreo_nombre = video.name
            st.session_state.ultimo_resultado = None
            st.session_state.video_anotado_resultado = None
        with frame_placeholder.container():
            if st.session_state.get("ultima_imagen_procesada") is not None:
                st.image(st.session_state.ultima_imagen_procesada, channels="RGB", use_container_width=True)
            else:
                st.info("Video cargado. Presione Iniciar monitoreo para ver el procesamiento frame por frame.")

    if detener:
        st.session_state.monitoreo_activo = False
        st.session_state.monitoreo_pausado = True

    if reiniciar:
        st.session_state.monitoreo_activo = False
        st.session_state.monitoreo_pausado = False
        st.session_state.frame_actual = 0
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_imagen_procesada = None
        st.session_state.ruta_video_monitoreo = None
        st.session_state.estado_persistencia = None
        st.session_state.evento_velocidad_guardado_clave = None
        st.session_state.evento_monitoreo_actual = None
        st.session_state.ultimo_recorte_ocr_procesado = None
        st.session_state.ultimo_resultado_ocr_monitoreo = None
        st.session_state.historial_detecciones_monitoreo = []
        st.session_state.ultima_clave_historial_monitoreo = None
        st.session_state.recortes_placas_monitoreo = []
        st.session_state.ultima_clave_recorte_monitoreo = None
        st.session_state.lecturas_evento_placa_monitoreo = []
        st.session_state.contador_lector_cnn_monitoreo = 0
        st.session_state.ultima_clave_panel_monitoreo = None
        st.session_state.ultimo_panel_vivo_ts = 0.0
        st.session_state.eventos_monitoreo_cola = []
        st.session_state.ultimo_evento_cerrado_registrado = 0
        st.session_state.ocr_evento_vivo_id = None
        st.session_state.lecturas_evento_id = None
        st.session_state.pasos_grupo_notificados = []
        st.session_state.pasos_grupo_descartados = []
        _limpiar_ocr_asincrono_monitoreo()

    def _refrescar_sidebar_correo() -> None:
        with sidebar_correo_placeholder.container():
            _render_sidebar_envio_correo(
                config,
                st.session_state.get("placa_controlada_monitoreo") or "PBC1234",
            )

    if posicion_linea_2 <= posicion_linea_1:
        st.warning("La Linea 2 debe estar debajo de la Linea 1 para medir movimiento de arriba hacia abajo.")
        _refrescar_sidebar_correo()
        return

    continuar_auto = bool(st.session_state.get("monitoreo_activo") and not st.session_state.get("monitoreo_pausado"))
    ejecutar_monitoreo = bool(iniciar or continuar_auto)

    if (
        not ejecutar_monitoreo
        and not analizar_video
        and not st.session_state.get("ultimo_resultado")
        and st.session_state.get("ultima_imagen_procesada") is None
    ):
        with panel_placeholder.container():
            st.info("Presione **Iniciar** para comenzar el monitoreo.")
        _refrescar_sidebar_correo()
        return

    if ejecutar_monitoreo and fuente_monitoreo == "Video de prueba" and not video and not st.session_state.get("ruta_video_monitoreo"):
        st.warning("Primero cargue un video de prueba.")
        _refrescar_sidebar_correo()
        return

    if iniciar:
        reanudar_video = (
            fuente_monitoreo == "Video de prueba"
            and st.session_state.get("monitoreo_pausado")
            and st.session_state.get("ruta_video_monitoreo")
            and st.session_state.get("frame_actual", 0) > 0
        )
        st.session_state.monitoreo_activo = True
        st.session_state.monitoreo_pausado = False
        if not reanudar_video:
            st.session_state.frame_actual = 0
            st.session_state.ultimo_resultado = None
            st.session_state.ultima_imagen_procesada = None
            st.session_state.estado_persistencia = None
            st.session_state.evento_velocidad_guardado_clave = None
            st.session_state.evento_monitoreo_actual = None
            st.session_state.ultimo_recorte_ocr_procesado = None
            st.session_state.ultimo_resultado_ocr_monitoreo = None
            st.session_state.historial_detecciones_monitoreo = []
            st.session_state.ultima_clave_historial_monitoreo = None
            st.session_state.recortes_placas_monitoreo = []
            st.session_state.ultima_clave_recorte_monitoreo = None
            st.session_state.contador_lector_cnn_monitoreo = 0
            st.session_state.ultima_clave_panel_monitoreo = None
            st.session_state.ultimo_panel_vivo_ts = 0.0
            st.session_state.eventos_monitoreo_cola = []
            st.session_state.ultimo_evento_cerrado_registrado = 0
            st.session_state.ocr_evento_vivo_id = None
            st.session_state.lecturas_evento_id = None
            st.session_state.pasos_grupo_notificados = []
            st.session_state.pasos_grupo_descartados = []
            _limpiar_ocr_asincrono_monitoreo()
        if fuente_monitoreo == "Video de prueba":
            if not reanudar_video:
                st.session_state.ruta_video_monitoreo = guardar_archivo_subido(video, config["paths"]["input_dir"])

    ruta_modelo_placa = config["models"].get("plate_detector_model", config["models"].get("plate_detector_path"))

    if analizar_video:
        if fuente_monitoreo != "Video de prueba" or not video:
            st.warning("Cargue un video de prueba antes de analizar.")
            _refrescar_sidebar_correo()
            return
        ruta_video_demo = st.session_state.get("ruta_video_monitoreo") or guardar_archivo_subido(video, Path(config["paths"]["input_dir"]) / "videos")
        progreso_analisis = st.progress(0)
        estado_analisis = st.empty()

        def _progreso_video_anotado(info: dict) -> None:
            progreso_analisis.progress(float(info.get("porcentaje", 0.0)))
            total = info.get("total_frames") or "?"
            estado_analisis.info(f"{info.get('estado', 'Analizando')} | frame {info.get('frame', 0)} / {total}")

        with st.spinner("Analizando video y generando video anotado..."):
            demo = generar_video_demo_anotado(
                ruta_video_demo,
                distancia_lineas_m=float(distancia_metros),
                limite_velocidad_kmh=float(limite_velocidad),
                posicion_linea_1=float(posicion_linea_1),
                posicion_linea_2=float(posicion_linea_2),
                frecuencia_deteccion=int(frecuencia_deteccion),
                conf_min=float(conf_min),
                persistencia_frames=int(persistencia_frames),
                rotacion=rotacion,
                model_path=ruta_modelo_placa,
                inference_size=int(inference_size),
                max_frames=int(max_frames),
                progreso_callback=_progreso_video_anotado,
            )
        if demo.get("estado") == "ok":
            st.session_state.video_anotado_resultado = demo
            st.session_state.ultimo_resultado = {
                "texto_ocr_crudo": demo.get("texto_reconocido_crudo") or "Pendiente",
                "texto_ocr_corregido": demo.get("texto_corregido_formato") or "Pendiente",
                "confianza_ocr": demo.get("confianza_cnn_caracteres"),
                "ultima_deteccion": {"confianza": demo.get("confianza_yolo")} if demo.get("confianza_yolo") is not None else {},
                "mejor_recorte_placa": demo.get("ultimo_recorte_placa"),
                "ultimo_recorte_placa": demo.get("ultimo_recorte_placa"),
                "frames_procesados": demo.get("frames_procesados"),
                "frames_yolo_analizados": demo.get("frames_yolo_analizados"),
                "estado_placa": "Finalizado",
                "modo_rendimiento": "Video anotado",
            }
            st.success("Video anotado generado correctamente.")
            st.subheader("Video anotado")
            st.video(demo["ruta_video"])
            with panel_placeholder.container():
                mostrar_panel_monitoreo_limpio(
                    st.session_state.ultimo_resultado,
                    config,
                    ejecutar_lector_cnn=False,
                    placa_controlada=placa_controlada,
                )
                eventos = demo.get("eventos") or []
                if eventos:
                    st.subheader("Historial de eventos de placa")
                    st.dataframe(pd.DataFrame(eventos), use_container_width=True, hide_index=True)
                    st.subheader("Mejores recortes por evento")
                    columnas = st.columns(min(4, len(eventos)))
                    for idx, evento in enumerate(eventos[:8]):
                        ruta_recorte = evento.get("ruta_mejor_recorte")
                        if ruta_recorte and Path(ruta_recorte).exists():
                            with columnas[idx % len(columnas)]:
                                st.image(ruta_recorte, use_container_width=True)
                                st.caption(
                                    f"Evento {evento.get('event_id')} | "
                                    f"{evento.get('texto_corregido_formato') or 'Pendiente'} | "
                                    f"YOLO {evento.get('confianza_yolo')}"
                                )
                if Path(demo["ruta_csv"]).exists():
                    try:
                        st.subheader("Historial reciente")
                        st.dataframe(pd.read_csv(demo["ruta_csv"]).tail(10), use_container_width=True, hide_index=True)
                    except Exception:
                        st.caption(f"CSV detecciones: {demo['ruta_csv']}")
            st.caption(f"CSV detecciones: {demo['ruta_csv']}")
            st.caption(f"JSON resumen: {demo['ruta_json']}")
            st.caption(f"Reporte: {demo['ruta_reporte']}")
        else:
            st.error(demo.get("mensaje", "No se pudo analizar el video."))
        _refrescar_sidebar_correo()
        return

    def _actualizar_panel_desde_estado(estado_frame: dict, monitoreo_activo: bool = False) -> None:
        resumen_parcial = dict(estado_frame)
        resumen_parcial.setdefault("frames_procesados", estado_frame.get("frame_actual", 0))
        resumen_parcial.setdefault("fuente", fuente_monitoreo)
        resumen_parcial.setdefault("rotacion", rotacion)
        resumen_parcial.setdefault("modelo_detector_disponible", True)
        resumen_parcial.setdefault("placa_controlada", placa_controlada)
        resumen_parcial.setdefault("modo_rendimiento", modo_rendimiento)
        st.session_state.ultimo_resultado_parcial = resumen_parcial

        if monitoreo_activo:
            _, hubo_cola = _procesar_cola_ocr_monitoreo(config, placa_controlada, resumen_parcial)
            st.session_state.ultimo_resultado_parcial = resumen_parcial
            ahora = time.perf_counter()
            ultimo_panel = float(st.session_state.get("ultimo_panel_vivo_ts") or 0.0)
            if hubo_cola or (ahora - ultimo_panel) >= 0.5:
                with panel_placeholder.container():
                    _mostrar_panel_compacto_en_vivo(resumen_parcial)
                st.session_state.ultimo_panel_vivo_ts = ahora
            return

        ejecutar_lector = modo_rendimiento != "Demo fluido"
        if ejecutar_lector:
            resumen_parcial = _enriquecer_resumen_monitoreo_con_ocr(resumen_parcial)
        with panel_placeholder.container():
            resumen_parcial = mostrar_panel_monitoreo_limpio(
                resumen_parcial,
                config,
                ejecutar_lector_cnn=False,
                placa_controlada=placa_controlada,
            )
        st.session_state.ultimo_resultado = resumen_parcial

    def actualizar_frame(frame_rgb, numero_frame: int, estado_frame: dict | None = None) -> None:
        imagen_ui = _preparar_imagen_streamlit(frame_rgb, int(ancho_visual_max))
        if imagen_ui is not None:
            frame_placeholder.image(imagen_ui, channels="RGB", use_container_width=True)
        if not estado_frame:
            return

        ahora = time.perf_counter()
        solo_video = bool(estado_frame.get("_solo_video"))
        evento_cerrado = estado_frame.get("evento_recien_cerrado")
        ocr_pendiente = estado_frame.get("ocr_evento_pendiente")
        es_camara = fuente_monitoreo != "Video de prueba"

        if solo_video and not ocr_pendiente and not evento_cerrado:
            ultimo_poll = float(st.session_state.get("ultimo_poll_ocr_ts") or 0.0)
            if _hay_ocr_pendiente_en_cache() and (ahora - ultimo_poll) >= 0.3:
                resumen_live = _fusionar_resumen_live_monitoreo(estado_frame, fuente_monitoreo, placa_controlada)
                _, hubo_ocr = _poll_ocr_monitoreo_instantaneo(
                    config, placa_controlada, resumen_live, frame_actual=int(estado_frame.get("frame_actual") or numero_frame)
                )
                st.session_state.ultimo_poll_ocr_ts = ahora
                if hubo_ocr:
                    st.session_state.ultimo_resultado_parcial = resumen_live
                    with panel_placeholder.container():
                        _mostrar_panel_compacto_en_vivo(resumen_live)
                    st.session_state.ultimo_panel_vivo_ts = ahora
            ultimo_estado_ui = float(st.session_state.get("ultimo_estado_ui_ts") or 0.0)
            intervalo_estado = 0.35 if es_camara else 0.5
            if (ahora - ultimo_estado_ui) >= intervalo_estado:
                fps_proc = estado_frame.get("fps_procesamiento", "—")
                estado_placeholder.info(f"Monitoreo en vivo · {fps_proc} FPS")
                st.session_state.ultimo_estado_ui_ts = ahora
            st.session_state.frame_actual = int(estado_frame.get("frame_actual", numero_frame) or 0)
            return

        st.session_state.frame_actual = int(estado_frame.get("frame_actual", numero_frame) or 0)
        frame_ref = int(estado_frame.get("frame_actual", numero_frame) or 0)

        if evento_cerrado:
            _registrar_evento_cerrado_monitoreo(
                evento_cerrado,
                config,
                placa_controlada,
                fuente_monitoreo,
            )

        if ocr_pendiente or not solo_video:
            _solicitar_ocr_desde_estado_frame(
                estado_frame, fuente_monitoreo, placa_controlada, config
            )

        resumen_live = _fusionar_resumen_live_monitoreo(estado_frame, fuente_monitoreo, placa_controlada)
        ultimo_poll = float(st.session_state.get("ultimo_poll_ocr_ts") or 0.0)
        debe_poll = (
            not es_camara
            or bool(evento_cerrado)
            or bool(ocr_pendiente)
            or (ahora - ultimo_poll) >= 0.25
        )
        hubo_ocr = False
        if debe_poll:
            _, hubo_ocr = _poll_ocr_monitoreo_instantaneo(
                config,
                placa_controlada,
                resumen_live,
                frame_actual=frame_ref,
                forzar_cola=bool(evento_cerrado),
            )
            st.session_state.ultimo_poll_ocr_ts = ahora

        ultimo_panel = float(st.session_state.get("ultimo_panel_vivo_ts") or 0.0)
        debe_mostrar_panel = (
            hubo_ocr
            or bool(evento_cerrado)
            or bool(ocr_pendiente)
            or int(estado_frame.get("detecciones_validas") or 0) > 0
            or _hay_ocr_pendiente_en_cache()
            or _evento_ocr_listo_en_cache(estado_frame.get("evento_id"))
        )
        if debe_mostrar_panel and (hubo_ocr or (ahora - ultimo_panel) >= (0.35 if es_camara else 0.15)):
            st.session_state.ultimo_resultado_parcial = resumen_live
            with panel_placeholder.container():
                _mostrar_panel_compacto_en_vivo(resumen_live)
            st.session_state.ultimo_panel_vivo_ts = ahora

        if solo_video:
            ultimo_estado_ui = float(st.session_state.get("ultimo_estado_ui_ts") or 0.0)
            if (ahora - ultimo_estado_ui) >= 0.5:
                placa_ui = _obtener_texto_placa_ui(_obtener_resumen_panel_vivo(resumen_live))
                fps_proc = estado_frame.get("fps_procesamiento", "—")
                if placa_ui.startswith("Capturando"):
                    texto_placa = f" · {placa_ui}"
                elif placa_ui == "Leyendo...":
                    texto_placa = " · Leyendo placa..."
                elif placa_ui != "—":
                    texto_placa = f" · {placa_ui}"
                else:
                    texto_placa = ""
                estado_placeholder.info(f"Monitoreo{texto_placa} · {fps_proc} FPS")
                st.session_state.ultimo_estado_ui_ts = ahora
            return

        if evento_cerrado or hubo_ocr:
            if estado_frame.get("estado_persistencia"):
                st.session_state.estado_persistencia = estado_frame["estado_persistencia"]
            st.session_state.ultimo_persistencia_ui_ts = ahora

        st.session_state.ultimo_resultado_parcial = resumen_live

        ultimo_estado_ui = float(st.session_state.get("ultimo_estado_ui_ts") or 0.0)
        if (ahora - ultimo_estado_ui) >= 0.25:
            confianza = estado_frame.get("ultima_confianza")
            texto_confianza = f"{confianza:.0%}" if confianza is not None else "—"
            fps_proc = estado_frame.get("fps_procesamiento", "—")
            placa_ui = _obtener_texto_placa_ui(_obtener_resumen_panel_vivo(resumen_live))
            if placa_ui.startswith("Capturando"):
                texto_placa = f" · {placa_ui}"
            elif placa_ui == "Leyendo...":
                texto_placa = " · Leyendo placa..."
            elif placa_ui != "—":
                texto_placa = f" · {placa_ui}"
            else:
                texto_placa = ""
            cola_n = len(st.session_state.get("eventos_monitoreo_cola") or [])
            texto_cola = f" · {cola_n} vehiculo(s) registrado(s)" if cola_n else ""
            estado_placeholder.info(
                f"{estado_frame.get('estado_placa', '—')}{texto_placa} · YOLO {texto_confianza} · {fps_proc} FPS{texto_cola}"
            )
            st.session_state.ultimo_estado_ui_ts = ahora

    def actualizar_progreso(valor: float) -> None:
        ahora = time.perf_counter()
        ultimo = float(st.session_state.get("ultimo_progreso_ui_ts") or 0.0)
        if (ahora - ultimo) >= 0.25 or valor >= 0.999:
            progreso.progress(valor)
            st.session_state.ultimo_progreso_ui_ts = ahora

    if not ejecutar_monitoreo:
        if st.session_state.get("ultima_imagen_procesada") is not None:
            frame_placeholder.image(st.session_state.ultima_imagen_procesada, channels="RGB", use_container_width=True)
            if st.session_state.get("monitoreo_pausado"):
                estado_placeholder.info(f"Video pausado en el frame {st.session_state.get('frame_actual', 0)}. Presione Reanudar monitoreo para continuar.")
        if st.session_state.get("ultimo_resultado") or st.session_state.get("ultimo_resultado_parcial"):
            _procesar_ocr_cola_eventos(config, placa_controlada)
            _procesar_envio_automatico_cola(
                config,
                placa_controlada,
                int(st.session_state.get("frame_actual") or 0),
                forzar=True,
            )
            resumen_pausa = dict(st.session_state.get("ultimo_resultado") or st.session_state.get("ultimo_resultado_parcial") or {})
            resumen_pausa = _enriquecer_resumen_monitoreo_con_ocr(resumen_pausa)
            placa_pausa = _obtener_placa_para_evento(resumen_pausa, placa_controlada)
            resumen_pausa = preparar_estado_notificacion_evento(resumen_pausa, placa_pausa, config)
            st.session_state.ultimo_resultado = resumen_pausa
            with panel_placeholder.container():
                mostrar_panel_monitoreo_limpio(
                    resumen_pausa,
                    config,
                    ejecutar_lector_cnn=False,
                    placa_controlada=placa_controlada,
                )
        _refrescar_sidebar_correo()
        return

    max_frames_solicitados = int(max_frames)
    frames_a_procesar = max_frames_solicitados

    with st.spinner("Monitoreando fuente de video..."):
        if fuente_monitoreo == "Video de prueba":
            resumen = procesar_video_monitoreo(
                st.session_state.ruta_video_monitoreo,
                distancia_lineas_m=float(distancia_metros),
                limite_velocidad_kmh=float(limite_velocidad),
                posicion_linea_1=float(posicion_linea_1),
                posicion_linea_2=float(posicion_linea_2),
                frecuencia_deteccion=int(frecuencia_deteccion),
                max_frames=int(frames_a_procesar),
                velocidad_reproduccion=velocidad_reproduccion,
                conf_min=float(conf_min),
                persistencia_frames=int(persistencia_frames),
                rotacion=rotacion,
                model_path=ruta_modelo_placa,
                start_frame=int(st.session_state.get("frame_actual", 0) or 0),
                estado_persistencia=st.session_state.get("estado_persistencia"),
                plate_crop_selection=plate_crop_cfg,
                inference_size=int(inference_size),
                render_every_n_frames=int(render_every_n_frames),
                max_display_fps=int(max_display_fps),
                max_frame_width=int(max_frame_width),
                demo_fluido=modo_rendimiento == "Demo fluido",
                guardar_debug=bool(guardar_debug_monitoreo),
                frame_callback=actualizar_frame,
                progreso_callback=actualizar_progreso,
                detener_callback=lambda: not st.session_state.get("monitoreo_activo", True),
                tiempo_real=True,
            )
        else:
            resumen = procesar_camara_monitoreo(
                indice_camara=int(indice_camara),
                distancia_lineas_m=float(distancia_metros),
                limite_velocidad_kmh=float(limite_velocidad),
                posicion_linea_1=float(posicion_linea_1),
                posicion_linea_2=float(posicion_linea_2),
                frecuencia_deteccion=int(frecuencia_deteccion),
                max_frames=int(max_frames),
                velocidad_reproduccion=velocidad_reproduccion,
                conf_min=float(conf_min),
                persistencia_frames=int(persistencia_frames),
                rotacion=rotacion,
                model_path=ruta_modelo_placa,
                plate_crop_selection=plate_crop_cfg,
                inference_size=int(inference_size),
                render_every_n_frames=int(render_every_n_frames),
                max_display_fps=int(max_display_fps),
                demo_fluido=modo_rendimiento == "Demo fluido",
                camera_width=int(ancho_camara),
                camera_height=int(alto_camara),
                camera_fps=int(fps_camara_objetivo),
                guardar_debug=bool(guardar_debug_monitoreo),
                frame_callback=actualizar_frame,
                progreso_callback=actualizar_progreso,
                detener_callback=lambda: not st.session_state.get("monitoreo_activo", True),
                tiempo_real=True,
            )

    if resumen.get("estado") == "error":
        st.error(resumen.get("mensaje_estado", "No se pudo completar el monitoreo."))
        _refrescar_sidebar_correo()
        return

    estado_final = resumen.get("estado_persistencia") or {}
    for evento in estado_final.get("eventos_cerrados") or []:
        _registrar_evento_cerrado_monitoreo(evento, config, placa_controlada, fuente_monitoreo)
    _procesar_ocr_cola_eventos(config, placa_controlada)
    _procesar_envio_automatico_cola(
        config,
        placa_controlada,
        int(resumen.get("frame_actual") or 0),
        forzar=True,
    )

    resumen = _enriquecer_resumen_monitoreo_con_ocr(resumen)
    placa_evento = _obtener_placa_para_evento(resumen, placa_controlada)
    resumen = preparar_estado_notificacion_evento(resumen, placa_evento, config)
    resumen["velocidad_reproduccion"] = velocidad_reproduccion
    resumen["modo_rendimiento"] = modo_rendimiento
    st.session_state.frame_actual = int(resumen.get("frame_actual", st.session_state.get("frame_actual", 0)) or 0)
    if resumen.get("estado_persistencia"):
        st.session_state.estado_persistencia = resumen["estado_persistencia"]
    st.session_state.ultimo_resultado = resumen
    progreso.progress(1.0)
    with panel_placeholder.container():
        mostrar_panel_monitoreo_limpio(
            resumen,
            config,
            ejecutar_lector_cnn=False,
            placa_controlada=placa_controlada,
        )

    if fuente_monitoreo == "Video de prueba":
        st.session_state.monitoreo_activo = False
        st.session_state.monitoreo_pausado = False
        estado_placeholder.success("Video finalizado.")
    _refrescar_sidebar_correo()

def pestana_pruebas(config: dict) -> None:
    st.header("Pruebas")

    with st.expander("Pruebas con imagen", expanded=True):
        imagen = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png", "bmp"], key="imagen_prueba")
        if imagen:
            ruta_imagen = guardar_archivo_subido(imagen, config["paths"]["input_dir"])
            st.image(ruta_imagen, use_container_width=True)
        else:
            ruta_imagen = None

    with st.expander("SimulaciÃ³n de velocidad", expanded=True):
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            fps_simulado = st.number_input("FPS simulado", min_value=1.0, value=30.0, step=1.0)
            placa_simulada = st.text_input(
                "Placa manual/controlada",
                value="PBC1234",
                help="Valor temporal para probar base de datos y eventos hasta integrar lector CNN automatico.",
            )
            distancia_simulada = st.number_input(
                "Distancia real entre lÃ­neas en metros",
                min_value=0.1,
                value=10.0,
                step=0.5,
                key="distancia_simulacion_velocidad",
            )
            limite_simulado = st.number_input(
                "LÃ­mite de velocidad del campus (km/h)",
                min_value=1.0,
                value=30.0,
                step=1.0,
                key="limite_simulacion_velocidad",
            )
            posicion_sim_linea_1 = st.slider("PosiciÃ³n LÃ­nea 1 (% altura)", 0.05, 0.95, 0.35, 0.01)
            posicion_sim_linea_2 = st.slider("PosiciÃ³n LÃ­nea 2 (% altura)", 0.05, 0.95, 0.75, 0.01)
        with col_sim2:
            frame_inicial_sim = st.number_input("Frame inicial de la placa", min_value=0, value=0, step=1)
            frame_cruce_linea_1_sim = st.number_input("Frame en que cruza LÃ­nea 1", min_value=0, value=30, step=1)
            frame_cruce_linea_2_sim = st.number_input("Frame en que cruza LÃ­nea 2", min_value=0, value=75, step=1)
            total_frames_sim = st.number_input("Total de frames simulados", min_value=1, value=120, step=1)

        ejecutar_simulacion = st.button("Ejecutar simulaciÃ³n de velocidad", type="primary")

        if ejecutar_simulacion:
            if posicion_sim_linea_2 <= posicion_sim_linea_1:
                st.warning("La LÃ­nea 2 debe estar debajo de la LÃ­nea 1 para simular movimiento de arriba hacia abajo.")
            elif frame_cruce_linea_2_sim <= frame_cruce_linea_1_sim:
                st.warning("El frame de cruce de LÃ­nea 2 debe ser mayor que el frame de cruce de LÃ­nea 1.")
            elif total_frames_sim <= frame_cruce_linea_2_sim:
                st.warning("El total de frames simulados debe ser mayor que el frame de cruce de LÃ­nea 2.")
            else:
                resultado_simulacion = ejecutar_simulacion_velocidad(
                    fps=float(fps_simulado),
                    distancia_metros=float(distancia_simulada),
                    limite_kmh=float(limite_simulado),
                    placa_manual=placa_simulada,
                    posicion_linea_1=float(posicion_sim_linea_1),
                    posicion_linea_2=float(posicion_sim_linea_2),
                    frame_inicial=int(frame_inicial_sim),
                    frame_linea_1=int(frame_cruce_linea_1_sim),
                    frame_linea_2=int(frame_cruce_linea_2_sim),
                    total_frames=int(total_frames_sim),
                )
                resumen_velocidad = resultado_simulacion["resumen"]
                datos_simulacion = resultado_simulacion["datos"]
                difuso = resultado_simulacion.get("difuso")

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Frame cruce LÃ­nea 1", resumen_velocidad.get("frame_cruce_linea_1") or "Pendiente")
                col_res2.metric("Frame cruce LÃ­nea 2", resumen_velocidad.get("frame_cruce_linea_2") or "Pendiente")
                col_res3.metric("Estado tracker", resumen_velocidad.get("estado", "Pendiente"))

                col_res4, col_res5, col_res6 = st.columns(3)
                tiempo_entre = resumen_velocidad.get("tiempo_entre_lineas")
                velocidad_kmh = resumen_velocidad.get("velocidad_kmh")
                col_res4.metric("Tiempo entre lÃ­neas", f"{tiempo_entre:.3f} s" if tiempo_entre is not None else "Pendiente")
                col_res5.metric("Distancia configurada", f"{distancia_simulada:.1f} m")
                col_res6.metric("Velocidad calculada", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "Pendiente")

                if difuso:
                    st.subheader("ClasificaciÃ³n difusa de la simulaciÃ³n")
                    col_dif1, col_dif2, col_dif3 = st.columns(3)
                    col_dif1.metric("Estado", difuso["estado"])
                    col_dif2.metric("Nivel de infracciÃ³n", difuso["nivel_infraccion"])
                    col_dif3.metric("Horas de suspensiÃ³n", difuso["horas_suspension"])

                    col_dif4, col_dif5 = st.columns(2)
                    col_dif4.metric("SanciÃ³n", difuso["sancion"])
                    col_dif5.metric("LÃ­mite evaluado", f"{difuso['limite_kmh']:.1f} km/h")
                    st.info(difuso["mensaje"])
                    st.json(difuso["grados"])

                frame_final_rgb = resultado_simulacion.get("frame_final_rgb")
                if frame_final_rgb is not None:
                    st.image(frame_final_rgb, channels="RGB", caption="Frame final de la simulaciÃ³n", width=640)

                st.caption(f"Evidencias guardadas en: reports/evidencias/simulacion_velocidad/")
                st.caption(f"Resumen JSON: {resultado_simulacion['ruta_json']}")
                st.json(datos_simulacion)

    with st.expander("Generar caracteres desde placas ecuatorianas", expanded=False):
        st.caption(
            "Use esta herramienta para reforzar el dataset de caracteres CNN con caracteres segmentados de placas reales. "
            "No usa motores OCR externos ni entrena el modelo."
        )
        imagenes_placas = listar_imagenes_placas()
        placa_generacion = st.text_input(
            "Placa esperada",
            value="",
            key="placa_generacion_caracteres",
            help="Formato aceptado: ABC123, ABC1234, ABC-123 o ABC-1234.",
        )
        usar_yolo_generacion = st.checkbox(
            "Usar YOLO si la imagen no parece recorte de placa",
            value=False,
            key="usar_yolo_generacion_caracteres",
        )

        if not imagenes_placas:
            st.info("No se encontraron imagenes en datasets/placas_ecuador/.")
        else:
            imagen_seleccionada = st.selectbox(
                "Imagen fuente",
                imagenes_placas,
                format_func=lambda ruta: str(ruta.relative_to(Path.cwd())),
                key="imagen_generacion_caracteres",
            )
            st.image(str(imagen_seleccionada), caption="Imagen seleccionada", use_container_width=False)

            col_gen_a, col_gen_b = st.columns(2)
            segmentar_generacion = col_gen_a.button("Segmentar caracteres", type="primary")
            guardar_generacion = col_gen_b.button("Guardar caracteres segmentados")

            if segmentar_generacion:
                placa_normalizada = normalizar_placa_esperada(placa_generacion)
                if not placa_normalizada:
                    st.warning("Ingrese una placa esperada valida antes de segmentar.")
                else:
                    imagen_cv = cv2.imread(str(imagen_seleccionada))
                    if imagen_cv is None:
                        st.error("No se pudo cargar la imagen seleccionada.")
                    else:
                        detector = PlateDetector() if usar_yolo_generacion else None
                        ruta_recorte, estado_recorte, motivo_recorte = preparar_recorte_placa(
                            imagen_seleccionada,
                            imagen_cv,
                            usar_yolo_generacion,
                            detector,
                        )
                        if ruta_recorte is None:
                            st.warning(motivo_recorte or "No se pudo obtener recorte de placa.")
                        else:
                            nombre_base = f"{imagen_seleccionada.stem}_{placa_normalizada}_{estado_recorte}"
                            preprocesamiento = preprocesar_placa(str(ruta_recorte), nombre_base)
                            ruta_preprocesada = preprocesamiento.get("ruta_imagen_procesada")
                            if not ruta_preprocesada:
                                st.error(preprocesamiento.get("mensaje", "Error de preprocesamiento."))
                            else:
                                segmentacion = segmentar_caracteres_v2(ruta_preprocesada, nombre_base)
                                caracteres = segmentacion.get("caracteres", [])
                                st.session_state["generacion_caracteres_ecuador"] = {
                                    "ruta_original": imagen_seleccionada,
                                    "placa": placa_normalizada,
                                    "ruta_recorte": str(ruta_recorte),
                                    "preprocesamiento": preprocesamiento,
                                    "segmentacion": segmentacion,
                                    "caracteres": caracteres,
                                }

            resultado_generacion = st.session_state.get("generacion_caracteres_ecuador")
            if resultado_generacion:
                placa_normalizada = resultado_generacion["placa"]
                caracteres = resultado_generacion.get("caracteres", [])
                cantidad_esperada = len(placa_normalizada)
                cantidad_segmentada = len(caracteres)

                col_res_1, col_res_2, col_res_3 = st.columns(3)
                col_res_1.metric("Placa normalizada", placa_normalizada)
                col_res_2.metric("Caracteres esperados", cantidad_esperada)
                col_res_3.metric("Caracteres segmentados", cantidad_segmentada)

                ruta_preprocesada = resultado_generacion.get("preprocesamiento", {}).get("ruta_imagen_procesada")
                if ruta_preprocesada:
                    st.image(ruta_preprocesada, caption="Placa preprocesada", use_container_width=False)
                ruta_banda = resultado_generacion.get("segmentacion", {}).get("ruta_banda")
                if ruta_banda:
                    st.image(ruta_banda, caption="Banda de caracteres", use_container_width=False)
                ruta_debug = resultado_generacion.get("segmentacion", {}).get("ruta_debug")
                if ruta_debug:
                    st.image(ruta_debug, caption="Debug de segmentacion", use_container_width=False)

                if caracteres:
                    columnas = st.columns(min(len(caracteres), 8))
                    indices_guardar = []
                    for idx, caracter in enumerate(caracteres[:16]):
                        etiqueta = placa_normalizada[idx] if idx < len(placa_normalizada) else ""
                        columna = columnas[idx % len(columnas)]
                        columna.image(
                            caracter["ruta_caracter"],
                            caption=f"{idx + 1}: {etiqueta}",
                            use_container_width=True,
                        )
                        guardar_item = columna.checkbox(
                            "Guardar este caracter",
                            value=True,
                            key=f"guardar_caracter_ecuador_{resultado_generacion['ruta_original']}_{placa_normalizada}_{idx}",
                        )
                        if guardar_item:
                            indices_guardar.append(idx)
                    st.session_state["indices_guardar_caracteres_ecuador"] = indices_guardar
                    st.caption(
                        f"Caracteres seleccionados para guardar: {len(indices_guardar)} de {cantidad_esperada} esperados."
                    )
                else:
                    indices_guardar = []

                if cantidad_segmentada != cantidad_esperada:
                    st.warning("No se guardan caracteres porque la cantidad segmentada no coincide con la placa esperada.")
                    st.info("Puede desmarcar segmentos de ruido y guardar solo si los caracteres seleccionados coinciden con la placa esperada.")

                if guardar_generacion:
                    indices_guardar = st.session_state.get("indices_guardar_caracteres_ecuador", [])
                    placa_valida, motivo_placa = validar_placa_para_guardado(placa_normalizada)
                    caracteres_validos = all(caracter in CLASES_CARACTERES_ECUADOR for caracter in placa_normalizada)
                    if not placa_valida or not caracteres_validos:
                        st.warning(motivo_placa or "La placa esperada contiene caracteres no permitidos.")
                        registrar_auditoria_guardado_caracteres(
                            resultado_generacion["ruta_original"],
                            placa_generacion,
                            placa_normalizada,
                            "no_guardado",
                            motivo_placa or "Caracteres no permitidos.",
                            cantidad_segmentada,
                            cantidad_esperada,
                            [],
                        )
                    elif len(indices_guardar) != cantidad_esperada:
                        mensaje_no_guardado = "No se guardan caracteres porque la cantidad segmentada no coincide con la placa esperada."
                        st.warning(mensaje_no_guardado)
                        registrar_auditoria_guardado_caracteres(
                            resultado_generacion["ruta_original"],
                            placa_generacion,
                            placa_normalizada,
                            "no_guardado",
                            mensaje_no_guardado,
                            cantidad_segmentada,
                            cantidad_esperada,
                            [],
                        )
                    else:
                        filas = leer_labels_caracteres_ecuador()
                        existentes = {
                            fila.get("ruta_imagen", "")
                            for fila in filas
                            if fila.get("origen") == "caracter_ecuador_segmentado"
                        }
                        try:
                            cantidad, rutas = guardar_caracteres_ecuador(
                                resultado_generacion["ruta_original"],
                                placa_normalizada,
                                caracteres,
                                filas,
                                existentes,
                                dry_run=False,
                                indices_guardar=indices_guardar,
                            )
                        except ValueError as exc:
                            st.warning(str(exc))
                            registrar_auditoria_guardado_caracteres(
                                resultado_generacion["ruta_original"],
                                placa_generacion,
                                placa_normalizada,
                                "no_guardado",
                                str(exc),
                                cantidad_segmentada,
                                cantidad_esperada,
                                [],
                            )
                        else:
                            escribir_labels_caracteres_ecuador(filas)
                            registrar_auditoria_guardado_caracteres(
                                resultado_generacion["ruta_original"],
                                placa_generacion,
                                placa_normalizada,
                                "guardado",
                                "Caracteres guardados usando etiquetas de placa esperada manual.",
                                cantidad_segmentada,
                                cantidad_esperada,
                                rutas,
                            )
                            st.success(f"Caracteres guardados: {cantidad}")
                            st.caption(f"Destino: {CARACTERES_ECUADOR_DIR}")
                            if rutas:
                                st.write(rutas)

    with st.expander("Diagnóstico YOLO placas", expanded=False):
        st.caption("Diagnóstico aislado del detector YOLO. No ejecuta reconocimiento de caracteres ni modifica el modelo.")
        detector_yolo = PlateDetector()
        modelo_existe = detector_yolo.model_path.exists()
        clases_modelo = getattr(detector_yolo.model, "names", {}) if detector_yolo.model is not None else {}

        col_yolo_info1, col_yolo_info2, col_yolo_info3 = st.columns(3)
        col_yolo_info1.metric("Modelo YOLO", str(detector_yolo.model_path))
        col_yolo_info2.metric("Modelo existe", "Sí" if modelo_existe else "No")
        col_yolo_info3.metric("Estado carga", detector_yolo.estado)
        st.caption(f"Clases del modelo: {clases_modelo}")

        rutas_yolo = []
        for carpeta in [
            Path("datasets") / "placas_ecuador" / "images",
            Path("reports") / "evidencias" / "monitoreo_video",
            Path("reports") / "evidencias" / "eventos_placa",
        ]:
            if carpeta.exists():
                rutas_yolo.extend(
                    sorted(
                        ruta
                        for ruta in carpeta.rglob("*")
                        if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    )
                )

        conf_yolo_diag = st.slider("Confidence threshold YOLO", 0.05, 0.90, 0.25, 0.05)
        ruta_yolo = None
        if rutas_yolo:
            ruta_yolo = st.selectbox(
                "Imagen de prueba",
                rutas_yolo,
                format_func=lambda ruta: str(ruta),
                key="imagen_diagnostico_yolo",
            )
            st.image(str(ruta_yolo), caption="Imagen original", use_container_width=False)
        else:
            st.info("No se encontraron imágenes en datasets/placas_ecuador/images ni evidencias.")

        if st.button("Ejecutar detección YOLO", type="primary"):
            if ruta_yolo is None:
                st.warning("Selecciona una imagen antes de ejecutar YOLO.")
            elif detector_yolo.model is None:
                st.error(detector_yolo.estado)
            else:
                imagen = cv2.imread(str(ruta_yolo))
                if imagen is None:
                    st.error("No se pudo leer la imagen seleccionada.")
                else:
                    resultado_yolo = detector_yolo.detectar_en_frame(imagen, conf_min=float(conf_yolo_diag))
                    st.metric("Confidence threshold usado", f"{conf_yolo_diag:.2f}")
                    st.info(resultado_yolo.get("mensaje", "Sin mensaje."))
                    frame_rgb = asegurar_rgb(resultado_yolo["frame_procesado"])
                    st.image(frame_rgb, caption="Resultado con bounding boxes", use_container_width=False)

                    detecciones = resultado_yolo.get("detecciones", [])
                    col_yolo_res1, col_yolo_res2, col_yolo_res3 = st.columns(3)
                    col_yolo_res1.metric("Detectada", "Sí" if resultado_yolo.get("detectada") else "No")
                    col_yolo_res2.metric("Detecciones válidas", len(detecciones))
                    col_yolo_res3.metric("Detecciones brutas", resultado_yolo.get("detecciones_brutas", 0))

                    if not detecciones:
                        debug = resultado_yolo.get("debug_detecciones") or []
                        motivo = debug[-1].get("motivo_rechazo") if debug else "Sin detección YOLO sobre el threshold."
                        st.warning(f"Motivo: {motivo}")
                    else:
                        filas_yolo = []
                        columnas_recortes = st.columns(min(len(detecciones), 4))
                        for idx, det in enumerate(detecciones, start=1):
                            bbox = det.get("bbox", [])
                            filas_yolo.append(
                                {
                                    "idx": idx,
                                    "confianza": round(float(det.get("confianza", 0.0)), 4),
                                    "bbox": bbox,
                                    "area_relativa": round(float(det.get("area_relativa", 0.0)), 6),
                                }
                            )
                            recorte = det.get("recorte_placa")
                            if recorte is not None:
                                columnas_recortes[(idx - 1) % len(columnas_recortes)].image(
                                    asegurar_rgb(recorte),
                                    caption=f"Recorte {idx}",
                                    use_container_width=True,
                                )
                        st.dataframe(pd.DataFrame(filas_yolo), use_container_width=True)

    with st.expander("Comparar detectores YOLO", expanded=False):
        st.caption("Comparación visual entre el modelo actual y el candidato entrenado con Roboflow.")
        modelo_actual_path = Path("models") / "plate_detector" / "placas_ecuador.pt"
        modelo_roboflow_path = Path("models") / "plate_detector" / "placas_roboflow.pt"
        conf_comp = st.slider("Confidence threshold comparación", 0.05, 0.90, 0.25, 0.05)

        col_cmp_info1, col_cmp_info2 = st.columns(2)
        col_cmp_info1.metric("Modelo actual", "Existe" if modelo_actual_path.exists() else "No existe")
        col_cmp_info1.caption(str(modelo_actual_path))
        col_cmp_info2.metric("Modelo Roboflow", "Existe" if modelo_roboflow_path.exists() else "No existe")
        col_cmp_info2.caption(str(modelo_roboflow_path))

        fuentes_cmp = []
        for carpeta in [
            Path("datasets") / "placas_ecuador" / "images",
            Path("datasets") / "placas_yolo_roboflow" / "test",
            Path("reports") / "evidencias" / "monitoreo_video",
            Path("data") / "videos_prueba",
        ]:
            if carpeta.exists():
                fuentes_cmp.extend(
                    sorted(
                        ruta
                        for ruta in carpeta.rglob("*")
                        if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
                    )
                )

        if fuentes_cmp:
            ruta_cmp = st.selectbox(
                "Imagen o video para comparar",
                fuentes_cmp,
                format_func=lambda ruta: str(ruta),
                key="comparar_yolo_fuente",
            )
            frame_cmp = 0
            if ruta_cmp.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                frame_cmp = st.number_input("Frame del video", min_value=0, value=0, step=30)
        else:
            ruta_cmp = None
            st.info("No se encontraron imágenes o videos para comparar.")

        def _leer_imagen_o_frame(ruta, frame_idx=0):
            if ruta.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                cap = cv2.VideoCapture(str(ruta))
                if not cap.isOpened():
                    return None
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                cap.release()
                return frame if ok else None
            return cv2.imread(str(ruta))

        def _inferir_yolo_streamlit(model_path, imagen, conf):
            if not model_path.exists():
                return {
                    "disponible": False,
                    "mensaje": "Modelo no encontrado.",
                    "frame": imagen,
                    "detecciones": [],
                    "tiempo_ms": 0.0,
                    "clases": {},
                }
            try:
                from ultralytics import YOLO
            except ImportError:
                return {
                    "disponible": False,
                    "mensaje": "Ultralytics no está instalado en este entorno.",
                    "frame": imagen,
                    "detecciones": [],
                    "tiempo_ms": 0.0,
                    "clases": {},
                }
            import time

            model = YOLO(str(model_path))
            inicio = time.perf_counter()
            results = model.predict(source=imagen, conf=float(conf), verbose=False)
            tiempo_ms = (time.perf_counter() - inicio) * 1000
            boxes = results[0].boxes if results else []
            frame = imagen.copy()
            detecciones = []
            alto, ancho = imagen.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                confianza = float(box.conf[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ancho - 1, x2), min(alto - 1, y2)
                recorte = imagen[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(frame, f"{confianza:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
                detecciones.append({"bbox": [x1, y1, x2, y2], "confianza": confianza, "recorte": recorte})
            return {
                "disponible": True,
                "mensaje": "OK" if detecciones else "Sin detección.",
                "frame": frame,
                "detecciones": detecciones,
                "tiempo_ms": tiempo_ms,
                "clases": getattr(model, "names", {}),
            }

        if st.button("Comparar detectores", type="primary"):
            if ruta_cmp is None:
                st.warning("Selecciona una fuente para comparar.")
            else:
                imagen_cmp = _leer_imagen_o_frame(ruta_cmp, frame_cmp)
                if imagen_cmp is None:
                    st.error("No se pudo leer la imagen o frame seleccionado.")
                else:
                    res_actual = _inferir_yolo_streamlit(modelo_actual_path, imagen_cmp, conf_comp)
                    res_robo = _inferir_yolo_streamlit(modelo_roboflow_path, imagen_cmp, conf_comp)
                    col_actual, col_robo = st.columns(2)
                    with col_actual:
                        st.subheader("Modelo actual")
                        st.caption(f"Clases: {res_actual.get('clases')}")
                        st.metric("Tiempo inferencia", f"{res_actual['tiempo_ms']:.2f} ms")
                        st.metric("Detecciones", len(res_actual["detecciones"]))
                        st.info(res_actual["mensaje"])
                        st.image(asegurar_rgb(res_actual["frame"]), caption="BBox actual", use_container_width=True)
                        if res_actual["detecciones"] and res_actual["detecciones"][0].get("recorte") is not None:
                            st.image(asegurar_rgb(res_actual["detecciones"][0]["recorte"]), caption="Recorte actual", use_container_width=False)
                        if res_actual["detecciones"]:
                            st.json([{k: v for k, v in det.items() if k != "recorte"} for det in res_actual["detecciones"]])
                    with col_robo:
                        st.subheader("Modelo Roboflow")
                        st.caption(f"Clases: {res_robo.get('clases')}")
                        st.metric("Tiempo inferencia", f"{res_robo['tiempo_ms']:.2f} ms")
                        st.metric("Detecciones", len(res_robo["detecciones"]))
                        st.info(res_robo["mensaje"])
                        st.image(asegurar_rgb(res_robo["frame"]), caption="BBox Roboflow", use_container_width=True)
                        if res_robo["detecciones"] and res_robo["detecciones"][0].get("recorte") is not None:
                            st.image(asegurar_rgb(res_robo["detecciones"][0]["recorte"]), caption="Recorte Roboflow", use_container_width=False)
                        if res_robo["detecciones"]:
                            st.json([{k: v for k, v in det.items() if k != "recorte"} for det in res_robo["detecciones"]])

    with st.expander("Reconocimiento CNN sobre recortes", expanded=False):
        recortes, fuente_recorte = _buscar_recortes_ocr()
        modo_ocr_exp = st.radio(
            "Fuente de recorte para lector CNN",
            ["Recorte generado por el sistema", "Carga manual"],
            horizontal=True,
        )
        placa_esperada_ocr = st.text_input(
            "Placa esperada para diagnostico",
            value="",
            key="placa_esperada_ocr",
            help="Opcional. Ejemplos: PDZ279, PDP6236, TDH493. Se usa solo para comparar caracter por caracter.",
        )
        metodo_segmentacion_ocr = st.selectbox(
            "MÃ©todo de segmentaciÃ³n",
            ["v2_banda_caracteres", "v1_contornos_globales"],
            index=0,
        )
        ruta_ocr = None
        fuente_ocr = fuente_recorte

        if modo_ocr_exp == "Recorte generado por el sistema":
            if recortes:
                seleccionado = st.selectbox(
                    "Seleccionar recorte",
                    recortes,
                    format_func=lambda ruta: f"{ruta.name} ({fuente_recorte})",
                )
                ruta_ocr = str(seleccionado)
                st.image(ruta_ocr, caption="Recorte original", use_container_width=False)
            else:
                st.info("Aun no hay recortes en eventos_placa ni placas_detectadas.")
        else:
            imagen_ocr = st.file_uploader(
                "Cargar imagen de placa para reconocimiento CNN",
                type=["jpg", "jpeg", "png", "bmp"],
                key="imagen_ocr_experimental",
            )
            if imagen_ocr:
                ruta_ocr = guardar_archivo_subido(imagen_ocr, config["paths"]["input_dir"])
                fuente_ocr = "carga_manual"
                st.image(ruta_ocr, caption="Imagen cargada", use_container_width=False)

        ejecutar_ocr = st.button("Ejecutar reconocimiento CNN", type="primary")
        if ejecutar_ocr:
            if not ruta_ocr:
                st.warning("Selecciona o carga un recorte de placa antes de ejecutar reconocimiento CNN.")
            else:
                resultado_ocr = leer_placa_desde_recorte(
                    ruta_ocr,
                    placa_esperada=placa_esperada_ocr,
                    metodo_segmentacion=metodo_segmentacion_ocr,
                )
                reporte_ocr = registrar_reporte_ocr(resultado_ocr, fuente_ocr, placa_esperada_ocr)
                reporte_diagnostico = registrar_diagnostico_recorte(resultado_ocr, placa_esperada_ocr)

                col_ocr1, col_ocr2, col_ocr3 = st.columns(3)
                col_ocr1.metric("Texto reconocido crudo", resultado_ocr.get("texto_detectado_crudo") or resultado_ocr["texto_detectado"])
                col_ocr2.metric("Texto corregido por formato", resultado_ocr.get("texto_postprocesado") or "Pendiente")
                col_ocr3.metric("Confianza CNN caracteres", f"{resultado_ocr.get('confianza_promedio', 0.0):.2f}")

                formato = resultado_ocr.get("formato", {})
                comparacion = resultado_ocr.get("comparacion", {})
                col_ocr4, col_ocr5, col_ocr6 = st.columns(3)
                col_ocr4.metric("Formato vÃ¡lido", "SÃ­" if formato.get("valido") else "No")
                col_ocr5.metric("Acierto", "SÃ­" if comparacion.get("coincide") else "No")
                col_ocr6.metric("Caracteres aceptados", resultado_ocr["cantidad_caracteres_segmentados"])

                col_ocr7, col_ocr8 = st.columns(2)
                col_ocr7.metric("Contornos rechazados", resultado_ocr.get("cantidad_contornos_rechazados", 0))
                col_ocr8.metric("MÃ©todo", resultado_ocr.get("metodo_segmentacion", metodo_segmentacion_ocr))

                st.caption(f"Estado: {resultado_ocr['estado']}")
                st.info(resultado_ocr["mensaje"])
                st.caption(comparacion.get("mensaje", "Sin comparaciÃ³n."))

                diagnostico = resultado_ocr.get("diagnostico", {})
                st.subheader("Diagnostico del lector CNN")
                col_diag1, col_diag2, col_diag3 = st.columns(3)
                col_diag1.metric("Caracteres esperados", diagnostico.get("cantidad_esperada", 0))
                col_diag2.metric("Caracteres detectados", diagnostico.get("cantidad_detectada", 0))
                col_diag3.metric("Causa probable", diagnostico.get("causa_probable", "Sin diagnostico"))
                st.info(diagnostico.get("mensaje", "Sin diagnostico disponible."))
                if diagnostico.get("cantidad_esperada") and diagnostico.get("cantidad_detectada") != diagnostico.get("cantidad_esperada"):
                    st.warning(
                        "Caracteres esperados: "
                        + " ".join(diagnostico.get("caracteres_esperados", []))
                        + " | Caracteres detectados: "
                        + " ".join(diagnostico.get("caracteres_detectados", []))
                        + ". Posible causa: segmentacion incompleta o caracteres descartados."
                    )

                preprocesamiento = resultado_ocr.get("preprocesamiento", {})
                ruta_preprocesada = preprocesamiento.get("ruta_imagen_procesada")
                if ruta_preprocesada:
                    st.image(ruta_preprocesada, caption="Imagen preprocesada final", use_container_width=False)

                ruta_banda = resultado_ocr.get("ruta_banda")
                if ruta_banda:
                    st.image(ruta_banda, caption="Banda principal de caracteres", use_container_width=False)

                ruta_debug = resultado_ocr.get("ruta_debug_segmentacion")
                if ruta_debug:
                    st.image(ruta_debug, caption="Debug segmentaciÃ³n: verde aceptado, rojo rechazado", use_container_width=False)

                motivos_rechazo = resultado_ocr.get("motivos_rechazo") or {}
                if motivos_rechazo:
                    st.caption("Motivos de rechazo")
                    st.json(motivos_rechazo)

                caracteres = resultado_ocr.get("caracteres", [])
                if caracteres:
                    st.subheader("Caracteres aceptados")
                    columnas = st.columns(min(len(caracteres), 8))
                    for idx, caracter in enumerate(caracteres[:16]):
                        columnas[idx % len(columnas)].image(caracter["ruta_caracter"], caption=f"Char {idx + 1}", use_container_width=True)
                else:
                    st.warning("No se segmentaron caracteres con los filtros actuales.")

                predicciones = resultado_ocr.get("predicciones_caracteres") or []
                diagnostico_caracteres = resultado_ocr.get("diagnostico", {}).get("diagnostico_caracteres") or []
                if predicciones:
                    st.subheader("Predicciones por carÃ¡cter")
                    for pred in predicciones:
                        indice_pred = int(pred.get("indice", 1))
                        esperado_pred = ""
                        if indice_pred - 1 < len(diagnostico_caracteres):
                            esperado_pred = diagnostico_caracteres[indice_pred - 1].get("etiqueta_esperada", "")
                        col_img, col_norm, col_info = st.columns([0.16, 0.16, 0.68])
                        if pred.get("ruta_caracter"):
                            col_img.image(pred["ruta_caracter"], caption=f"Original #{pred.get('indice')}", use_container_width=True)
                        if pred.get("ruta_debug_normalizada"):
                            col_norm.image(pred["ruta_debug_normalizada"], caption="CNN 32x32", use_container_width=True)
                        top3 = ", ".join(
                            f"{item['caracter']} ({item['confianza']:.2f})"
                            for item in pred.get("top3_predicciones", [])
                        )
                        col_info.write(
                            {
                                "indice": pred.get("indice"),
                                "esperado": esperado_pred,
                                "prediccion": pred.get("caracter_predicho"),
                                "confianza": round(float(pred.get("confianza", 0.0)), 4),
                                "top3": top3,
                            }
                        )

                if diagnostico_caracteres:
                    st.subheader("Comparacion caracter por caracter")
                    filas_diag = []
                    for item in diagnostico_caracteres:
                        filas_diag.append(
                            {
                                "indice": item.get("indice"),
                                "esperado": item.get("etiqueta_esperada"),
                                "crudo": item.get("caracter_crudo"),
                                "postprocesado": item.get("caracter_postprocesado"),
                                "prediccion": item.get("prediccion"),
                                "confianza": round(float(item.get("confianza", 0.0)), 4),
                                "estado": item.get("estado"),
                                "causa": item.get("causa_probable"),
                                "observacion": item.get("observacion"),
                            }
                        )
                    st.dataframe(pd.DataFrame(filas_diag), use_container_width=True)

                cambios_post = resultado_ocr.get("cambios_postprocesamiento") or []
                if cambios_post:
                    st.subheader("Cambios de postprocesamiento por formato")
                    st.dataframe(pd.DataFrame(cambios_post), use_container_width=True)

                st.caption(f"Reporte JSON: {reporte_ocr['ruta_json']}")
                st.caption(f"Reporte CSV: {reporte_ocr['ruta_csv']}")
                st.caption(f"Diagnostico JSON: {reporte_diagnostico['ruta_json']}")
                st.caption(f"Diagnostico CSV: {reporte_diagnostico['ruta_csv']}")

    with st.expander("Modo desarrollo / pruebas internas", expanded=False):
        placa_manual = st.text_input("Placa manual", value=config["ocr"]["manual_test_plate"])
        tiempo_manual = st.number_input(
            "Tiempo manual entre lineas virtuales (s)",
            min_value=0.1,
            value=float(config["speed"]["default_time_seconds"]),
            step=0.1,
        )
        distancia = st.number_input(
            "Distancia para prueba (m)",
            min_value=0.1,
            value=float(config["speed"]["default_distance_meters"]),
            step=0.5,
            key="distancia_prueba",
        )
        limite = st.number_input(
            "Limite para prueba (km/h)",
            min_value=1.0,
            value=float(config["speed"]["campus_speed_limit_kmh"]),
            step=1.0,
            key="limite_prueba",
        )
        ejecutar = st.button("Ejecutar pipeline simulado", type="primary")

    if ejecutar:
        if not ruta_imagen:
            st.warning("Carga una imagen en 'Pruebas con imagen' antes de ejecutar.")
            return

        config["speed"]["default_distance_meters"] = distancia
        config["speed"]["campus_speed_limit_kmh"] = limite
        resultado = procesar_imagen_prueba(
            ruta_imagen,
            config,
            modo_ocr="manual_controlado",
            placa_manual=placa_manual,
            tiempo_segundos=tiempo_manual,
        )
        mostrar_resultado_prueba(resultado)


def pestana_evidencias(config: dict) -> None:
    st.header("Evidencias")
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    archivos = sorted(reports_dir.glob("*.json"), key=lambda ruta: ruta.stat().st_mtime, reverse=True)

    for titulo, subdir in [
        ("Monitoreo de video", "monitoreo_video"),
        ("Monitoreo de camara", "monitoreo_camara"),
    ]:
        monitoreo_dir = reports_dir / subdir
        frames = sorted(monitoreo_dir.glob("*.jpg")) if monitoreo_dir.exists() else []
        if frames:
            st.subheader(titulo)
            cols = st.columns(min(len(frames), 2))
            for idx, frame in enumerate(frames[:2]):
                cols[idx].caption(frame.name)
                cols[idx].image(str(frame), use_container_width=True)

    st.subheader("Notificaciones por correo (copia local)")
    st.caption("Cada correo enviado o generado en modo simulado guarda una copia aqui.")
    notificaciones_dir = reports_dir / "notificaciones"
    notificaciones = sorted(notificaciones_dir.glob("*.txt"), key=lambda ruta: ruta.stat().st_mtime, reverse=True) if notificaciones_dir.exists() else []
    if notificaciones:
        datos_notificaciones = [
            {
                "archivo": ruta.name,
                "fecha_modificacion": datetime.fromtimestamp(ruta.stat().st_mtime).isoformat(timespec="seconds"),
                "ruta": str(ruta),
            }
            for ruta in notificaciones[:20]
        ]
        st.dataframe(pd.DataFrame(datos_notificaciones), use_container_width=True, hide_index=True)
        seleccionado_notificacion = st.selectbox(
            "Ver notificacion",
            notificaciones,
            format_func=lambda ruta: ruta.name,
        )
        st.code(seleccionado_notificacion.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Aun no existen notificaciones guardadas.")

    st.subheader("Reportes JSON")
    if not archivos:
        st.info("Aun no existen reportes JSON guardados.")
        return

    seleccionado = st.selectbox("Evidencia", archivos, format_func=lambda ruta: ruta.name)
    st.code(seleccionado.read_text(encoding="utf-8"), language="json")


def pestana_configuracion(config: dict) -> None:
    st.header("Configuracion")
    st.info(
        "No se utilizan motores OCR externos. El reconocimiento se realiza con segmentacion OpenCV "
        "y una CNN propia entrenada desde cero para clasificar caracteres."
    )
    st.json(config)
    st.caption("Los cambios persistentes se realizan editando config.yaml.")


def main() -> None:
    config = cargar_config()

    st.title("Fotorradar Ecuador IA")
    st.caption("Consola de monitoreo por video para placas ecuatorianas.")

    tabs = st.tabs(["Monitoreo", "Pruebas", "Evidencias", "Configuracion"])
    with tabs[0]:
        pestana_monitoreo(config)
    with tabs[1]:
        pestana_pruebas(config)
    with tabs[2]:
        pestana_evidencias(config)
    with tabs[3]:
        pestana_configuracion(config)


if __name__ == "__main__":
    main()


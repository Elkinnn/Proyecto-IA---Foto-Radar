from datetime import datetime
from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd, listar_eventos, listar_vehiculos
from src.pipeline import (
    generar_video_demo_anotado,
    procesar_camara_monitoreo,
    procesar_frame_video_monitoreo,
    procesar_imagen_prueba,
    procesar_video_monitoreo,
)
from src.fuzzy_system import clasificar_velocidad
from src.notifier import generar_notificacion_simulada, guardar_notificacion_simulada
from src.plate_detector import PlateDetector
from src.plate_reader import (
    leer_placa_desde_recorte,
    preprocesar_placa,
    registrar_diagnostico_recorte,
    registrar_reporte_ocr,
    segmentar_caracteres_v2,
)
from src.speed_estimator import SpeedTracker
from src.utils import cargar_config, guardar_archivo_subido
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
    "Demo fluido": "demo_fluido",
}


def _config_modo_rendimiento(config: dict, modo: str) -> dict:
    clave = PERFORMANCE_MODE_LABELS.get(modo, "balanceado")
    defaults = {
        "rapido": {"yolo_every_n_frames": 10, "inference_size": 416, "render_every_n_frames": 3, "history_max": 10},
        "balanceado": {"yolo_every_n_frames": 5, "inference_size": 640, "render_every_n_frames": 2, "history_max": 15},
        "preciso": {"yolo_every_n_frames": 3, "inference_size": 640, "render_every_n_frames": 1, "history_max": 20},
        "demo_fluido": {"yolo_every_n_frames": 15, "inference_size": 416, "render_every_n_frames": 1, "history_max": 5, "max_display_fps": 24},
    }
    salida = defaults[clave].copy()
    salida.update((config.get("monitoring_performance") or {}).get(clave, {}))
    return salida


st.set_page_config(
    page_title="Fotorradar Ecuador IA",
    page_icon=":vertical_traffic_light:",
    layout="wide",
)


def _guardar_evento_velocidad(
    placa: str,
    velocidad_kmh: float,
    limite_kmh: float,
    difuso: dict,
    vehiculo: dict | None,
    evidencia_frame: str | None,
    evidencia_placa: str | None,
    fuente: str,
    ruta_bd: str,
) -> tuple[int, dict]:
    vehiculo = vehiculo or {}
    evento = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "placa": placa,
        "vehiculo": vehiculo,
        "velocidad_kmh": velocidad_kmh,
        "limite_kmh": limite_kmh,
        "estado_difuso": difuso["estado"],
        "nivel_infraccion": difuso["nivel_infraccion"],
        "sancion": difuso["sancion"],
        "horas_suspension": difuso["horas_suspension"],
        "mensaje": difuso["mensaje"],
        "evidencia_frame": evidencia_frame,
        "evidencia_placa": evidencia_placa,
        "fuente": fuente,
    }
    evento_id = guardar_evento(evento, ruta_bd)
    return evento_id, evento


def _generar_notificacion_para_evento(evento_id: int, evento: dict, vehiculo: dict | None, difuso: dict) -> dict | None:
    notificacion = generar_notificacion_simulada(evento, vehiculo, difuso)
    if not notificacion:
        return None
    rutas = guardar_notificacion_simulada(notificacion, evento_id)
    notificacion.update(rutas)
    return notificacion


def panel_resultados(evento: dict | None) -> None:
    st.subheader("Panel de resultados")

    if not evento:
        col1, col2, col3 = st.columns(3)
        col1.metric("Placa detectada", "Pendiente")
        col2.metric("Velocidad", "Pendiente")
        col3.metric("Estado", "Sin evento")
        st.info("La deteccion de placa, reconocimiento de caracteres CNN, velocidad real y sanciones se integraran despues del flujo de video.")
        return

    vehiculo = evento.get("vehiculo") or {}
    clasificacion = evento.get("clasificacion_difusa") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Placa detectada", "Si" if evento.get("placa_detectada") else "No")
    col2.metric("Texto reconocido", evento.get("texto_placa") or "Pendiente")
    col3.metric("Velocidad km/h", f"{evento.get('velocidad_kmh', 0):.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Clasificacion", clasificacion.get("estado", "Pendiente"))
    col5.metric("Marca", vehiculo.get("marca", "Sin registro"))
    col6.metric("Color", vehiculo.get("color", "Sin registro"))

    col7, col8, col9 = st.columns(3)
    col7.metric("Propietario", vehiculo.get("propietario", "Sin registro"))
    col8.metric("Correo", vehiculo.get("correo", "Sin registro"))
    col9.metric("Sancion", "Si" if evento.get("sancion_generada") else "No")

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
    evento_id = resumen.get("evento_bd_id")
    notificacion = resumen.get("notificacion_simulada")

    estado_bd = "no_aplica"
    if evento_id:
        estado_bd = "completado" if resumen.get("vehiculo_encontrado") else "advertencia"
    elif velocidad_kmh is not None:
        estado_bd = "pendiente"

    estados = [
        ("DetecciÃ³n de placa", "completado" if placa_detectada else "pendiente"),
        ("Cruce LÃ­nea 1", "completado" if frame_linea_1 else "pendiente"),
        ("Cruce LÃ­nea 2", "completado" if frame_linea_2 else "pendiente"),
        ("Velocidad calculada", "completado" if velocidad_kmh is not None else "pendiente"),
        ("ClasificaciÃ³n difusa", "completado" if difuso else ("pendiente" if velocidad_kmh is not None else "no_aplica")),
        ("Consulta en base de datos", estado_bd),
        (
            "NotificaciÃ³n simulada",
            "completado"
            if notificacion
            else ("no_aplica" if difuso and difuso.get("nivel_infraccion") == "Sin infracciÃ³n" else ("pendiente" if difuso else "no_aplica")),
        ),
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
    ruta_bd: str,
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
        "tiempo_segundos": resumen.get("tiempo_entre_lineas"),
        "velocidad_kmh": resumen.get("velocidad_kmh"),
        "estado": resumen.get("estado"),
        "ruta_frame_cruce_linea_1": str(rutas["linea_1"]) if guardo_linea_1 else None,
        "ruta_frame_cruce_linea_2": str(rutas["linea_2"]) if guardo_linea_2 else None,
        "ruta_frame_velocidad_calculada": str(rutas["calculada"]) if guardo_calculada else None,
    }
    if resumen.get("velocidad_kmh") is not None:
        difuso = clasificar_velocidad(resumen["velocidad_kmh"], limite_kmh)
        vehiculo = buscar_vehiculo_por_placa(placa_manual, ruta_bd)
        evento_id, evento = _guardar_evento_velocidad(
            placa=placa_manual,
            velocidad_kmh=resumen["velocidad_kmh"],
            limite_kmh=limite_kmh,
            difuso=difuso,
            vehiculo=vehiculo,
            evidencia_frame=str(rutas["calculada"]) if guardo_calculada else None,
            evidencia_placa=None,
            fuente="SimulaciÃ³n de velocidad",
            ruta_bd=ruta_bd,
        )
        notificacion = _generar_notificacion_para_evento(evento_id, evento, vehiculo, difuso)
        datos.update(
            {
                "datos_vehiculo": vehiculo,
                "evento_id": evento_id,
                "velocidad": resumen["velocidad_kmh"],
                "resultado_difuso": difuso,
                "notificacion_simulada": notificacion,
                "estado_difuso": difuso["estado"],
                "nivel_infraccion": difuso["nivel_infraccion"],
                "sancion": difuso["sancion"],
                "horas_suspension": difuso["horas_suspension"],
                "mensaje_difuso": difuso["mensaje"],
                "grados_pertenencia": difuso["grados"],
            }
        )
    else:
        difuso = None
        vehiculo = None
        evento_id = None
        notificacion = None

    with open(rutas["json"], "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)

    return {
        "resumen": resumen,
        "difuso": difuso,
        "vehiculo": vehiculo,
        "evento_id": evento_id,
        "notificacion": notificacion,
        "datos": datos,
        "frame_final_rgb": cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2RGB) if ultimo_frame is not None else None,
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
    }
    col_vel1, col_vel2, col_vel3 = st.columns(3)
    col_vel1.metric("Estado de velocidad", etiquetas_estado.get(estado_velocidad, estado_velocidad))
    col_vel2.metric("Frame cruce linea 1", velocidad.get("frame_cruce_linea_1") or "Pendiente")
    col_vel3.metric("Frame cruce linea 2", velocidad.get("frame_cruce_linea_2") or "Pendiente")

    col_vel4, col_vel5, col_vel6 = st.columns(3)
    tiempo_entre = velocidad.get("tiempo_entre_lineas")
    velocidad_kmh = velocidad.get("velocidad_kmh")
    col_vel4.metric("Tiempo entre lineas", f"{tiempo_entre:.3f} s" if tiempo_entre is not None else "Pendiente")
    col_vel5.metric("Distancia configurada", f"{velocidad.get('distancia_metros', resumen.get('distancia_lineas_m', 0)):.1f} m")
    col_vel6.metric("Velocidad estimada", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "Pendiente")

    col_vel7, col_vel8, col_vel9 = st.columns(3)
    col_vel7.metric("Posicion Linea 1", f"{resumen.get('posicion_linea_1', 0.45):.2f}")
    col_vel8.metric("Posicion Linea 2", f"{resumen.get('posicion_linea_2', 0.65):.2f}")
    col_vel9.metric("Estado de cruce", etiquetas_estado.get(estado_velocidad, estado_velocidad))

    _mostrar_estado_flujo(resumen, velocidad, velocidad_kmh)

    st.subheader("ClasificaciÃ³n difusa")
    if velocidad_kmh is not None:
        difuso = resumen.get("clasificacion_difusa") or clasificar_velocidad(velocidad_kmh, resumen.get("limite_velocidad_kmh", 30.0))
        col_dif1, col_dif2, col_dif3 = st.columns(3)
        col_dif1.metric("Estado difuso", difuso["estado"])
        col_dif2.metric("Nivel de infracciÃ³n", difuso["nivel_infraccion"])
        col_dif3.metric("Horas de suspensiÃ³n", difuso["horas_suspension"])

        col_dif4, col_dif5 = st.columns(2)
        col_dif4.metric("SanciÃ³n", difuso["sancion"])
        col_dif5.metric("LÃ­mite evaluado", f"{difuso['limite_kmh']:.1f} km/h")
        st.info(difuso["mensaje"])
    else:
        col_dif1, col_dif2, col_dif3 = st.columns(3)
        col_dif1.metric("Estado difuso", "Pendiente")
        col_dif2.metric("Nivel de infracciÃ³n", "Pendiente")
        col_dif3.metric("SanciÃ³n", "Pendiente")

    if resumen.get("placa_controlada") or resumen.get("evento_bd_id"):
        st.subheader("Resultado del evento")
        vehiculo = resumen.get("vehiculo") or {}
        difuso_evento = resumen.get("clasificacion_difusa") or {}
        notificacion = resumen.get("notificacion_simulada")
        vehiculo_encontrado = "SÃ­" if resumen.get("vehiculo_encontrado") else "No"

        if resumen.get("vehiculo_encontrado") is False:
            st.warning("Placa no encontrada en la base de datos.")

        col_evt_res1, col_evt_res2, col_evt_res3 = st.columns(3)
        col_evt_res1.metric("Placa usada", resumen.get("placa_controlada", "Pendiente"))
        col_evt_res2.metric("VehÃ­culo encontrado", vehiculo_encontrado)
        col_evt_res3.metric("Evento ID", resumen.get("evento_bd_id", "Pendiente"))

        col_evt_res4, col_evt_res5, col_evt_res6 = st.columns(3)
        col_evt_res4.metric("Marca", vehiculo.get("marca", "No registrado"))
        col_evt_res5.metric("Modelo", vehiculo.get("modelo", "No registrado"))
        col_evt_res6.metric("Color", vehiculo.get("color", "No registrado"))

        col_evt_res7, col_evt_res8, col_evt_res9 = st.columns(3)
        col_evt_res7.metric("Propietario", vehiculo.get("propietario", "No registrado"))
        col_evt_res8.metric("Correo", vehiculo.get("correo", "No registrado"))
        col_evt_res9.metric("Velocidad calculada", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "Pendiente")

        col_evt_res10, col_evt_res11, col_evt_res12 = st.columns(3)
        col_evt_res10.metric("LÃ­mite de velocidad", f"{resumen.get('limite_velocidad_kmh', 0):.1f} km/h")
        col_evt_res11.metric("Estado difuso", difuso_evento.get("estado", "Pendiente"))
        col_evt_res12.metric("Nivel de infracciÃ³n", difuso_evento.get("nivel_infraccion", "Pendiente"))

        col_evt_res13, col_evt_res14, col_evt_res15 = st.columns(3)
        col_evt_res13.metric("SanciÃ³n", difuso_evento.get("sancion", "Pendiente"))
        col_evt_res14.metric("Horas de suspensiÃ³n", difuso_evento.get("horas_suspension", "Pendiente"))
        col_evt_res15.metric("NotificaciÃ³n generada", "SÃ­" if notificacion else "No")

        if notificacion:
            col_not1, col_not2 = st.columns(2)
            col_not1.metric("Destinatario", notificacion.get("destinatario") or "Sin correo")
            col_not2.metric("Asunto", notificacion.get("asunto") or "Pendiente")
            col_not3, col_not4 = st.columns(2)
            col_not3.metric("Ruta TXT notificaciÃ³n", notificacion.get("ruta_txt") or "Pendiente")
            col_not4.metric("Ruta JSON notificaciÃ³n", notificacion.get("ruta_json") or "Pendiente")
        elif resumen.get("evento_bd_id"):
            st.info("No se generÃ³ notificaciÃ³n porque no existe infracciÃ³n.")

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


def registrar_evento_monitoreo_si_corresponde(resumen: dict, placa_controlada: str, ruta_bd: str) -> dict:
    velocidad = resumen.get("velocidad") or {}
    velocidad_kmh = velocidad.get("velocidad_kmh")
    if velocidad_kmh is None:
        return resumen

    frame_linea_2 = velocidad.get("frame_cruce_linea_2")
    clave_evento = f"{placa_controlada}-{resumen.get('fuente')}-{frame_linea_2}-{velocidad_kmh:.3f}"
    if st.session_state.get("evento_velocidad_guardado_clave") == clave_evento:
        resumen.update(st.session_state.get("evento_monitoreo_actual") or {})
        return resumen

    limite_kmh = float(resumen.get("limite_velocidad_kmh", 30.0))
    difuso = clasificar_velocidad(velocidad_kmh, limite_kmh)
    vehiculo = buscar_vehiculo_por_placa(placa_controlada, ruta_bd)
    evento_id, evento = _guardar_evento_velocidad(
        placa=placa_controlada,
        velocidad_kmh=velocidad_kmh,
        limite_kmh=limite_kmh,
        difuso=difuso,
        vehiculo=vehiculo,
        evidencia_frame=resumen.get("ultimo_frame_deteccion") or resumen.get("ultimo_frame_evidencia"),
        evidencia_placa=resumen.get("ultimo_recorte_placa") or resumen.get("ruta_mejor_recorte_evento"),
        fuente=resumen.get("fuente", "Monitoreo"),
        ruta_bd=ruta_bd,
    )
    notificacion = _generar_notificacion_para_evento(evento_id, evento, vehiculo, difuso)

    resumen["placa_controlada"] = placa_controlada
    resumen["vehiculo"] = vehiculo
    resumen["vehiculo_encontrado"] = vehiculo is not None
    resumen["clasificacion_difusa"] = difuso
    resumen["evento_bd_id"] = evento_id
    resumen["notificacion_simulada"] = notificacion
    st.session_state.evento_monitoreo_actual = {
        "placa_controlada": placa_controlada,
        "vehiculo": vehiculo,
        "vehiculo_encontrado": vehiculo is not None,
        "clasificacion_difusa": difuso,
        "evento_bd_id": evento_id,
        "notificacion_simulada": notificacion,
    }
    st.session_state.evento_velocidad_guardado_clave = clave_evento
    return resumen


def _enriquecer_resumen_monitoreo_con_ocr(resumen: dict) -> dict:
    selector_activo = "ultimos_candidatos_recorte" in resumen or "mejor_recorte_placa_info" in resumen
    ruta_recorte = resumen.get("mejor_recorte_placa") or resumen.get("ruta_mejor_recorte_evento")
    if not selector_activo:
        ruta_recorte = ruta_recorte or resumen.get("ultimo_recorte_placa")
    if not ruta_recorte:
        resumen.setdefault("texto_ocr_crudo", "Pendiente")
        resumen.setdefault("texto_ocr_corregido", "Pendiente")
        resumen.setdefault("confianza_ocr", None)
        resumen.setdefault("formato_ocr_valido", False)
        return resumen

    if st.session_state.get("ultimo_recorte_ocr_procesado") == ruta_recorte:
        resumen.update(st.session_state.get("ultimo_resultado_ocr_monitoreo") or {})
        return resumen

    try:
        inicio_lector = time.perf_counter()
        resultado_ocr = leer_placa_desde_recorte(str(ruta_recorte), placa_esperada="")
        tiempo_lector_ms = (time.perf_counter() - inicio_lector) * 1000
        datos_ocr = {
            "texto_ocr_crudo": resultado_ocr.get("texto_detectado_crudo") or "Pendiente",
            "texto_ocr_corregido": resultado_ocr.get("texto_postprocesado") or "Pendiente",
            "confianza_ocr": resultado_ocr.get("confianza_promedio"),
            "formato_ocr_valido": bool((resultado_ocr.get("formato") or {}).get("valido")),
            "estado_ocr": resultado_ocr.get("estado"),
            "mensaje_ocr": resultado_ocr.get("mensaje"),
            "causa_probable_ocr": resultado_ocr.get("causa_probable"),
            "ruta_placa_preprocesada_ocr": (resultado_ocr.get("preprocesamiento") or {}).get("ruta_imagen_procesada"),
            "ruta_debug_segmentacion_ocr": resultado_ocr.get("ruta_debug_segmentacion"),
            "ruta_banda_ocr": resultado_ocr.get("ruta_banda"),
            "caracteres_segmentados_ocr": resultado_ocr.get("caracteres", []),
            "cantidad_caracteres_segmentados_ocr": resultado_ocr.get("cantidad_caracteres_segmentados", 0),
            "tiempo_lector_cnn_ms": round(tiempo_lector_ms, 3),
        }
        st.session_state.contador_lector_cnn_monitoreo = int(st.session_state.get("contador_lector_cnn_monitoreo", 0) or 0) + 1
        _registrar_metricas_ocr_mejor_recorte(resumen, resultado_ocr)
    except Exception as exc:
        datos_ocr = {
            "texto_ocr_crudo": "Error lector CNN",
            "texto_ocr_corregido": "Error lector CNN",
            "confianza_ocr": None,
            "formato_ocr_valido": False,
            "estado_ocr": "error",
            "mensaje_ocr": str(exc),
        }

    st.session_state.ultimo_recorte_ocr_procesado = ruta_recorte
    st.session_state.ultimo_resultado_ocr_monitoreo = datos_ocr
    resumen.update(datos_ocr)
    return resumen


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
    texto_ocr = (resumen.get("texto_ocr_corregido") or "").strip().upper()
    if resumen.get("formato_ocr_valido") and texto_ocr and texto_ocr != "PENDIENTE":
        return texto_ocr
    return (placa_controlada or "PBC1234").strip().upper()


def _accion_monitoreo(resumen: dict) -> str:
    difuso = resumen.get("clasificacion_difusa") or {}
    if resumen.get("notificacion_simulada"):
        return "Notificacion simulada"
    if difuso.get("nivel_infraccion") == "Sin infraccion":
        return "Sin infraccion"
    if resumen.get("evento_bd_id"):
        return "Evento guardado"
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
        ruta_historial,
        velocidad.get("frame_cruce_linea_2"),
        resumen.get("evento_bd_id"),
        resumen.get("frames_procesados") or resumen.get("frame_actual"),
    )
    if st.session_state.get("ultima_clave_historial_monitoreo") == clave:
        return
    if not clave[0] and velocidad.get("velocidad_kmh") is None and not resumen.get("evento_bd_id"):
        return

    fila = {
        "hora": datetime.now().strftime("%H:%M:%S"),
        "placa detectada": resumen.get("texto_ocr_crudo") or "Pendiente",
        "placa corregida": resumen.get("texto_ocr_corregido") or resumen.get("placa_controlada") or "Pendiente",
        "confianza YOLO": (
            f"{float(ultima_deteccion.get('confianza')):.2f}"
            if ultima_deteccion.get("confianza") is not None
            else "Pendiente"
        ),
        "confianza CNN caracteres": (
            f"{float(resumen.get('confianza_ocr')):.2f}"
            if resumen.get("confianza_ocr") is not None
            else "Pendiente"
        ),
        "velocidad": (
            f"{velocidad.get('velocidad_kmh'):.2f} km/h"
            if velocidad.get("velocidad_kmh") is not None
            else "Pendiente"
        ),
        "estado": _accion_monitoreo(resumen),
        "encontrada en BD": "Si" if resumen.get("vehiculo_encontrado") else "No",
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
        "lector CNN": resumen.get("texto_ocr_corregido") or resumen.get("texto_ocr_crudo") or "Pendiente",
        "confianza_yolo": (resumen.get("ultima_deteccion") or {}).get("confianza"),
        "confianza_ocr": resumen.get("confianza_ocr"),
    }
    recortes = st.session_state.get("recortes_placas_monitoreo", [])
    if not any(actual.get("ruta") == clave for actual in recortes):
        recortes.insert(0, item)
    max_historial = int(st.session_state.get("historial_maximo_monitoreo", 15) or 15)
    st.session_state.recortes_placas_monitoreo = recortes[:max_historial]
    st.session_state.ultima_clave_recorte_monitoreo = clave


def mostrar_panel_monitoreo_limpio(resumen: dict, config: dict, ejecutar_lector_cnn: bool = True) -> None:
    if ejecutar_lector_cnn:
        resumen = _enriquecer_resumen_monitoreo_con_ocr(resumen)
    else:
        resumen.setdefault("texto_ocr_crudo", "Analizando placa..." if resumen.get("mejor_recorte_placa") else "Pendiente")
        resumen.setdefault("texto_ocr_corregido", "Pendiente")
        resumen.setdefault("confianza_ocr", None)
        resumen.setdefault("formato_ocr_valido", False)
    _actualizar_historial_monitoreo(resumen)
    _actualizar_recortes_monitoreo(resumen)

    velocidad = resumen.get("velocidad") or {}
    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    difuso = resumen.get("clasificacion_difusa") or {}
    vehiculo = resumen.get("vehiculo") or {}
    confianza_yolo = ultima_deteccion.get("confianza") or resumen.get("mejor_confianza_evento")
    confianza_ocr = resumen.get("confianza_ocr")
    velocidad_kmh = velocidad.get("velocidad_kmh")
    ruta_recorte_actual = resumen.get("ultimo_recorte_placa")
    ruta_mejor_recorte = resumen.get("mejor_recorte_placa") or resumen.get("ruta_mejor_recorte_evento")
    mejor_info = resumen.get("mejor_recorte_placa_info") or {}
    ruta_recorte = ruta_mejor_recorte

    st.subheader("Ultima placa detectada")
    col_recorte, col_info = st.columns([0.35, 0.65])
    with col_recorte:
        if ruta_mejor_recorte:
            st.image(ruta_mejor_recorte, caption="Mejor recorte para lector CNN", use_container_width=True)
            if ruta_recorte_actual and ruta_recorte_actual != ruta_mejor_recorte:
                st.image(ruta_recorte_actual, caption="Recorte actual detectado", use_container_width=True)
        elif ruta_recorte_actual:
            st.image(ruta_recorte_actual, caption="Recorte actual detectado (esperando mejor recorte)", use_container_width=True)
        else:
            st.info("Esperando deteccion de placa.")

    with col_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Texto reconocido crudo", resumen.get("texto_ocr_crudo", "Pendiente"))
        c2.metric("Texto corregido por formato", resumen.get("texto_ocr_corregido", "Pendiente"))
        c3.metric("Confianza CNN caracteres", f"{confianza_ocr:.2f}" if confianza_ocr is not None else "Pendiente")

        c4, c5, c6 = st.columns(3)
        c4.metric("Confianza YOLO", f"{float(confianza_yolo):.2f}" if confianza_yolo is not None else "Pendiente")
        c5.metric("Velocidad", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "Pendiente")
        c6.metric("Estado difuso", difuso.get("estado", "Pendiente"))

        c_best1, c_best2, c_best3 = st.columns(3)
        c_best1.metric("Puntaje recorte", f"{mejor_info.get('puntaje_total'):.3f}" if mejor_info.get("puntaje_total") is not None else "Pendiente")
        c_best2.metric("Nitidez", f"{mejor_info.get('nitidez'):.1f}" if mejor_info.get("nitidez") is not None else "Pendiente")
        c_best3.metric("Aspect ratio", f"{mejor_info.get('aspect_ratio'):.2f}" if mejor_info.get("aspect_ratio") is not None else "Pendiente")

        c7, c8, c9 = st.columns(3)
        c7.metric("Encontrada en BD", "Si" if resumen.get("vehiculo_encontrado") else "No")
        c8.metric("Fecha/hora", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        c9.metric("Accion", _accion_monitoreo(resumen))

        c10, c11, c12 = st.columns(3)
        c10.metric("Caracteres CNN", resumen.get("cantidad_caracteres_segmentados_ocr", "Pendiente"))
        c11.metric("Formato", "Valido" if resumen.get("formato_ocr_valido") else "Pendiente")
        c12.metric("Causa probable", resumen.get("causa_probable_ocr") or "Pendiente")

        if resumen.get("vehiculo_encontrado"):
            st.caption(
                f"{vehiculo.get('marca', 'Vehiculo')} {vehiculo.get('modelo', '')} | "
                f"{vehiculo.get('color', 'Sin color')} | {vehiculo.get('propietario', 'Sin propietario')}"
            )

    st.subheader("Historial reciente")
    historial = st.session_state.get("historial_detecciones_monitoreo", [])
    if historial:
        st.dataframe(pd.DataFrame(historial), use_container_width=True, hide_index=True)
    else:
        st.caption("Aun no hay detecciones registradas en esta sesion.")

    recortes = st.session_state.get("recortes_placas_monitoreo", [])
    if recortes:
        st.subheader("Placas capturadas")
        columnas = st.columns(min(4, len(recortes)))
        for idx, recorte in enumerate(recortes[:8]):
            with columnas[idx % len(columnas)]:
                st.image(recorte["ruta"], use_container_width=True)
                conf_yolo = recorte.get("confianza_yolo")
                conf_ocr = recorte.get("confianza_ocr")
                texto_yolo = f"{conf_yolo:.2f}" if conf_yolo is not None else "Pendiente"
                texto_ocr = f"{conf_ocr:.2f}" if conf_ocr is not None else "Pendiente"
                st.caption(f"{recorte['hora']} | {recorte['lector CNN']} | YOLO {texto_yolo} | CNN {texto_ocr}")

    with st.expander("Debug mejor recorte", expanded=False):
        candidatos = resumen.get("ultimos_candidatos_recorte") or []
        if mejor_info:
            st.write(
                {
                    "motivo": mejor_info.get("motivo"),
                    "frame_elegido": mejor_info.get("frame_index"),
                    "puntaje_total": mejor_info.get("puntaje_total"),
                    "conf_yolo": mejor_info.get("conf_yolo"),
                    "nitidez": mejor_info.get("nitidez"),
                    "area_relativa": mejor_info.get("area_relativa"),
                    "aspect_ratio": mejor_info.get("aspect_ratio"),
                    "ruta_recorte": mejor_info.get("ruta_recorte"),
                    "ruta_frame_bbox": mejor_info.get("ruta_frame_bbox"),
                }
            )
        if candidatos:
            st.dataframe(pd.DataFrame(candidatos), use_container_width=True, hide_index=True)
        else:
            st.caption("Aun no hay suficientes candidatos para seleccionar un mejor recorte.")

        ruta_pre = resumen.get("ruta_placa_preprocesada_ocr")
        ruta_banda = resumen.get("ruta_banda_ocr")
        ruta_debug_seg = resumen.get("ruta_debug_segmentacion_ocr")
        imgs = [ruta for ruta in [ruta_pre, ruta_banda, ruta_debug_seg] if ruta]
        if imgs:
            st.caption("Preprocesamiento y segmentacion para lector CNN")
            cols = st.columns(min(3, len(imgs)))
            for idx, ruta in enumerate(imgs):
                with cols[idx % len(cols)]:
                    st.image(ruta, use_container_width=True)
        caracteres = resumen.get("caracteres_segmentados_ocr") or []
        if caracteres:
            st.caption("Caracteres segmentados")
            cols = st.columns(min(7, len(caracteres)))
            for idx, caracter in enumerate(caracteres[:7]):
                ruta_char = caracter.get("ruta_caracter")
                if ruta_char:
                    with cols[idx % len(cols)]:
                        st.image(ruta_char, use_container_width=True)
                        st.caption(str(idx + 1))

    with st.expander("Debug avanzado", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Modelo YOLO", config["models"].get("plate_detector_model", config["models"].get("plate_detector_path", "No configurado")))
        d2.metric("Frames procesados", resumen.get("frames_procesados", resumen.get("frame_actual", 0)))
        d3.metric("Detecciones validas", resumen.get("detecciones_validas", 0))

        d4, d5, d6 = st.columns(3)
        d4.metric("BBox", str(ultima_deteccion.get("bbox") or "Pendiente"))
        d5.metric("Frecuencia", resumen.get("frecuencia_deteccion", "Pendiente"))
        d6.metric("Resolucion", f"{resumen.get('ancho', 0)} x {resumen.get('alto', 0)}")

        d7, d8, d9 = st.columns(3)
        d7.metric("FPS procesamiento", resumen.get("fps_procesamiento", "Pendiente"))
        d8.metric("YOLO ms", resumen.get("tiempo_yolo_ms", "Pendiente"))
        d9.metric("Lector CNN ms", resumen.get("tiempo_lector_cnn_ms", "Pendiente"))

        d10, d11, d12 = st.columns(3)
        d10.metric("Frames saltados YOLO", resumen.get("frames_saltados", "Pendiente"))
        d11.metric("Resolucion inferencia", resumen.get("resolucion_inferencia", "Pendiente"))
        d12.metric("Flujo", resumen.get("modo_rendimiento", "Pendiente"))

        d13, d14, d15 = st.columns(3)
        d13.metric("Frames mostrados", resumen.get("frames_mostrados", "Pendiente"))
        d14.metric("Frames YOLO", resumen.get("frames_yolo_analizados", "Pendiente"))
        d15.metric("Ejecuciones lector CNN", st.session_state.get("contador_lector_cnn_monitoreo", 0))

        tiempos = resumen.get("tiempos_etapa") or {}
        if tiempos:
            st.caption("Tiempos por etapa (ms)")
            st.json(tiempos)

        st.write(
            {
                "character_reader_path": config["models"].get("character_reader_path"),
                "motivo_rechazo": resumen.get("motivos_rechazo") or "Ninguno",
                "primer_frame_evidencia": resumen.get("primer_frame_evidencia"),
                "ultimo_frame_evidencia": resumen.get("ultimo_frame_evidencia"),
                "ultimo_frame_deteccion": resumen.get("ultimo_frame_deteccion"),
                "ultimo_recorte_placa": ruta_recorte,
                "mensaje_detector": resumen.get("mensaje_detector"),
                "mensaje_lector_cnn": resumen.get("mensaje_ocr"),
                "parametros": {
                    "distancia_lineas_m": resumen.get("distancia_lineas_m"),
                    "limite_velocidad_kmh": resumen.get("limite_velocidad_kmh"),
                    "posicion_linea_1": resumen.get("posicion_linea_1"),
                    "posicion_linea_2": resumen.get("posicion_linea_2"),
                },
            }
        )


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
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def pestana_monitoreo(config: dict) -> None:
    st.header("Monitoreo")
    _inicializar_estado_monitoreo()

    fuente_col, visor_col = st.columns([0.28, 0.72])
    with fuente_col:
        fuente_monitoreo = st.selectbox("Fuente de monitoreo", ["Video de prueba", "Camara en vivo"])
        video = None
        indice_camara = 0

        if fuente_monitoreo == "Video de prueba":
            video = st.file_uploader("Cargar video de prueba", type=["mp4", "avi", "mov", "mkv"], key="video_monitoreo")
        else:
            indice_camara = st.number_input(
                "Selector de camara",
                min_value=0,
                value=0,
                step=1,
                help="0 normalmente corresponde a la camara principal. Si usa una camara externa o Iriun, pruebe 1 o 2.",
            )

        rotacion_ui = st.selectbox("Rotacion de imagen", ["Sin rotacion", "90 grados", "180 grados", "270 grados"])
        rotacion = {
            "Sin rotacion": "Sin rotacion",
            "90 grados": "Rotar 90 derecha",
            "180 grados": "Rotar 180",
            "270 grados": "Rotar 90 izquierda",
        }[rotacion_ui]

        modo_rendimiento = "Balanceado"
        config_rendimiento = _config_modo_rendimiento(config, modo_rendimiento)
        max_display_fps = int(config_rendimiento.get("max_display_fps", 0) or 0)

        if fuente_monitoreo == "Video de prueba":
            iniciar = st.button(
                "Reanudar monitoreo" if st.session_state.get("monitoreo_pausado") else "Iniciar monitoreo",
                type="primary",
                use_container_width=True,
            )
            detener = st.button("Detener / pausar", use_container_width=True)
            reiniciar = st.button("Reiniciar", use_container_width=True)
            with st.expander("Procesar video y generar evidencia", expanded=False):
                st.caption(
                    "Opcion secundaria: analiza el video completo sin mostrar cada frame en vivo y genera un MP4 anotado como evidencia."
                )
                analizar_video = st.button("Analizar video y generar evidencia", use_container_width=True)
        else:
            iniciar = st.button(
                "Reanudar monitoreo" if st.session_state.get("monitoreo_pausado") else "Iniciar monitoreo",
                type="primary",
                use_container_width=True,
            )
            detener = st.button("Detener / pausar", use_container_width=True)
            reiniciar = st.button("Reiniciar", use_container_width=True)
            analizar_video = False

        distancia_metros = float(config["speed"].get("default_distance_meters", 10.0))
        limite_velocidad = float(config["speed"].get("campus_speed_limit_kmh", 30.0))
        posicion_linea_1 = 0.45
        posicion_linea_2 = 0.65
        frecuencia_deteccion = int(config_rendimiento["yolo_every_n_frames"])
        inference_size = int(config_rendimiento["inference_size"])
        render_every_n_frames = int(config_rendimiento["render_every_n_frames"])
        st.session_state.historial_maximo_monitoreo = int(config_rendimiento["history_max"])
        conf_min = 0.45
        persistencia_frames = 3
        max_frames = 0
        velocidad_reproduccion = "Normal (1x)"
        placa_controlada = str(config.get("ocr", {}).get("manual_test_plate", "PBC1234"))

        with st.expander("Configuracion avanzada", expanded=False):
            distancia_metros = st.number_input(
                "Distancia real entre lineas (m)",
                min_value=0.1,
                value=distancia_metros,
                step=0.5,
                key="distancia_monitoreo_avanzada",
            )
            limite_velocidad = st.number_input(
                "Limite de velocidad (km/h)",
                min_value=1.0,
                value=limite_velocidad,
                step=1.0,
                key="limite_monitoreo_avanzada",
            )
            posicion_linea_1 = st.slider("Posicion Linea 1", 0.05, 0.95, posicion_linea_1, 0.01)
            posicion_linea_2 = st.slider("Posicion Linea 2", 0.05, 0.95, posicion_linea_2, 0.01)
            frecuencia_deteccion = st.slider("Frecuencia YOLO", 1, 60, frecuencia_deteccion)
            inference_size = st.select_slider("Resolucion inferencia YOLO", options=[320, 416, 512, 640, 768], value=inference_size)
            render_every_n_frames = st.slider("Actualizar video cada N frames", 1, 10, render_every_n_frames, 1)
            max_display_fps = st.slider("FPS maximo visual (0 = sin limite)", 0, 30, max_display_fps, 1)
            conf_min = st.slider("Confianza minima YOLO", 0.10, 0.90, conf_min, 0.05)
            persistencia_frames = st.slider("Persistencia de deteccion", 0, 30, persistencia_frames, 1)
            max_frames = st.number_input(
                "Frames maximos a procesar (0 = completo)",
                min_value=0,
                max_value=10000,
                value=max_frames,
                step=100,
            )
            placa_controlada = st.text_input(
                "Placa controlada temporal",
                value=placa_controlada,
                help="Se usa solo como respaldo para BD/fuzzy si el lector CNN automatico aun no entrega una placa valida.",
            )
            st.caption(f"Modelo YOLO: {config['models'].get('plate_detector_model', config['models'].get('plate_detector_path'))}")
            st.caption(f"Modelo lector CNN de caracteres: {config['models'].get('character_reader_path')}")
            if fuente_monitoreo == "Camara en vivo":
                st.caption(
                    f"Camara: YOLO cada {frecuencia_deteccion} frames | inferencia {inference_size}px"
                )

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
        st.session_state.contador_lector_cnn_monitoreo = 0

    if posicion_linea_2 <= posicion_linea_1:
        st.warning("La Linea 2 debe estar debajo de la Linea 1 para medir movimiento de arriba hacia abajo.")
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
            st.info("Seleccione video o camara en vivo y presione Iniciar monitoreo.")
        return

    if ejecutar_monitoreo and fuente_monitoreo == "Video de prueba" and not video and not st.session_state.get("ruta_video_monitoreo"):
        st.warning("Primero cargue un video de prueba.")
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
        if fuente_monitoreo == "Video de prueba":
            if not reanudar_video:
                st.session_state.ruta_video_monitoreo = guardar_archivo_subido(video, config["paths"]["input_dir"])

    ruta_modelo_placa = config["models"].get("plate_detector_model", config["models"].get("plate_detector_path"))

    if analizar_video:
        if fuente_monitoreo != "Video de prueba" or not video:
            st.warning("Cargue un video de prueba antes de analizar.")
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
                mostrar_panel_monitoreo_limpio(st.session_state.ultimo_resultado, config, ejecutar_lector_cnn=False)
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
        return

    def _actualizar_panel_desde_estado(estado_frame: dict) -> None:
        resumen_parcial = dict(estado_frame)
        resumen_parcial.setdefault("frames_procesados", estado_frame.get("frame_actual", 0))
        resumen_parcial.setdefault("fuente", fuente_monitoreo)
        resumen_parcial.setdefault("rotacion", rotacion)
        resumen_parcial.setdefault("modelo_detector_disponible", True)
        resumen_parcial.setdefault("placa_controlada", placa_controlada)
        resumen_parcial.setdefault("modo_rendimiento", modo_rendimiento)
        ejecutar_lector = modo_rendimiento != "Demo fluido"
        if ejecutar_lector:
            resumen_parcial = _enriquecer_resumen_monitoreo_con_ocr(resumen_parcial)
        with panel_placeholder.container():
            mostrar_panel_monitoreo_limpio(resumen_parcial, config, ejecutar_lector_cnn=False)

    def actualizar_frame(frame_rgb, numero_frame: int, estado_frame: dict | None = None) -> None:
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        st.session_state.ultima_imagen_procesada = frame_rgb
        if estado_frame:
            st.session_state.frame_actual = int(estado_frame.get("frame_actual", numero_frame) or 0)
            if estado_frame.get("estado_persistencia"):
                st.session_state.estado_persistencia = estado_frame["estado_persistencia"]
            confianza = estado_frame.get("ultima_confianza")
            texto_confianza = f"{confianza:.2f}" if confianza is not None else "Pendiente"
            estado_placeholder.info(
                f"Frame {estado_frame.get('frame_actual', numero_frame)} | "
                f"Estado placa: {estado_frame.get('estado_placa', 'Pendiente')} | "
                f"Confianza YOLO: {texto_confianza} | "
                f"FPS proc: {estado_frame.get('fps_procesamiento', 'Pendiente')}"
            )
            if estado_frame.get("ultimo_recorte_placa") or estado_frame.get("ruta_mejor_recorte_evento"):
                _actualizar_panel_desde_estado(estado_frame)
        else:
            estado_placeholder.info(f"Procesando frame {numero_frame}")

    def actualizar_progreso(valor: float) -> None:
        progreso.progress(valor)

    if not ejecutar_monitoreo:
        if st.session_state.get("ultima_imagen_procesada") is not None:
            frame_placeholder.image(st.session_state.ultima_imagen_procesada, channels="RGB", use_container_width=True)
            if st.session_state.get("monitoreo_pausado"):
                estado_placeholder.info(f"Video pausado en el frame {st.session_state.get('frame_actual', 0)}. Presione Reanudar monitoreo para continuar.")
        if st.session_state.get("ultimo_resultado"):
            with panel_placeholder.container():
                mostrar_panel_monitoreo_limpio(st.session_state.ultimo_resultado, config)
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
                plate_crop_selection=config.get("plate_crop_selection"),
                inference_size=int(inference_size),
                render_every_n_frames=int(render_every_n_frames),
                max_display_fps=int(max_display_fps),
                demo_fluido=modo_rendimiento == "Demo fluido",
                frame_callback=actualizar_frame,
                progreso_callback=actualizar_progreso,
                detener_callback=lambda: not st.session_state.get("monitoreo_activo", True),
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
                plate_crop_selection=config.get("plate_crop_selection"),
                inference_size=int(inference_size),
                render_every_n_frames=int(render_every_n_frames),
                max_display_fps=int(max_display_fps),
                demo_fluido=modo_rendimiento == "Demo fluido",
                frame_callback=actualizar_frame,
                progreso_callback=actualizar_progreso,
                detener_callback=lambda: not st.session_state.get("monitoreo_activo", True),
            )

    if resumen.get("estado") == "error":
        st.error(resumen.get("mensaje_estado", "No se pudo completar el monitoreo."))
        return

    resumen = _enriquecer_resumen_monitoreo_con_ocr(resumen)
    placa_evento = _obtener_placa_para_evento(resumen, placa_controlada)
    resumen = registrar_evento_monitoreo_si_corresponde(resumen, placa_evento, config["database"]["path"])
    resumen["velocidad_reproduccion"] = velocidad_reproduccion
    resumen["modo_rendimiento"] = modo_rendimiento
    st.session_state.frame_actual = int(resumen.get("frame_actual", st.session_state.get("frame_actual", 0)) or 0)
    if resumen.get("estado_persistencia"):
        st.session_state.estado_persistencia = resumen["estado_persistencia"]
    st.session_state.ultimo_resultado = resumen
    progreso.progress(1.0)
    with panel_placeholder.container():
        mostrar_panel_monitoreo_limpio(resumen, config, ejecutar_lector_cnn=False)

    if fuente_monitoreo == "Video de prueba":
        st.session_state.monitoreo_activo = False
        st.session_state.monitoreo_pausado = False
        estado_placeholder.success("Video finalizado.")

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
                    ruta_bd=config["database"]["path"],
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

                vehiculo = resultado_simulacion.get("vehiculo")
                st.subheader("VehÃ­culo consultado")
                if vehiculo:
                    col_veh1, col_veh2, col_veh3 = st.columns(3)
                    col_veh1.metric("Marca", vehiculo.get("marca", "Sin registro"))
                    col_veh2.metric("Modelo", vehiculo.get("modelo", "Sin registro"))
                    col_veh3.metric("Color", vehiculo.get("color", "Sin registro"))

                    col_veh4, col_veh5, col_veh6 = st.columns(3)
                    col_veh4.metric("Propietario", vehiculo.get("propietario", "Sin registro"))
                    col_veh5.metric("Correo", vehiculo.get("correo", "Sin registro"))
                    col_veh6.metric("Estado", vehiculo.get("estado", "Sin registro"))
                else:
                    st.warning("La placa no existe en la base de datos de vehÃ­culos.")

                st.success(f"Evento guardado en base de datos con ID: {resultado_simulacion.get('evento_id')}")

                notificacion = resultado_simulacion.get("notificacion")
                if notificacion:
                    st.subheader("NotificaciÃ³n simulada")
                    st.metric("Destinatario", notificacion.get("destinatario") or "Sin correo")
                    st.caption(f"Asunto: {notificacion.get('asunto')}")
                    st.text_area("Mensaje", notificacion.get("mensaje", ""), height=260)
                    st.caption(f"TXT: {notificacion.get('ruta_txt')}")
                    st.caption(f"JSON: {notificacion.get('ruta_json')}")
                else:
                    st.info("No se generÃ³ notificaciÃ³n porque no existe infracciÃ³n.")

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
                    frame_rgb = cv2.cvtColor(resultado_yolo["frame_procesado"], cv2.COLOR_BGR2RGB)
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
                                    cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB),
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
                        st.image(cv2.cvtColor(res_actual["frame"], cv2.COLOR_BGR2RGB), caption="BBox actual", use_container_width=True)
                        if res_actual["detecciones"] and res_actual["detecciones"][0].get("recorte") is not None:
                            st.image(cv2.cvtColor(res_actual["detecciones"][0]["recorte"], cv2.COLOR_BGR2RGB), caption="Recorte actual", use_container_width=False)
                        if res_actual["detecciones"]:
                            st.json([{k: v for k, v in det.items() if k != "recorte"} for det in res_actual["detecciones"]])
                    with col_robo:
                        st.subheader("Modelo Roboflow")
                        st.caption(f"Clases: {res_robo.get('clases')}")
                        st.metric("Tiempo inferencia", f"{res_robo['tiempo_ms']:.2f} ms")
                        st.metric("Detecciones", len(res_robo["detecciones"]))
                        st.info(res_robo["mensaje"])
                        st.image(cv2.cvtColor(res_robo["frame"], cv2.COLOR_BGR2RGB), caption="BBox Roboflow", use_container_width=True)
                        if res_robo["detecciones"] and res_robo["detecciones"][0].get("recorte") is not None:
                            st.image(cv2.cvtColor(res_robo["detecciones"][0]["recorte"], cv2.COLOR_BGR2RGB), caption="Recorte Roboflow", use_container_width=False)
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


def pestana_base_datos(config: dict) -> None:
    st.header("Base de datos")
    ruta_bd = config["database"]["path"]

    if st.button("Inicializar base de datos", type="primary"):
        inicializar_bd(ruta_bd)
        st.success(f"Base de datos inicializada en: {ruta_bd}")

    placa_busqueda = st.text_input("Buscar placa", value="PBC1234")
    if placa_busqueda:
        vehiculo = buscar_vehiculo_por_placa(placa_busqueda, ruta_bd)
        if vehiculo:
            st.success("VehÃ­culo encontrado")
            st.json(vehiculo)
        else:
            st.warning("No existe un vehÃ­culo registrado con esa placa.")

    vehiculos = listar_vehiculos(ruta_bd)
    eventos = listar_eventos(ruta_bd)

    st.subheader("Vehiculos registrados")
    st.dataframe(pd.DataFrame(vehiculos), use_container_width=True, hide_index=True)

    st.subheader("Eventos recientes")
    st.dataframe(pd.DataFrame(eventos), use_container_width=True, hide_index=True)


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

    st.subheader("Notificaciones simuladas")
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
            "Ver notificaciÃ³n simulada",
            notificaciones,
            format_func=lambda ruta: ruta.name,
        )
        st.code(seleccionado_notificacion.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Aun no existen notificaciones simuladas guardadas.")

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
    inicializar_bd(config["database"]["path"])

    st.title("Fotorradar Ecuador IA")
    st.caption("Consola de monitoreo por video para placas ecuatorianas.")

    tabs = st.tabs(["Monitoreo", "Pruebas", "Base de datos", "Evidencias", "Configuracion"])
    with tabs[0]:
        pestana_monitoreo(config)
    with tabs[1]:
        pestana_pruebas(config)
    with tabs[2]:
        pestana_base_datos(config)
    with tabs[3]:
        pestana_evidencias(config)
    with tabs[4]:
        pestana_configuracion(config)


if __name__ == "__main__":
    main()


from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd, listar_eventos, listar_vehiculos
from src.pipeline import (
    procesar_camara_monitoreo,
    procesar_frame_video_monitoreo,
    procesar_imagen_prueba,
    procesar_video_monitoreo,
)
from src.fuzzy_system import clasificar_velocidad
from src.notifier import generar_notificacion_simulada, guardar_notificacion_simulada
from src.plate_reader import leer_placa_desde_recorte, registrar_reporte_ocr
from src.speed_estimator import SpeedTracker
from src.utils import cargar_config, guardar_archivo_subido


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
        st.info("La deteccion de placa, OCR, velocidad real y sanciones se integraran despues del flujo de video.")
        return

    vehiculo = evento.get("vehiculo") or {}
    clasificacion = evento.get("clasificacion_difusa") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Placa detectada", "Si" if evento.get("placa_detectada") else "No")
    col2.metric("Texto OCR", evento.get("texto_placa") or "Pendiente")
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
        ("Detección de placa", "completado" if placa_detectada else "pendiente"),
        ("Cruce Línea 1", "completado" if frame_linea_1 else "pendiente"),
        ("Cruce Línea 2", "completado" if frame_linea_2 else "pendiente"),
        ("Velocidad calculada", "completado" if velocidad_kmh is not None else "pendiente"),
        ("Clasificación difusa", "completado" if difuso else ("pendiente" if velocidad_kmh is not None else "no_aplica")),
        ("Consulta en base de datos", estado_bd),
        (
            "Notificación simulada",
            "completado"
            if notificacion
            else ("no_aplica" if difuso and difuso.get("nivel_infraccion") == "Sin infracción" else ("pendiente" if difuso else "no_aplica")),
        ),
    ]

    columnas = st.columns(4)
    for idx, (nombre, estado) in enumerate(estados):
        columnas[idx % 4].metric(nombre, _etiqueta_estado_flujo(estado))

    col_det1, col_det2, col_det3 = st.columns(3)
    col_det1.metric("Placa detectada", "Sí" if placa_detectada else "No")
    confianza = ultima_deteccion.get("confianza")
    col_det2.metric("Confianza", f"{float(confianza):.2f}" if confianza is not None else "Pendiente")
    mejor_confianza = resumen.get("mejor_confianza_evento")
    col_det3.metric("Mejor confianza evento", f"{mejor_confianza:.2f}" if mejor_confianza is not None else "Pendiente")

    ruta_mejor_recorte = resumen.get("ruta_mejor_recorte_evento") or resumen.get("ultimo_recorte_placa")
    if ruta_mejor_recorte:
        st.caption(f"Mejor recorte de placa: {ruta_mejor_recorte}")
        st.image(ruta_mejor_recorte, use_container_width=False)

    col_cruce1, col_cruce2 = st.columns(2)
    col_cruce1.metric("Cruce Línea 1", "Sí" if frame_linea_1 else "No")
    col_cruce1.caption(f"Frame Línea 1: {frame_linea_1 or 'Pendiente'}")
    col_cruce2.metric("Cruce Línea 2", "Sí" if frame_linea_2 else "No")
    col_cruce2.caption(f"Frame Línea 2: {frame_linea_2 or 'Pendiente'}")

    if frame_linea_1 and not frame_linea_2:
        st.warning("La placa cruzó la Línea 1, pero no cruzó la Línea 2. No se puede calcular velocidad hasta completar el cruce entre ambas líneas.")
        st.info("Esperando cruce de Línea 2.")
    elif not frame_linea_1:
        st.info("Esperando que el centro de la placa cruce la Línea 1.")


def _mostrar_diagnostico_monitoreo(resumen: dict, velocidad: dict, velocidad_kmh: float | None) -> None:
    st.subheader("Diagnóstico del monitoreo")

    placa_detectada = bool(resumen.get("ultima_deteccion")) or resumen.get("estado_placa") in {"Detectada", "Mantenida"}
    frame_linea_1 = velocidad.get("frame_cruce_linea_1")
    frame_linea_2 = velocidad.get("frame_cruce_linea_2")

    if velocidad_kmh is not None:
        motivo = "La medición de velocidad se completó correctamente."
        recomendacion = "Revise el resultado difuso, la consulta en base de datos y la notificación si corresponde."
    elif not placa_detectada:
        motivo = "No se detectó placa válida."
        recomendacion = "Ajuste confianza mínima, iluminación, enfoque, rotación, zona de cámara o posición del vehículo."
    elif not frame_linea_1:
        motivo = "La placa fue detectada, pero el centro de la placa todavía no cruzó la Línea 1."
        recomendacion = "Ubique la Línea 1 sobre la trayectoria real de la placa o use un video donde el vehículo avance hacia ambas líneas."
    elif frame_linea_1 and not frame_linea_2:
        motivo = "La placa no cruzó Línea 2."
        recomendacion = "Use un video donde el vehículo pase completamente entre ambas líneas o ajuste la posición de Línea 2."
    else:
        motivo = "La detección fue válida, pero no hubo movimiento suficiente para medir velocidad."
        recomendacion = "Verifique que la cámara esté fija, que la placa se desplace de arriba hacia abajo y que las líneas estén separadas correctamente."

    col_diag1, col_diag2 = st.columns(2)
    col_diag1.info(f"Motivo: {motivo}")
    col_diag2.info(f"Recomendación: {recomendacion}")

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
            fuente="Simulación de velocidad",
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
    col10.metric("Rotacion aplicada", resumen.get("rotacion", "Sin rotación"))

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

    st.subheader("Clasificación difusa")
    if velocidad_kmh is not None:
        difuso = resumen.get("clasificacion_difusa") or clasificar_velocidad(velocidad_kmh, resumen.get("limite_velocidad_kmh", 30.0))
        col_dif1, col_dif2, col_dif3 = st.columns(3)
        col_dif1.metric("Estado difuso", difuso["estado"])
        col_dif2.metric("Nivel de infracción", difuso["nivel_infraccion"])
        col_dif3.metric("Horas de suspensión", difuso["horas_suspension"])

        col_dif4, col_dif5 = st.columns(2)
        col_dif4.metric("Sanción", difuso["sancion"])
        col_dif5.metric("Límite evaluado", f"{difuso['limite_kmh']:.1f} km/h")
        st.info(difuso["mensaje"])
    else:
        col_dif1, col_dif2, col_dif3 = st.columns(3)
        col_dif1.metric("Estado difuso", "Pendiente")
        col_dif2.metric("Nivel de infracción", "Pendiente")
        col_dif3.metric("Sanción", "Pendiente")

    if resumen.get("placa_controlada") or resumen.get("evento_bd_id"):
        st.subheader("Resultado del evento")
        vehiculo = resumen.get("vehiculo") or {}
        difuso_evento = resumen.get("clasificacion_difusa") or {}
        notificacion = resumen.get("notificacion_simulada")
        vehiculo_encontrado = "Sí" if resumen.get("vehiculo_encontrado") else "No"

        if resumen.get("vehiculo_encontrado") is False:
            st.warning("Placa no encontrada en la base de datos.")

        col_evt_res1, col_evt_res2, col_evt_res3 = st.columns(3)
        col_evt_res1.metric("Placa usada", resumen.get("placa_controlada", "Pendiente"))
        col_evt_res2.metric("Vehículo encontrado", vehiculo_encontrado)
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
        col_evt_res10.metric("Límite de velocidad", f"{resumen.get('limite_velocidad_kmh', 0):.1f} km/h")
        col_evt_res11.metric("Estado difuso", difuso_evento.get("estado", "Pendiente"))
        col_evt_res12.metric("Nivel de infracción", difuso_evento.get("nivel_infraccion", "Pendiente"))

        col_evt_res13, col_evt_res14, col_evt_res15 = st.columns(3)
        col_evt_res13.metric("Sanción", difuso_evento.get("sancion", "Pendiente"))
        col_evt_res14.metric("Horas de suspensión", difuso_evento.get("horas_suspension", "Pendiente"))
        col_evt_res15.metric("Notificación generada", "Sí" if notificacion else "No")

        if notificacion:
            col_not1, col_not2 = st.columns(2)
            col_not1.metric("Destinatario", notificacion.get("destinatario") or "Sin correo")
            col_not2.metric("Asunto", notificacion.get("asunto") or "Pendiente")
            col_not3, col_not4 = st.columns(2)
            col_not3.metric("Ruta TXT notificación", notificacion.get("ruta_txt") or "Pendiente")
            col_not4.metric("Ruta JSON notificación", notificacion.get("ruta_json") or "Pendiente")
        elif resumen.get("evento_bd_id"):
            st.info("No se generó notificación porque no existe infracción.")

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
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def pestana_monitoreo(config: dict) -> None:
    st.header("Monitoreo")
    _inicializar_estado_monitoreo()

    control, visor = st.columns([0.32, 0.68])
    with control:
        fuente_monitoreo = st.selectbox("Fuente de monitoreo", ["Video de prueba", "Camara en vivo"])
        video = None
        indice_camara = 0

        if fuente_monitoreo == "Video de prueba":
            video = st.file_uploader("Cargar video de prueba", type=["mp4", "avi", "mov", "mkv"], key="video_monitoreo")
        else:
            indice_camara = st.number_input(
                "Indice de camara",
                min_value=0,
                value=0,
                step=1,
                help="0 normalmente corresponde a la camara principal. Si usa camara externa, pruebe 1 o 2.",
            )

        with st.expander("Modo OCR pendiente / placa controlada", expanded=False):
            placa_controlada = st.text_input(
                "Placa manual/controlada para consulta",
                value="PBC1234",
                help=(
                    "Este campo se usa temporalmente hasta integrar OCR automático. "
                    "El detector ubica visualmente la placa, pero el texto se ingresa de forma controlada "
                    "para probar BD, fuzzy, eventos y notificaciones."
                ),
            )

        rotacion = st.selectbox(
            "Rotación de imagen",
            ["Sin rotación", "Rotar 90° derecha", "Rotar 90° izquierda", "Rotar 180°"],
            index=0,
        )

        distancia_metros = st.number_input(
            "Distancia real entre lineas (m)",
            min_value=0.1,
            value=float(config["speed"]["default_distance_meters"]),
            step=0.5,
            key="distancia_monitoreo",
        )
        limite_velocidad = st.number_input(
            "Limite de velocidad del campus (km/h)",
            min_value=1.0,
            value=float(config["speed"]["campus_speed_limit_kmh"]),
            step=1.0,
            key="limite_monitoreo",
        )
        posicion_linea_1 = st.slider("Posicion Linea 1 (% altura)", 0.05, 0.95, 0.45, 0.01)
        posicion_linea_2 = st.slider("Posicion Linea 2 (% altura)", 0.05, 0.95, 0.65, 0.01)
        st.caption(
            "Ubique las lineas de forma que el centro de la placa cruce primero la Linea 1 y luego la Linea 2. "
            "Para medir velocidad real, la camara debe estar fija y la distancia fisica entre ambas lineas debe ser conocida."
        )
        frecuencia_deteccion = st.slider(
            "Frecuencia de deteccion",
            1,
            60,
            10,
            help="Cada cuantos frames se ejecuta el detector YOLO de placas si el modelo ya fue entrenado.",
        )
        conf_min = st.slider("Confianza minima de placa", 0.10, 0.90, 0.30, 0.05)
        persistencia_frames = st.slider("Persistencia de deteccion (frames)", 0, 30, 10, 1)
        max_frames = st.number_input(
            "Frames maximos a procesar (0 = video completo)",
            min_value=0,
            max_value=10000,
            value=0,
            step=100,
            help="Use 0 para procesar el video completo. Si desea limitar la prueba, use valores como 900 para 30 segundos a 30 FPS.",
        )
        modo_revision = st.radio("Modo de revision", ["Automatico", "Paso a paso"])

        opciones_velocidad = ["Normal (1x)", "Rapida (sin espera)"] if fuente_monitoreo == "Camara en vivo" else [
            "Lenta (0.25x)",
            "Media (0.5x)",
            "Normal (1x)",
            "Rapida (sin espera)",
        ]
        velocidad_reproduccion = st.selectbox("Velocidad de reproduccion", opciones_velocidad, index=2 if fuente_monitoreo == "Video de prueba" else 0)
        ancho_visualizacion = st.selectbox(
            "Ancho de visualizacion",
            ["Pequeno: 640 px", "Mediano: 800 px", "Grande: 1000 px"],
            index=1,
        )
        ancho_px = {"Pequeno: 640 px": 640, "Mediano: 800 px": 800, "Grande: 1000 px": 1000}[ancho_visualizacion]

        iniciar = st.button("Iniciar monitoreo", type="primary", use_container_width=True)
        col_btn1, col_btn2 = st.columns(2)
        pausar = col_btn1.button("Pausar", use_container_width=True)
        reanudar = col_btn2.button("Reanudar", use_container_width=True)
        detener = st.button("Detener", use_container_width=True)
        reiniciar_velocidad = st.button("Reiniciar medicion de velocidad", use_container_width=True)
        reiniciar_evento = st.button("Reiniciar evento", use_container_width=True)

        procesar_siguiente = False
        reiniciar_revision = False
        if modo_revision == "Paso a paso":
            procesar_siguiente = st.button("Procesar siguiente frame", use_container_width=True)
            reiniciar_revision = st.button("Reiniciar revision", use_container_width=True)

    with visor:
        frame_placeholder = st.empty()
        progreso = st.progress(0)
        estado_placeholder = st.empty()
        resumen_placeholder = st.container()

    if pausar:
        st.session_state.monitoreo_pausado = True
    if reanudar:
        st.session_state.monitoreo_pausado = False
    if detener:
        st.session_state.monitoreo_activo = False
        st.session_state.monitoreo_pausado = False
    if reiniciar_velocidad:
        if st.session_state.get("estado_persistencia"):
            st.session_state.estado_persistencia["speed_tracker"] = None
        if st.session_state.get("ultimo_resultado"):
            st.session_state.ultimo_resultado["velocidad"] = {}
        st.session_state.evento_velocidad_guardado_clave = None
        st.session_state.evento_monitoreo_actual = None
    if reiniciar_evento:
        if st.session_state.get("estado_persistencia"):
            st.session_state.estado_persistencia["speed_tracker"] = None
        if st.session_state.get("ultimo_resultado"):
            for clave in [
                "velocidad",
                "clasificacion_difusa",
                "vehiculo",
                "vehiculo_encontrado",
                "evento_bd_id",
                "notificacion_simulada",
                "placa_controlada",
            ]:
                st.session_state.ultimo_resultado.pop(clave, None)
        st.session_state.evento_velocidad_guardado_clave = None
        st.session_state.evento_monitoreo_actual = None
    if reiniciar_revision:
        st.session_state.frame_actual = 0
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_imagen_procesada = None
        st.session_state.placas_detectadas_acumuladas = 0
        st.session_state.estado_persistencia = None
        st.session_state.evento_velocidad_guardado_clave = None
        st.session_state.evento_monitoreo_actual = None

    if not iniciar and not procesar_siguiente and not st.session_state.get("ultimo_resultado"):
        with resumen_placeholder:
            st.info("Selecciona una fuente y presiona 'Iniciar monitoreo' para ver el procesamiento frame por frame.")
        return

    if fuente_monitoreo == "Video de prueba" and not video:
        st.warning("No se pudo iniciar el monitoreo: primero carga un video.")
        return
    if (iniciar or procesar_siguiente) and posicion_linea_2 <= posicion_linea_1:
        st.warning("La Línea 2 debe estar debajo de la Línea 1 para medir movimiento de arriba hacia abajo.")
        return

    if fuente_monitoreo == "Video de prueba" and (iniciar or procesar_siguiente) and not st.session_state.get("ruta_video_monitoreo"):
        st.session_state.ruta_video_monitoreo = guardar_archivo_subido(video, config["paths"]["input_dir"])

    if iniciar and fuente_monitoreo == "Video de prueba":
        st.session_state.ruta_video_monitoreo = guardar_archivo_subido(video, config["paths"]["input_dir"])
        st.session_state.frame_actual = 0
        st.session_state.monitoreo_activo = True
        st.session_state.monitoreo_pausado = False
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_imagen_procesada = None
        st.session_state.placas_detectadas_acumuladas = 0
        st.session_state.estado_persistencia = None
        st.session_state.evento_velocidad_guardado_clave = None
        st.session_state.evento_monitoreo_actual = None
    elif iniciar:
        st.session_state.monitoreo_activo = True
        st.session_state.monitoreo_pausado = False
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_imagen_procesada = None
        st.session_state.evento_velocidad_guardado_clave = None
        st.session_state.evento_monitoreo_actual = None

    def actualizar_frame(frame_rgb, numero_frame: int, estado_frame: dict | None = None) -> None:
        frame_placeholder.image(frame_rgb, channels="RGB", width=ancho_px)
        st.session_state.ultima_imagen_procesada = frame_rgb
        if estado_frame:
            ultima_confianza = estado_frame.get("ultima_confianza")
            texto_confianza = f"{ultima_confianza:.2f}" if ultima_confianza is not None else "Pendiente"
            estado_placeholder.info(
                f"Frame {estado_frame.get('frame_actual', numero_frame)} / {estado_frame.get('total_frames', 0)} | "
                f"FPS {estado_frame.get('fps', 0):.2f} | "
                f"Modo {estado_frame.get('modo_reproduccion')} | "
                f"Velocidad {estado_frame.get('velocidad_reproduccion')} | "
                f"Estado placa {estado_frame.get('estado_placa', 'Pendiente')} | "
                f"Eventos {estado_frame.get('eventos_placa', 0)} | "
                f"Confianza {texto_confianza}"
            )
        else:
            estado_placeholder.info(f"Procesando frame {numero_frame}")

    def actualizar_progreso(valor: float) -> None:
        progreso.progress(valor)

    if modo_revision == "Paso a paso" and fuente_monitoreo == "Video de prueba":
        if iniciar or procesar_siguiente:
            if st.session_state.monitoreo_pausado:
                st.info("La revision esta pausada. Presiona Reanudar para continuar.")
            else:
                resultado = procesar_frame_video_monitoreo(
                    st.session_state.ruta_video_monitoreo,
                    st.session_state.frame_actual,
                    distancia_lineas_m=distancia_metros,
                    limite_velocidad_kmh=limite_velocidad,
                    posicion_linea_1=posicion_linea_1,
                    posicion_linea_2=posicion_linea_2,
                    frecuencia_deteccion=frecuencia_deteccion,
                    conf_min=conf_min,
                    persistencia_frames=persistencia_frames,
                    rotacion=rotacion,
                    estado_persistencia=st.session_state.estado_persistencia,
                )
                if resultado.get("frame_rgb") is not None:
                    st.session_state.frame_actual = resultado.get("frame_actual", st.session_state.frame_actual + 1)
                    st.session_state.estado_persistencia = resultado.get("estado_persistencia")
                    resultado["modo_reproduccion"] = "Paso a paso"
                    resultado["velocidad_reproduccion"] = "Manual"
                    resultado = registrar_evento_monitoreo_si_corresponde(
                        resultado,
                        placa_controlada,
                        config["database"]["path"],
                    )
                    st.session_state.ultimo_resultado = resultado
                    st.session_state.ultima_imagen_procesada = resultado["frame_rgb"]
                else:
                    st.session_state.ultimo_resultado = resultado

        resultado = st.session_state.get("ultimo_resultado")
        if resultado and resultado.get("frame_rgb") is not None:
            frame_placeholder.image(resultado["frame_rgb"], channels="RGB", width=ancho_px)
            total = max(resultado.get("total_frames", 1), 1)
            progreso.progress(min(resultado.get("frame_actual", 0) / total, 1.0))
            estado_placeholder.info(f"Frame actual: {resultado.get('frame_actual', 0)} / {resultado.get('total_frames', 0)}")
            with resumen_placeholder:
                mostrar_resumen_monitoreo(resultado)
        elif resultado:
            st.info(resultado.get("mensaje_estado", "Revision finalizada."))
        return

    if st.session_state.monitoreo_pausado and st.session_state.get("ultima_imagen_procesada") is not None:
        frame_placeholder.image(st.session_state.ultima_imagen_procesada, channels="RGB", width=ancho_px)
        st.info("Monitoreo pausado. Presiona Reanudar para continuar.")
        if st.session_state.get("ultimo_resultado"):
            with resumen_placeholder:
                mostrar_resumen_monitoreo(st.session_state.ultimo_resultado)
        return

    if not iniciar:
        if st.session_state.get("ultimo_resultado"):
            if st.session_state.get("ultima_imagen_procesada") is not None:
                frame_placeholder.image(st.session_state.ultima_imagen_procesada, channels="RGB", width=ancho_px)
            with resumen_placeholder:
                mostrar_resumen_monitoreo(st.session_state.ultimo_resultado)
        return

    with st.spinner("Procesando monitoreo..."):
        if fuente_monitoreo == "Video de prueba":
            resumen = procesar_video_monitoreo(
                st.session_state.ruta_video_monitoreo,
                distancia_lineas_m=distancia_metros,
                limite_velocidad_kmh=limite_velocidad,
                posicion_linea_1=posicion_linea_1,
                posicion_linea_2=posicion_linea_2,
                frecuencia_deteccion=frecuencia_deteccion,
                max_frames=max_frames,
                velocidad_reproduccion=velocidad_reproduccion,
                conf_min=conf_min,
                persistencia_frames=persistencia_frames,
                rotacion=rotacion,
                frame_callback=actualizar_frame,
                progreso_callback=actualizar_progreso,
                detener_callback=lambda: st.session_state.get("monitoreo_pausado", False) or not st.session_state.get("monitoreo_activo", True),
            )
        else:
            resumen = procesar_camara_monitoreo(
                indice_camara=int(indice_camara),
                distancia_lineas_m=distancia_metros,
                limite_velocidad_kmh=limite_velocidad,
                posicion_linea_1=posicion_linea_1,
                posicion_linea_2=posicion_linea_2,
                frecuencia_deteccion=frecuencia_deteccion,
                max_frames=max_frames,
                velocidad_reproduccion=velocidad_reproduccion,
                conf_min=conf_min,
                persistencia_frames=persistencia_frames,
                rotacion=rotacion,
                frame_callback=actualizar_frame,
                progreso_callback=actualizar_progreso,
                detener_callback=lambda: st.session_state.get("monitoreo_pausado", False) or not st.session_state.get("monitoreo_activo", True),
            )

    if resumen.get("estado") == "error":
        st.error(resumen["mensaje_estado"])
        return

    resumen = registrar_evento_monitoreo_si_corresponde(
        resumen,
        placa_controlada,
        config["database"]["path"],
    )
    progreso.progress(1.0)
    resumen["modo_reproduccion"] = modo_revision
    resumen["velocidad_reproduccion"] = velocidad_reproduccion
    st.session_state.ultimo_resultado = resumen
    with resumen_placeholder:
        mostrar_resumen_monitoreo(resumen)


def pestana_pruebas(config: dict) -> None:
    st.header("Pruebas")

    with st.expander("Pruebas con imagen", expanded=True):
        imagen = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png", "bmp"], key="imagen_prueba")
        if imagen:
            ruta_imagen = guardar_archivo_subido(imagen, config["paths"]["input_dir"])
            st.image(ruta_imagen, use_container_width=True)
        else:
            ruta_imagen = None

    with st.expander("Simulación de velocidad", expanded=True):
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            fps_simulado = st.number_input("FPS simulado", min_value=1.0, value=30.0, step=1.0)
            placa_simulada = st.text_input(
                "Placa manual/controlada",
                value="PBC1234",
                help="Valor temporal para probar base de datos y eventos hasta integrar OCR automático.",
            )
            distancia_simulada = st.number_input(
                "Distancia real entre líneas en metros",
                min_value=0.1,
                value=10.0,
                step=0.5,
                key="distancia_simulacion_velocidad",
            )
            limite_simulado = st.number_input(
                "Límite de velocidad del campus (km/h)",
                min_value=1.0,
                value=30.0,
                step=1.0,
                key="limite_simulacion_velocidad",
            )
            posicion_sim_linea_1 = st.slider("Posición Línea 1 (% altura)", 0.05, 0.95, 0.35, 0.01)
            posicion_sim_linea_2 = st.slider("Posición Línea 2 (% altura)", 0.05, 0.95, 0.75, 0.01)
        with col_sim2:
            frame_inicial_sim = st.number_input("Frame inicial de la placa", min_value=0, value=0, step=1)
            frame_cruce_linea_1_sim = st.number_input("Frame en que cruza Línea 1", min_value=0, value=30, step=1)
            frame_cruce_linea_2_sim = st.number_input("Frame en que cruza Línea 2", min_value=0, value=75, step=1)
            total_frames_sim = st.number_input("Total de frames simulados", min_value=1, value=120, step=1)

        ejecutar_simulacion = st.button("Ejecutar simulación de velocidad", type="primary")

        if ejecutar_simulacion:
            if posicion_sim_linea_2 <= posicion_sim_linea_1:
                st.warning("La Línea 2 debe estar debajo de la Línea 1 para simular movimiento de arriba hacia abajo.")
            elif frame_cruce_linea_2_sim <= frame_cruce_linea_1_sim:
                st.warning("El frame de cruce de Línea 2 debe ser mayor que el frame de cruce de Línea 1.")
            elif total_frames_sim <= frame_cruce_linea_2_sim:
                st.warning("El total de frames simulados debe ser mayor que el frame de cruce de Línea 2.")
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
                col_res1.metric("Frame cruce Línea 1", resumen_velocidad.get("frame_cruce_linea_1") or "Pendiente")
                col_res2.metric("Frame cruce Línea 2", resumen_velocidad.get("frame_cruce_linea_2") or "Pendiente")
                col_res3.metric("Estado tracker", resumen_velocidad.get("estado", "Pendiente"))

                col_res4, col_res5, col_res6 = st.columns(3)
                tiempo_entre = resumen_velocidad.get("tiempo_entre_lineas")
                velocidad_kmh = resumen_velocidad.get("velocidad_kmh")
                col_res4.metric("Tiempo entre líneas", f"{tiempo_entre:.3f} s" if tiempo_entre is not None else "Pendiente")
                col_res5.metric("Distancia configurada", f"{distancia_simulada:.1f} m")
                col_res6.metric("Velocidad calculada", f"{velocidad_kmh:.2f} km/h" if velocidad_kmh is not None else "Pendiente")

                if difuso:
                    st.subheader("Clasificación difusa de la simulación")
                    col_dif1, col_dif2, col_dif3 = st.columns(3)
                    col_dif1.metric("Estado", difuso["estado"])
                    col_dif2.metric("Nivel de infracción", difuso["nivel_infraccion"])
                    col_dif3.metric("Horas de suspensión", difuso["horas_suspension"])

                    col_dif4, col_dif5 = st.columns(2)
                    col_dif4.metric("Sanción", difuso["sancion"])
                    col_dif5.metric("Límite evaluado", f"{difuso['limite_kmh']:.1f} km/h")
                    st.info(difuso["mensaje"])
                    st.json(difuso["grados"])

                vehiculo = resultado_simulacion.get("vehiculo")
                st.subheader("Vehículo consultado")
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
                    st.warning("La placa no existe en la base de datos de vehículos.")

                st.success(f"Evento guardado en base de datos con ID: {resultado_simulacion.get('evento_id')}")

                notificacion = resultado_simulacion.get("notificacion")
                if notificacion:
                    st.subheader("Notificación simulada")
                    st.metric("Destinatario", notificacion.get("destinatario") or "Sin correo")
                    st.caption(f"Asunto: {notificacion.get('asunto')}")
                    st.text_area("Mensaje", notificacion.get("mensaje", ""), height=260)
                    st.caption(f"TXT: {notificacion.get('ruta_txt')}")
                    st.caption(f"JSON: {notificacion.get('ruta_json')}")
                else:
                    st.info("No se generó notificación porque no existe infracción.")

                frame_final_rgb = resultado_simulacion.get("frame_final_rgb")
                if frame_final_rgb is not None:
                    st.image(frame_final_rgb, channels="RGB", caption="Frame final de la simulación", width=640)

                st.caption(f"Evidencias guardadas en: reports/evidencias/simulacion_velocidad/")
                st.caption(f"Resumen JSON: {resultado_simulacion['ruta_json']}")
                st.json(datos_simulacion)

    with st.expander("OCR experimental sobre recortes", expanded=False):
        recortes, fuente_recorte = _buscar_recortes_ocr()
        modo_ocr_exp = st.radio(
            "Fuente de recorte OCR",
            ["Recorte generado por el sistema", "Carga manual"],
            horizontal=True,
        )
        placa_esperada_ocr = st.text_input("Placa esperada", value="", key="placa_esperada_ocr")
        metodo_segmentacion_ocr = st.selectbox(
            "Método de segmentación",
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
                "Cargar imagen de placa para OCR experimental",
                type=["jpg", "jpeg", "png", "bmp"],
                key="imagen_ocr_experimental",
            )
            if imagen_ocr:
                ruta_ocr = guardar_archivo_subido(imagen_ocr, config["paths"]["input_dir"])
                fuente_ocr = "carga_manual"
                st.image(ruta_ocr, caption="Imagen cargada", use_container_width=False)

        ejecutar_ocr = st.button("Ejecutar OCR experimental", type="primary")
        if ejecutar_ocr:
            if not ruta_ocr:
                st.warning("Selecciona o carga un recorte de placa antes de ejecutar OCR experimental.")
            else:
                resultado_ocr = leer_placa_desde_recorte(
                    ruta_ocr,
                    placa_esperada=placa_esperada_ocr,
                    metodo_segmentacion=metodo_segmentacion_ocr,
                )
                reporte_ocr = registrar_reporte_ocr(resultado_ocr, fuente_ocr, placa_esperada_ocr)

                col_ocr1, col_ocr2, col_ocr3 = st.columns(3)
                col_ocr1.metric("Texto detectado crudo", resultado_ocr.get("texto_detectado_crudo") or resultado_ocr["texto_detectado"])
                col_ocr2.metric("Texto postprocesado", resultado_ocr.get("texto_postprocesado") or "Pendiente")
                col_ocr3.metric("Confianza promedio", f"{resultado_ocr.get('confianza_promedio', 0.0):.2f}")

                formato = resultado_ocr.get("formato", {})
                comparacion = resultado_ocr.get("comparacion", {})
                col_ocr4, col_ocr5, col_ocr6 = st.columns(3)
                col_ocr4.metric("Formato válido", "Sí" if formato.get("valido") else "No")
                col_ocr5.metric("Acierto", "Sí" if comparacion.get("coincide") else "No")
                col_ocr6.metric("Caracteres aceptados", resultado_ocr["cantidad_caracteres_segmentados"])

                col_ocr7, col_ocr8 = st.columns(2)
                col_ocr7.metric("Contornos rechazados", resultado_ocr.get("cantidad_contornos_rechazados", 0))
                col_ocr8.metric("Método", resultado_ocr.get("metodo_segmentacion", metodo_segmentacion_ocr))

                st.caption(f"Estado: {resultado_ocr['estado']}")
                st.info(resultado_ocr["mensaje"])
                st.caption(comparacion.get("mensaje", "Sin comparación."))

                preprocesamiento = resultado_ocr.get("preprocesamiento", {})
                ruta_preprocesada = preprocesamiento.get("ruta_imagen_procesada")
                if ruta_preprocesada:
                    st.image(ruta_preprocesada, caption="Imagen preprocesada final", use_container_width=False)

                ruta_banda = resultado_ocr.get("ruta_banda")
                if ruta_banda:
                    st.image(ruta_banda, caption="Banda principal de caracteres", use_container_width=False)

                ruta_debug = resultado_ocr.get("ruta_debug_segmentacion")
                if ruta_debug:
                    st.image(ruta_debug, caption="Debug segmentación: verde aceptado, rojo rechazado", use_container_width=False)

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
                if predicciones:
                    st.subheader("Predicciones por carácter")
                    for pred in predicciones:
                        col_img, col_info = st.columns([0.18, 0.82])
                        if pred.get("ruta_caracter"):
                            col_img.image(pred["ruta_caracter"], caption=f"#{pred.get('indice')}", use_container_width=True)
                        top3 = ", ".join(
                            f"{item['caracter']} ({item['confianza']:.2f})"
                            for item in pred.get("top3_predicciones", [])
                        )
                        col_info.write(
                            {
                                "indice": pred.get("indice"),
                                "prediccion": pred.get("caracter_predicho"),
                                "confianza": round(float(pred.get("confianza", 0.0)), 4),
                                "top3": top3,
                            }
                        )

                st.caption(f"Reporte JSON: {reporte_ocr['ruta_json']}")
                st.caption(f"Reporte CSV: {reporte_ocr['ruta_csv']}")

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
            st.success("Vehículo encontrado")
            st.json(vehiculo)
        else:
            st.warning("No existe un vehículo registrado con esa placa.")

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
            "Ver notificación simulada",
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

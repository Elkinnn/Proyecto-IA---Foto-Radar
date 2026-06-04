from datetime import datetime
from functools import lru_cache
from pathlib import Path
import csv
import json
import time

import cv2

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd
from src.fuzzy_system import clasificar_velocidad
from src.notifier import enviar_notificacion_sancion
from src.plate_detector import PlateDetector, dibujar_deteccion
from src.plate_reader import PlateReader, leer_placa_desde_recorte
from src.report_generator import guardar_reporte
from src.speed_estimator import SpeedTracker, estimar_velocidad


EXCLUIR_ZONA_SUPERIOR_PORCENTAJE = 0.20
GUARDAR_RECORTE_CADA_N_FRAMES = 15
BBOX_SUAVIZADO_ALPHA = 0.6
DEFAULT_PLATE_CROP_SELECTION = {
    "enabled": True,
    "buffer_frames": 15,
    "min_aspect_ratio": 1.5,
    "max_aspect_ratio": 6.5,
    "min_area_relative": 0.0003,
    "max_area_relative": 0.08,
    "border_margin_px": 5,
    "min_sharpness": 30.0,
    "cooldown_frames": 45,
    "min_score_improvement": 0.03,
}


@lru_cache(maxsize=4)
def _obtener_detector_cache(model_path: str | None) -> PlateDetector:
    return PlateDetector(model_path) if model_path else PlateDetector()


def _registrar_resultado(resultado: dict, config: dict) -> dict:
    ruta_bd = config["database"]["path"]
    clasificacion = resultado["clasificacion_difusa"]

    sancion_aplica = bool(clasificacion.get("sancion_aplica"))

    if sancion_aplica:
        resultado["notificacion"] = enviar_notificacion_sancion(resultado.get("vehiculo"), resultado, config)

    ruta_reporte = guardar_reporte(resultado, config["paths"]["reports_dir"])
    resultado["ruta_reporte"] = ruta_reporte

    evento = {
        "fecha_hora": resultado["fecha_hora"],
        "placa": resultado["texto_placa"],
        "velocidad": resultado["velocidad_kmh"],
        "estado": clasificacion["estado"],
        "sancion": "SI" if sancion_aplica else "NO",
        "evidencia": ruta_reporte,
    }
    resultado["id_evento"] = guardar_evento(evento, ruta_bd)
    return resultado


def procesar_imagen_prueba(
    ruta_archivo: str,
    config: dict,
    modo_ocr: str = "manual_controlado",
    placa_manual: str | None = None,
    tiempo_segundos: float | None = None,
) -> dict:
    ruta_bd = config["database"]["path"]
    inicializar_bd(ruta_bd)

    detector = PlateDetector(config["models"].get("plate_detector_model", config["models"]["plate_detector_path"]))
    lector = PlateReader()

    deteccion = detector.detectar(ruta_archivo)
    lectura = lector.leer(
        deteccion,
        modo_ocr,
        placa_manual=placa_manual or config["ocr"]["manual_test_plate"],
    )
    velocidad_kmh = estimar_velocidad(config, tiempo_segundos=tiempo_segundos)
    clasificacion = clasificar_velocidad(velocidad_kmh, config)
    vehiculo = buscar_vehiculo_por_placa(lectura["texto"], ruta_bd)
    ruta_procesada = dibujar_deteccion(ruta_archivo, deteccion, config["paths"]["output_dir"])

    resultado = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "ruta_archivo": ruta_archivo,
        "ruta_archivo_procesado": ruta_procesada,
        "deteccion": deteccion,
        "placa_detectada": deteccion["detectada"],
        "texto_placa": lectura["texto"],
        "lectura": lectura,
        "velocidad_kmh": velocidad_kmh,
        "clasificacion_difusa": clasificacion,
        "vehiculo": vehiculo,
        "sancion_generada": bool(clasificacion.get("sancion_aplica")),
        "notificacion": None,
        "ruta_reporte": None,
        "id_evento": None,
    }

    return _registrar_resultado(resultado, config)


def procesar_video_monitoreo(
    ruta_video,
    distancia_lineas_m: float = 10.0,
    limite_velocidad_kmh: float = 30.0,
    posicion_linea_1: float = 0.45,
    posicion_linea_2: float = 0.65,
    frecuencia_deteccion: int = 10,
    max_frames: int = 300,
    velocidad_reproduccion: str = "Rapida",
    conf_min: float = 0.30,
    persistencia_frames: int = 10,
    rotacion: str = "Sin rotación",
    model_path: str | None = None,
    start_frame: int = 0,
    estado_persistencia: dict | None = None,
    plate_crop_selection: dict | None = None,
    inference_size: int = 640,
    render_every_n_frames: int = 1,
    max_display_fps: int = 24,
    demo_fluido: bool = False,
    guardar_debug: bool = False,
    frame_callback=None,
    progreso_callback=None,
    detener_callback=None,
) -> dict:
    return _procesar_fuente_monitoreo(
        fuente=ruta_video,
        nombre_fuente="Video de prueba",
        mensaje_error="No se pudo abrir el video seleccionado.",
        evidencia_subdir="monitoreo_video",
        distancia_lineas_m=distancia_lineas_m,
        limite_velocidad_kmh=limite_velocidad_kmh,
        posicion_linea_1=posicion_linea_1,
        posicion_linea_2=posicion_linea_2,
        frecuencia_deteccion=frecuencia_deteccion,
        max_frames=max_frames,
        velocidad_reproduccion=velocidad_reproduccion,
        conf_min=conf_min,
        persistencia_frames=persistencia_frames,
        rotacion=rotacion,
        model_path=model_path,
        start_frame=start_frame,
        estado_persistencia=estado_persistencia,
        plate_crop_selection=plate_crop_selection,
        inference_size=inference_size,
        render_every_n_frames=render_every_n_frames,
        max_display_fps=max_display_fps,
        demo_fluido=demo_fluido,
        guardar_debug=guardar_debug,
        frame_callback=frame_callback,
        progreso_callback=progreso_callback,
        detener_callback=detener_callback,
    )


def procesar_camara_monitoreo(
    indice_camara=0,
    distancia_lineas_m: float = 10.0,
    limite_velocidad_kmh: float = 30.0,
    posicion_linea_1: float = 0.45,
    posicion_linea_2: float = 0.65,
    frecuencia_deteccion: int = 10,
    max_frames: int = 300,
    velocidad_reproduccion: str = "Normal",
    conf_min: float = 0.30,
    persistencia_frames: int = 10,
    rotacion: str = "Sin rotación",
    model_path: str | None = None,
    plate_crop_selection: dict | None = None,
    inference_size: int = 640,
    render_every_n_frames: int = 1,
    max_display_fps: int = 24,
    demo_fluido: bool = False,
    camera_width: int = 1280,
    camera_height: int = 720,
    camera_fps: int = 30,
    guardar_debug: bool = False,
    frame_callback=None,
    progreso_callback=None,
    detener_callback=None,
) -> dict:
    return _procesar_fuente_monitoreo(
        fuente=int(indice_camara),
        nombre_fuente="Camara en vivo",
        mensaje_error="No se pudo abrir la camara. Pruebe con otro indice de camara o verifique permisos.",
        evidencia_subdir="monitoreo_camara",
        distancia_lineas_m=distancia_lineas_m,
        limite_velocidad_kmh=limite_velocidad_kmh,
        posicion_linea_1=posicion_linea_1,
        posicion_linea_2=posicion_linea_2,
        frecuencia_deteccion=frecuencia_deteccion,
        max_frames=max_frames,
        velocidad_reproduccion=velocidad_reproduccion,
        conf_min=conf_min,
        persistencia_frames=persistencia_frames,
        rotacion=rotacion,
        model_path=model_path,
        plate_crop_selection=plate_crop_selection,
        inference_size=inference_size,
        render_every_n_frames=render_every_n_frames,
        max_display_fps=max_display_fps,
        demo_fluido=demo_fluido,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        guardar_debug=guardar_debug,
        frame_callback=frame_callback,
        progreso_callback=progreso_callback,
        detener_callback=detener_callback,
    )


def procesar_frame_video_monitoreo(
    ruta_video,
    numero_frame: int,
    distancia_lineas_m: float = 10.0,
    limite_velocidad_kmh: float = 30.0,
    posicion_linea_1: float = 0.45,
    posicion_linea_2: float = 0.65,
    frecuencia_deteccion: int = 10,
    conf_min: float = 0.30,
    persistencia_frames: int = 10,
    rotacion: str = "Sin rotación",
    estado_persistencia: dict | None = None,
) -> dict:
    captura = cv2.VideoCapture(ruta_video)
    detector = PlateDetector()

    if not captura.isOpened():
        return {
            "estado": "error",
            "mensaje_estado": "No se pudo abrir el video seleccionado.",
            "frame_rgb": None,
        }

    fps, ancho, alto, total_frames, duracion = _leer_metadata_video(captura)
    if total_frames > 0 and numero_frame >= total_frames:
        captura.release()
        return {
            "estado": "finalizado",
            "mensaje_estado": "Se llego al final del video.",
            "frame_rgb": None,
            "frame_actual": total_frames,
            "total_frames": total_frames,
            "fps": fps,
            "ancho": ancho,
            "alto": alto,
            "duracion_segundos": duracion,
            "segundos_procesados": _calcular_segundos_procesados(total_frames, fps),
            "modo_procesamiento": "video completo",
        }

    captura.set(cv2.CAP_PROP_POS_FRAMES, max(numero_frame, 0))
    ok, frame = captura.read()
    captura.release()

    if not ok:
        return {
            "estado": "error",
            "mensaje_estado": "No se pudo leer el frame solicitado.",
            "frame_rgb": None,
        }
    frame = aplicar_rotacion(frame, rotacion)
    alto_rotado, ancho_rotado = frame.shape[:2]

    evidencia_dir = Path("reports") / "evidencias" / "monitoreo_video"
    placas_dir = Path("reports") / "evidencias" / "placas_detectadas"
    evidencia_dir.mkdir(parents=True, exist_ok=True)
    placas_dir.mkdir(parents=True, exist_ok=True)

    estado_frame = _procesar_frame_monitoreo(
        frame,
        detector,
        numero_frame + 1,
        distancia_lineas_m,
        limite_velocidad_kmh,
        posicion_linea_1,
        posicion_linea_2,
        "Video de prueba",
        frecuencia_deteccion,
        evidencia_dir,
        placas_dir,
        estado_persistencia or _crear_estado_persistencia(),
        conf_min,
        persistencia_frames,
        fps,
        None,
    )
    frame_rgb = cv2.cvtColor(estado_frame["frame_visual"], cv2.COLOR_BGR2RGB)

    return {
        "estado": "finalizado",
        "mensaje_estado": estado_frame["mensaje_detector"],
        "frame_rgb": frame_rgb,
        "frame_actual": numero_frame + 1,
        "frames_procesados": numero_frame + 1,
        "fps": fps,
        "ancho": ancho_rotado,
        "alto": alto_rotado,
        "total_frames": total_frames,
        "duracion_segundos": duracion,
        "segundos_procesados": _calcular_segundos_procesados(numero_frame + 1, fps),
        "modo_procesamiento": "paso a paso",
        "fuente": "Video de prueba",
        "rotacion": rotacion,
        "distancia_lineas_m": distancia_lineas_m,
        "posicion_linea_1": posicion_linea_1,
        "posicion_linea_2": posicion_linea_2,
        "limite_velocidad_kmh": limite_velocidad_kmh,
        "modelo_detector_disponible": detector.model is not None,
        "mensaje_detector": estado_frame["mensaje_detector"],
        "estado_placa": estado_frame["estado_placa"],
        "detecciones_frame": estado_frame["detecciones_frame"],
        "detecciones_brutas": estado_frame["detecciones_brutas"],
        "detecciones_validas": estado_frame["detecciones_validas"],
        "motivos_rechazo": estado_frame["motivos_rechazo"],
        "eventos_placa": estado_frame["eventos_placa"],
        "placas_detectadas": estado_frame["eventos_placa"],
        "frames_desde_ultima_deteccion": estado_frame["frames_desde_ultima_deteccion"],
        "ultima_deteccion": estado_frame["ultima_deteccion"],
        "ultimo_recorte_placa": estado_frame["ultimo_recorte_placa"],
        "ultimo_frame_deteccion": estado_frame["ultimo_frame_deteccion"],
        "estado_persistencia": estado_frame["estado_persistencia"],
        "velocidad": estado_frame["velocidad"],
        "evento_activo": estado_frame["evento_activo"],
        "evento_id": estado_frame["evento_id"],
        "mejor_confianza_evento": estado_frame["mejor_confianza_evento"],
        "frame_mejor_evento": estado_frame["frame_mejor_evento"],
        "frames_sin_deteccion": estado_frame["frames_sin_deteccion"],
        "ruta_mejor_recorte_evento": estado_frame["ruta_mejor_recorte_evento"],
    }


def _procesar_fuente_monitoreo(
    fuente,
    nombre_fuente: str,
    mensaje_error: str,
    evidencia_subdir: str,
    distancia_lineas_m: float,
    limite_velocidad_kmh: float,
    posicion_linea_1: float,
    posicion_linea_2: float,
    frecuencia_deteccion: int,
    max_frames: int,
    velocidad_reproduccion: str,
    conf_min: float,
    persistencia_frames: int,
    rotacion: str,
    model_path: str | None = None,
    start_frame: int = 0,
    estado_persistencia: dict | None = None,
    plate_crop_selection: dict | None = None,
    inference_size: int = 640,
    render_every_n_frames: int = 1,
    max_display_fps: int = 24,
    demo_fluido: bool = False,
    camera_width: int | None = None,
    camera_height: int | None = None,
    camera_fps: int | None = None,
    guardar_debug: bool = False,
    frame_callback=None,
    progreso_callback=None,
    detener_callback=None,
) -> dict:
    if isinstance(fuente, int):
        captura = cv2.VideoCapture(fuente, cv2.CAP_DSHOW)
        if camera_width:
            captura.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_width))
        if camera_height:
            captura.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_height))
        if camera_fps:
            captura.set(cv2.CAP_PROP_FPS, int(camera_fps))
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        captura = cv2.VideoCapture(fuente)
    detector = _obtener_detector_cache(str(model_path) if model_path else None)

    if not captura.isOpened():
        return {
            "estado": "error",
            "mensaje_estado": mensaje_error,
            "frames_procesados": 0,
            "fps": 0.0,
            "ancho": 0,
            "alto": 0,
            "total_frames": 0,
            "duracion_segundos": 0.0,
            "segundos_procesados": 0.0,
            "modo_procesamiento": _obtener_modo_procesamiento(max_frames),
            "fuente": nombre_fuente,
            "rotacion": rotacion,
            "distancia_lineas_m": distancia_lineas_m,
            "limite_velocidad_kmh": limite_velocidad_kmh,
            "primer_frame_evidencia": None,
            "ultimo_frame_evidencia": None,
            "modelo_detector_disponible": detector.model is not None,
            "mensaje_detector": detector.estado,
            "placas_detectadas": 0,
            "detecciones_frame": 0,
            "eventos_placa": 0,
            "estado_placa": "Pendiente",
            "frames_desde_ultima_deteccion": 0,
            "ultima_deteccion": None,
            "ultimo_recorte_placa": None,
            "ultimo_frame_deteccion": None,
            "velocidad": _resumen_velocidad_vacio(distancia_lineas_m, 0.0),
        }

    fps, ancho, alto, total_frames, duracion = _leer_metadata_video(captura)
    if isinstance(fuente, int) and fps <= 0 and camera_fps:
        fps = float(camera_fps)
    start_frame = max(int(start_frame or 0), 0)
    if start_frame > 0 and total_frames > 0:
        start_frame = min(start_frame, max(total_frames - 1, 0))
    if start_frame > 0:
        captura.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_procesados = 0
    ultimo_frame = None

    evidencia_dir = Path("reports") / "evidencias" / evidencia_subdir
    evidencia_dir.mkdir(parents=True, exist_ok=True)
    primer_frame_evidencia = evidencia_dir / "primer_frame_procesado.jpg"
    ultimo_frame_evidencia = evidencia_dir / "ultimo_frame_procesado.jpg"
    placas_dir = Path("reports") / "evidencias" / "placas_detectadas"
    placas_dir.mkdir(parents=True, exist_ok=True)
    estado_persistencia = estado_persistencia or _crear_estado_persistencia()
    detecciones_frame = 0
    detecciones_brutas = 0
    detecciones_validas = 0
    motivos_rechazo = ""
    eventos_placa = 0
    estado_placa = "Pendiente"
    ultima_deteccion = None
    ultimo_recorte_placa = None
    ultimo_frame_deteccion = None
    mensaje_detector = detector.estado
    tiempo_yolo_ms = 0.0
    resolucion_inferencia = f"{ancho}x{alto}"
    tiempo_inicio_total = time.perf_counter()
    tiempos_etapa = {
        "lectura_frame_ms": 0.0,
        "rotacion_ms": 0.0,
        "procesamiento_frame_ms": 0.0,
        "render_streamlit_ms": 0.0,
        "guardado_evidencia_ms": 0.0,
    }
    frames_mostrados = 0
    frames_yolo_analizados = 0
    mejor_confianza_evento = None
    frame_mejor_evento = None
    ruta_mejor_recorte_evento = None

    objetivo_frames = max_frames if max_frames > 0 else total_frames
    if total_frames > 0 and max_frames > 0:
        objetivo_frames = min(max_frames, total_frames)

    while captura.isOpened():
        if detener_callback and detener_callback():
            break

        t_lectura = time.perf_counter()
        ok, frame = captura.read()
        tiempos_etapa["lectura_frame_ms"] = round((time.perf_counter() - t_lectura) * 1000, 3)
        if not ok:
            break
        t_rotacion = time.perf_counter()
        frame = aplicar_rotacion(frame, rotacion)
        tiempos_etapa["rotacion_ms"] = round((time.perf_counter() - t_rotacion) * 1000, 3)
        alto_actual, ancho_actual = frame.shape[:2]

        frames_procesados += 1
        numero_frame_actual = start_frame + frames_procesados
        t_procesamiento = time.perf_counter()
        estado_frame = _procesar_frame_monitoreo(
            frame,
            detector,
            numero_frame_actual,
            distancia_lineas_m,
            limite_velocidad_kmh,
            posicion_linea_1,
            posicion_linea_2,
            nombre_fuente,
            frecuencia_deteccion,
            evidencia_dir,
            placas_dir,
            estado_persistencia,
            conf_min,
            persistencia_frames,
            fps,
            plate_crop_selection,
            inference_size,
            guardar_debug,
        )
        tiempos_etapa["procesamiento_frame_ms"] = round((time.perf_counter() - t_procesamiento) * 1000, 3)
        frame = estado_frame["frame_visual"]
        estado_persistencia = estado_frame["estado_persistencia"]
        detecciones_frame = estado_frame["detecciones_frame"]
        detecciones_brutas = estado_frame["detecciones_brutas"]
        detecciones_validas = estado_frame["detecciones_validas"]
        motivos_rechazo = estado_frame["motivos_rechazo"]
        eventos_placa = estado_frame["eventos_placa"]
        estado_placa = estado_frame["estado_placa"]
        mensaje_detector = estado_frame["mensaje_detector"]
        tiempo_yolo_ms = estado_frame.get("tiempo_yolo_ms", tiempo_yolo_ms)
        if estado_frame.get("yolo_ejecutado"):
            frames_yolo_analizados += 1
        resolucion_inferencia = estado_frame.get("resolucion_inferencia", resolucion_inferencia)
        ultima_deteccion = estado_frame["ultima_deteccion"] or ultima_deteccion
        ultimo_recorte_placa = estado_frame["ultimo_recorte_placa"] or ultimo_recorte_placa
        ultimo_frame_deteccion = estado_frame["ultimo_frame_deteccion"] or ultimo_frame_deteccion
        mejor_confianza_evento = estado_frame["mejor_confianza_evento"] if estado_frame["mejor_confianza_evento"] is not None else mejor_confianza_evento
        frame_mejor_evento = estado_frame["frame_mejor_evento"] or frame_mejor_evento
        ruta_mejor_recorte_evento = estado_frame["ruta_mejor_recorte_evento"] or ruta_mejor_recorte_evento

        t_guardado = time.perf_counter()
        if frames_procesados == 1:
            cv2.imwrite(str(primer_frame_evidencia), frame)
        tiempos_etapa["guardado_evidencia_ms"] = round((time.perf_counter() - t_guardado) * 1000, 3)
        ultimo_frame = frame.copy()

        debe_renderizar = (
            frame_callback
            and (
                frames_procesados == 1
                or max(int(render_every_n_frames or 1), 1) <= 1
                or frames_procesados % max(int(render_every_n_frames or 1), 1) == 0
                or estado_frame.get("detecciones_validas", 0) > 0
                or estado_frame.get("mejor_recorte_placa")
            )
        )
        if debe_renderizar:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tiempo_transcurrido = max(time.perf_counter() - tiempo_inicio_total, 0.001)
            estado_callback = {
                "frame_actual": numero_frame_actual,
                "frames_procesados": frames_procesados,
                "start_frame": start_frame,
                "total_frames": total_frames,
                "fps": fps,
                "duracion_segundos": duracion,
                "rotacion": rotacion,
                "ancho": ancho_actual,
                "alto": alto_actual,
                "segundos_procesados": _calcular_segundos_procesados(frames_procesados, fps),
                "modo_procesamiento": _obtener_modo_procesamiento(max_frames),
                "modo_reproduccion": "Automatico",
                "velocidad_reproduccion": velocidad_reproduccion,
                "detecciones_frame": detecciones_frame,
                "detecciones_brutas": detecciones_brutas,
                "detecciones_validas": detecciones_validas,
                "motivos_rechazo": motivos_rechazo,
                "eventos_placa": eventos_placa,
                "placas_detectadas": eventos_placa,
                "estado_placa": estado_placa,
                "frames_desde_ultima_deteccion": estado_persistencia["frames_desde_ultima_deteccion"],
                "ultima_confianza": ultima_deteccion["confianza"] if ultima_deteccion else None,
                "ultima_deteccion": ultima_deteccion,
                "ultimo_recorte_placa": ultimo_recorte_placa,
                "ultimo_frame_deteccion": ultimo_frame_deteccion,
                "mejor_recorte_placa": estado_persistencia.get("ruta_mejor_recorte_placa"),
                "mejor_recorte_placa_info": estado_persistencia.get("mejor_recorte_placa_info"),
                "ultimos_candidatos_recorte": estado_persistencia.get("ultimo_candidatos_recorte", []),
                "distancia_lineas_m": distancia_lineas_m,
                "limite_velocidad_kmh": limite_velocidad_kmh,
                "posicion_linea_1": posicion_linea_1,
                "posicion_linea_2": posicion_linea_2,
                "frecuencia_deteccion": frecuencia_deteccion,
                "frames_saltados": max(frames_procesados - (frames_procesados // max(int(frecuencia_deteccion or 1), 1)), 0),
                "fps_procesamiento": round(frames_procesados / tiempo_transcurrido, 2),
                "tiempo_yolo_ms": tiempo_yolo_ms,
                "tiempo_lector_cnn_ms": estado_persistencia.get("ultimo_tiempo_lector_cnn_ms"),
                "resolucion_inferencia": resolucion_inferencia,
                "render_every_n_frames": render_every_n_frames,
                "frames_mostrados": frames_mostrados + 1,
                "frames_yolo_analizados": frames_yolo_analizados,
                "frames_cnn_ejecutados": estado_persistencia.get("frames_cnn_ejecutados", 0),
                "demo_fluido": demo_fluido,
                "guardar_debug": guardar_debug,
                "camera_width": camera_width,
                "camera_height": camera_height,
                "camera_fps": camera_fps,
                "tiempos_etapa": tiempos_etapa.copy(),
                "estado_persistencia": estado_persistencia,
                "evento_activo": estado_frame["evento_activo"],
                "evento_id": estado_frame["evento_id"],
                "mejor_confianza_evento": estado_frame["mejor_confianza_evento"],
                "frame_mejor_evento": estado_frame["frame_mejor_evento"],
                "frames_sin_deteccion": estado_frame["frames_sin_deteccion"],
                "ruta_mejor_recorte_evento": estado_frame["ruta_mejor_recorte_evento"],
                "velocidad": estado_frame["velocidad"],
            }
            try:
                t_render = time.perf_counter()
                frame_callback(frame_rgb, frames_procesados, estado_callback)
                tiempos_etapa["render_streamlit_ms"] = round((time.perf_counter() - t_render) * 1000, 3)
                frames_mostrados += 1
            except TypeError:
                frame_callback(frame_rgb, frames_procesados)
                frames_mostrados += 1

        if progreso_callback:
            if objetivo_frames and objetivo_frames > 0:
                if total_frames > 0 and max_frames == 0:
                    progreso_callback(min(numero_frame_actual / total_frames, 1.0))
                else:
                    progreso_callback(min(frames_procesados / objetivo_frames, 1.0))
            else:
                progreso_callback(0.0)

        delay = _calcular_delay_reproduccion(fps, velocidad_reproduccion)
        if max_display_fps and max_display_fps > 0:
            delay = max(delay, 1 / float(max_display_fps))
        if delay > 0:
            time.sleep(delay)

        if max_frames > 0 and frames_procesados >= max_frames:
            break

    captura.release()

    frame_final = start_frame + frames_procesados

    if estado_persistencia.get("buffer_recortes_placa"):
        _seleccionar_mejor_recorte_buffer(estado_persistencia)

    if estado_persistencia["evento_activo"]:
        _cerrar_evento_placa(estado_persistencia, frame_final)
        ruta_mejor_recorte_evento = estado_persistencia["ruta_mejor_recorte_evento"] or ruta_mejor_recorte_evento

    if ultimo_frame is not None:
        cv2.imwrite(str(ultimo_frame_evidencia), ultimo_frame)

    return {
        "estado": "finalizado",
        "mensaje_estado": f"Monitoreo desde {nombre_fuente.lower()} finalizado. Flujo visual listo; deteccion, OCR y velocidad real quedan para la siguiente etapa.",
        "frames_procesados": frames_procesados,
        "frame_actual": frame_final,
        "start_frame": start_frame,
        "estado_persistencia": estado_persistencia,
        "fps": fps,
        "ancho": ultimo_frame.shape[1] if ultimo_frame is not None else ancho,
        "alto": ultimo_frame.shape[0] if ultimo_frame is not None else alto,
        "total_frames": total_frames,
        "duracion_segundos": duracion,
        "segundos_procesados": _calcular_segundos_procesados(frame_final, fps),
        "modo_procesamiento": _obtener_modo_procesamiento(max_frames),
        "fuente": nombre_fuente,
        "rotacion": rotacion,
        "distancia_lineas_m": distancia_lineas_m,
        "posicion_linea_1": posicion_linea_1,
        "posicion_linea_2": posicion_linea_2,
        "limite_velocidad_kmh": limite_velocidad_kmh,
        "frecuencia_deteccion": frecuencia_deteccion,
        "frames_saltados": max(frames_procesados - (frames_procesados // max(int(frecuencia_deteccion or 1), 1)), 0),
        "fps_procesamiento": round(frames_procesados / max(time.perf_counter() - tiempo_inicio_total, 0.001), 2),
        "tiempo_yolo_ms": tiempo_yolo_ms,
        "tiempo_lector_cnn_ms": estado_persistencia.get("ultimo_tiempo_lector_cnn_ms"),
        "resolucion_inferencia": resolucion_inferencia,
        "render_every_n_frames": render_every_n_frames,
        "max_display_fps": max_display_fps,
        "frames_mostrados": frames_mostrados,
        "frames_yolo_analizados": frames_yolo_analizados,
        "frames_cnn_ejecutados": estado_persistencia.get("frames_cnn_ejecutados", 0),
        "demo_fluido": demo_fluido,
        "guardar_debug": guardar_debug,
        "camera_width": camera_width,
        "camera_height": camera_height,
        "camera_fps": camera_fps,
        "tiempos_etapa": tiempos_etapa,
        "primer_frame_evidencia": str(primer_frame_evidencia) if frames_procesados else None,
        "ultimo_frame_evidencia": str(ultimo_frame_evidencia) if frames_procesados else None,
        "modelo_detector_disponible": detector.model is not None,
        "mensaje_detector": mensaje_detector,
        "estado_placa": estado_placa,
        "detecciones_frame": detecciones_frame,
        "detecciones_brutas": detecciones_brutas,
        "detecciones_validas": detecciones_validas,
        "motivos_rechazo": motivos_rechazo,
        "eventos_placa": eventos_placa,
        "placas_detectadas": eventos_placa,
        "frames_desde_ultima_deteccion": estado_persistencia["frames_desde_ultima_deteccion"],
        "ultima_deteccion": ultima_deteccion,
        "ultimo_recorte_placa": ultimo_recorte_placa,
        "ultimo_frame_deteccion": ultimo_frame_deteccion,
        "mejor_recorte_placa": estado_persistencia.get("ruta_mejor_recorte_placa"),
        "mejor_recorte_placa_info": estado_persistencia.get("mejor_recorte_placa_info"),
        "ultimos_candidatos_recorte": estado_persistencia.get("ultimo_candidatos_recorte", []),
        "mejor_frame_recorte_placa": estado_persistencia.get("ruta_mejor_frame_recorte_placa"),
        "mejor_frame_bbox_placa": estado_persistencia.get("ruta_mejor_frame_bbox_placa"),
        "evento_activo": estado_persistencia["evento_activo"],
        "evento_id": estado_persistencia["evento_id"],
        "mejor_confianza_evento": estado_persistencia["mejor_confianza_evento"] if estado_persistencia["mejor_confianza_evento"] is not None else mejor_confianza_evento,
        "frame_mejor_evento": estado_persistencia["frame_mejor_evento"] or frame_mejor_evento,
        "frames_sin_deteccion": estado_persistencia["frames_sin_deteccion"],
        "ruta_mejor_recorte_evento": ruta_mejor_recorte_evento,
        "velocidad": estado_persistencia.get("speed_tracker").resumen() if estado_persistencia.get("speed_tracker") else _resumen_velocidad_vacio(distancia_lineas_m, fps),
    }


def _crear_estado_persistencia() -> dict:
    return {
        "ultima_bbox_valida": None,
        "ultima_confianza_valida": None,
        "frames_desde_ultima_deteccion": 0,
        "ultimo_recorte_placa": None,
        "ultimo_frame_deteccion": None,
        "eventos_placa": 0,
        "bbox_persistente_activa": False,
        "ultimo_frame_recorte": -GUARDAR_RECORTE_CADA_N_FRAMES,
        "evento_activo": False,
        "evento_id": 0,
        "mejor_confianza_evento": None,
        "mejor_bbox_evento": None,
        "mejor_frame_evento": None,
        "mejor_recorte_evento": None,
        "frame_inicio_evento": None,
        "frame_mejor_evento": None,
        "frame_ultimo_evento": None,
        "frames_sin_deteccion": 0,
        "ruta_mejor_frame_evento": None,
        "ruta_mejor_recorte_evento": None,
        "speed_tracker": None,
        "buffer_recortes_placa": [],
        "ultimo_candidatos_recorte": [],
        "mejor_recorte_placa": None,
        "mejor_recorte_placa_info": None,
        "ruta_mejor_recorte_placa": None,
        "ruta_mejor_frame_recorte_placa": None,
        "ruta_mejor_frame_bbox_placa": None,
        "ultimo_frame_mejor_recorte_placa": None,
        "ultimo_puntaje_recorte_placa": None,
        "ultimo_tiempo_lector_cnn_ms": None,
    }


def generar_video_demo_anotado(
    ruta_video: str,
    salida_dir: str = "reports/evidencias/video_anotado",
    distancia_lineas_m: float = 10.0,
    limite_velocidad_kmh: float = 30.0,
    posicion_linea_1: float = 0.45,
    posicion_linea_2: float = 0.65,
    frecuencia_deteccion: int = 15,
    conf_min: float = 0.45,
    persistencia_frames: int = 5,
    rotacion: str = "Sin rotacion",
    model_path: str | None = None,
    inference_size: int = 416,
    max_frames: int = 0,
    progreso_callback=None,
) -> dict:
    captura = cv2.VideoCapture(ruta_video)
    detector = _obtener_detector_cache(str(model_path) if model_path else None)
    salida = Path(salida_dir)
    salida.mkdir(parents=True, exist_ok=True)

    if not captura.isOpened():
        return {"estado": "error", "mensaje": "No se pudo abrir el video para generar demo."}

    fps, ancho, alto, total_frames, duracion = _leer_metadata_video(captura)
    fps_salida = fps if fps and fps > 0 else 24.0
    ok, frame_prueba = captura.read()
    if not ok:
        captura.release()
        return {"estado": "error", "mensaje": "No se pudo leer el primer frame."}
    frame_prueba = aplicar_rotacion(frame_prueba, rotacion)
    alto_out, ancho_out = frame_prueba.shape[:2]
    captura.set(cv2.CAP_PROP_POS_FRAMES, 0)

    recortes_dir = salida / "mejores_recortes"
    candidatos_dir = salida / "recortes_candidatos"
    debug_segmentacion_dir = salida / "debug_segmentacion"
    videos_dir = salida / "videos"
    for carpeta in [recortes_dir, candidatos_dir, debug_segmentacion_dir, videos_dir]:
        carpeta.mkdir(parents=True, exist_ok=True)
    timestamp_salida = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_video_salida = videos_dir / f"video_anotado_{timestamp_salida}.mp4"
    ruta_csv = salida / "detecciones_video.csv"
    ruta_json = salida / "resumen_video.json"
    ruta_reporte = salida / "reporte_video_anotado.md"
    writer_video = cv2.VideoWriter(str(ruta_video_salida), cv2.VideoWriter_fourcc(*"mp4v"), fps_salida, (ancho_out, alto_out))
    eventos_activos = []
    eventos_cerrados = []
    historial_frames = {}
    siguiente_evento_id = 1
    frames_procesados = 0
    frames_yolo = 0
    candidatos_totales = 0
    predicciones_cnn = 0
    tiempo_inicio = time.perf_counter()
    iou_min_evento = 0.25
    max_frames_sin_deteccion = max(int(persistencia_frames), 15)
    min_candidatos_para_leer = 3
    max_candidatos_por_evento = 30

    while captura.isOpened():
        ok, frame = captura.read()
        if not ok:
            break
        frames_procesados += 1
        if max_frames > 0 and frames_procesados > max_frames:
            break
        frame = aplicar_rotacion(frame, rotacion)
        frame_visual = frame.copy()
        _dibujar_marcas_monitoreo(frame_visual, distancia_lineas_m, limite_velocidad_kmh, "Video anotado", posicion_linea_1, posicion_linea_2)
        frame_info = {"frame": frame_visual, "detecciones": []}

        if frecuencia_deteccion > 0 and frames_procesados % frecuencia_deteccion == 0:
            frames_yolo += 1
            resultado_detector = detector.detectar_en_frame(frame, conf_min=conf_min, imgsz=inference_size)
            detecciones = resultado_detector.get("detecciones", [])
            for deteccion in detecciones:
                bbox = [int(v) for v in deteccion["bbox"]]
                recorte = deteccion.get("recorte_placa")
                confianza = float(deteccion.get("confianza", 0.0))
                metricas = calcular_puntaje_recorte_placa(frame, bbox, recorte, confianza)
                evento = _asignar_evento_video_anotado(
                    eventos_activos,
                    siguiente_evento_id,
                    bbox,
                    frames_procesados,
                    fps_salida,
                    iou_min_evento,
                )
                if evento["nuevo"]:
                    siguiente_evento_id += 1
                candidato = {
                    "frame_index": frames_procesados,
                    "timestamp": round(frames_procesados / fps_salida, 4),
                    "bbox": bbox,
                    "conf_yolo": confianza,
                    "recorte": recorte.copy() if recorte is not None else None,
                    **metricas,
                }
                evento["candidatos"].append(candidato)
                evento["candidatos"] = sorted(evento["candidatos"], key=lambda item: item.get("puntaje_total", 0.0), reverse=True)[:max_candidatos_por_evento]
                evento["ultimo_bbox"] = bbox
                evento["frame_fin"] = frames_procesados
                evento["timestamp_fin"] = round(frames_procesados / fps_salida, 4)
                evento["frames_sin_deteccion"] = 0
                candidatos_totales += 1
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_visual, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(frame_visual, f"Placa detectada | analizando {confianza:.2f}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
                frame_info["detecciones"].append({"evento_id": evento["event_id"], "bbox": bbox})

        for evento in list(eventos_activos):
            if evento["frame_fin"] != frames_procesados:
                evento["frames_sin_deteccion"] += 1
            if evento["frames_sin_deteccion"] > max_frames_sin_deteccion:
                _cerrar_evento_video_anotado(evento, recortes_dir, candidatos_dir, min_candidatos_para_leer)
                predicciones_cnn += 1 if evento.get("texto_corregido_formato") else 0
                eventos_cerrados.append(evento)
                eventos_activos.remove(evento)

        historial_frames[frames_procesados] = frame_info
        writer_video.write(frame_visual)
        if progreso_callback:
            total = total_frames if total_frames > 0 else max_frames
            progreso_callback(
                {
                    "frame": frames_procesados,
                    "total_frames": total,
                    "porcentaje": min(frames_procesados / total, 1.0) if total else 0.0,
                    "estado": "Analizando video",
                }
            )

    captura.release()
    for evento in list(eventos_activos):
        _cerrar_evento_video_anotado(evento, recortes_dir, candidatos_dir, min_candidatos_para_leer)
        predicciones_cnn += 1 if evento.get("texto_corregido_formato") else 0
        eventos_cerrados.append(evento)
        eventos_activos.remove(evento)
    writer_video.release()

    _propagar_texto_eventos_video(ruta_video_salida, historial_frames, eventos_cerrados, fps_salida, (ancho_out, alto_out))

    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        campos = [
            "event_id",
            "frame_inicio",
            "frame_fin",
            "timestamp_inicio",
            "timestamp_fin",
            "frame_mejor_recorte",
            "bbox_mejor_recorte",
            "confianza_yolo",
            "puntaje_recorte",
            "sharpness",
            "aspect_ratio",
            "area_relativa",
            "texto_reconocido_crudo",
            "texto_corregido_formato",
            "confianza_cnn_caracteres",
            "formato_valido",
            "cantidad_caracteres_segmentados",
            "ruta_mejor_recorte",
        ]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows([_fila_evento_video_anotado(evento, ruta_video_salida) for evento in eventos_cerrados])

    ultimo_evento = eventos_cerrados[-1] if eventos_cerrados else {}
    mejor_global = max(
        (evento for evento in eventos_cerrados if evento.get("mejor_candidato")),
        key=lambda item: item["mejor_candidato"].get("puntaje_total", 0.0),
        default={},
    )

    resumen = {
        "estado": "ok",
        "video_original": str(ruta_video),
        "video_anotado": str(ruta_video_salida),
        "ruta_video": str(ruta_video_salida),
        "ruta_csv": str(ruta_csv),
        "ruta_json": str(ruta_json),
        "ruta_reporte": str(ruta_reporte),
        "ultimo_recorte_placa": ultimo_evento.get("ruta_mejor_recorte"),
        "texto_reconocido_crudo": ultimo_evento.get("texto_crudo", ""),
        "texto_corregido_formato": ultimo_evento.get("texto_corregido_formato", ""),
        "confianza_cnn_caracteres": ultimo_evento.get("confianza_cnn_caracteres"),
        "confianza_yolo": (ultimo_evento.get("mejor_candidato") or {}).get("conf_yolo"),
        "eventos": [_resumen_evento_video_anotado(evento) for evento in eventos_cerrados],
        "frames_procesados": frames_procesados,
        "total_frames": total_frames,
        "frames_yolo_analizados": frames_yolo,
        "eventos_placa": len(eventos_cerrados),
        "eventos_detectados": len(eventos_cerrados),
        "candidatos_totales": candidatos_totales,
        "mejores_recortes_seleccionados": len([e for e in eventos_cerrados if e.get("ruta_mejor_recorte")]),
        "mejores_recortes": [e.get("ruta_mejor_recorte") for e in eventos_cerrados if e.get("ruta_mejor_recorte")],
        "predicciones_cnn": predicciones_cnn,
        "placas_validas": len([e for e in eventos_cerrados if e.get("formato_valido")]),
        "placas_dudosas": len([e for e in eventos_cerrados if e.get("texto_corregido_formato") and not e.get("formato_valido")]),
        "detecciones": candidatos_totales,
        "mejor_recorte_global": (mejor_global.get("ruta_mejor_recorte") if mejor_global else None),
        "fps_salida": fps_salida,
        "duracion_original": duracion,
        "tiempo_total_s": round(time.perf_counter() - tiempo_inicio, 2),
    }
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    ruta_reporte.write_text(
        "\n".join(
            [
                "# Reporte de procesamiento de video anotado",
                "",
                f"Video fuente: {ruta_video}",
                f"Video anotado: {ruta_video_salida}",
                f"CSV detecciones: {ruta_csv}",
                f"Frames procesados: {frames_procesados}",
                f"Frames analizados por YOLO: {frames_yolo}",
                f"Eventos detectados: {len(eventos_cerrados)}",
                f"Recortes candidatos: {candidatos_totales}",
                f"Mejores recortes seleccionados: {len([e for e in eventos_cerrados if e.get('ruta_mejor_recorte')])}",
                f"Predicciones CNN realizadas: {predicciones_cnn}",
                f"Ultima placa cruda: {ultimo_evento.get('texto_crudo') or 'Pendiente'}",
                f"Ultima placa corregida: {ultimo_evento.get('texto_corregido_formato') or 'Pendiente'}",
                f"Tiempo total: {resumen['tiempo_total_s']} s",
                "",
                "El analisis se ve directamente en el video anotado: cada evento muestra bounding box, estado de analisis y, cuando existe lectura, texto reconocido con confianza CNN o aviso de formato dudoso.",
                "El video fue procesado en multiples frames. Las detecciones se agruparon por evento de placa usando IoU/cercania de bbox, se guardaron multiples candidatos por evento y la CNN se ejecuto una vez sobre el mejor recorte seleccionado.",
            ]
        ),
        encoding="utf-8",
    )
    return resumen


def _asignar_evento_video_anotado(eventos: list[dict], siguiente_id: int, bbox: list[int], frame_index: int, fps: float, iou_min: float) -> dict:
    mejor_evento = None
    mejor_iou = 0.0
    for evento in eventos:
        iou = _iou_xyxy(bbox, evento.get("ultimo_bbox") or bbox)
        if iou > mejor_iou:
            mejor_iou = iou
            mejor_evento = evento
    if mejor_evento is not None and mejor_iou >= iou_min:
        mejor_evento["nuevo"] = False
        return mejor_evento

    evento = {
        "event_id": siguiente_id,
        "nuevo": True,
        "frame_inicio": frame_index,
        "frame_fin": frame_index,
        "timestamp_inicio": round(frame_index / fps, 4) if fps else 0.0,
        "timestamp_fin": round(frame_index / fps, 4) if fps else 0.0,
        "ultimo_bbox": bbox,
        "frames_sin_deteccion": 0,
        "candidatos": [],
        "mejor_candidato": None,
        "ruta_mejor_recorte": None,
        "texto_crudo": "",
        "texto_corregido_formato": "",
        "confianza_cnn_caracteres": None,
        "formato_valido": False,
        "cantidad_caracteres_segmentados": 0,
    }
    eventos.append(evento)
    return evento


def _cerrar_evento_video_anotado(evento: dict, recortes_dir: Path, candidatos_dir: Path, min_candidatos: int) -> None:
    candidatos = evento.get("candidatos") or []
    if not candidatos:
        return
    mejor = max(candidatos, key=lambda item: item.get("puntaje_total", 0.0))
    evento["mejor_candidato"] = mejor
    event_id = int(evento["event_id"])

    for idx, candidato in enumerate(sorted(candidatos, key=lambda item: item.get("puntaje_total", 0.0), reverse=True)[:5], start=1):
        recorte = candidato.get("recorte")
        if recorte is not None:
            cv2.imwrite(str(candidatos_dir / f"evento_{event_id:04d}_candidato_{idx:02d}.jpg"), recorte)

    ruta_recorte = recortes_dir / f"evento_{event_id:04d}_mejor_recorte.jpg"
    if mejor.get("recorte") is not None:
        cv2.imwrite(str(ruta_recorte), mejor["recorte"])
        evento["ruta_mejor_recorte"] = str(ruta_recorte)

    if len(candidatos) >= int(min_candidatos) and evento.get("ruta_mejor_recorte"):
        try:
            resultado = leer_placa_desde_recorte(evento["ruta_mejor_recorte"])
            evento["texto_crudo"] = resultado.get("texto_detectado_crudo") or ""
            evento["texto_corregido_formato"] = resultado.get("texto_postprocesado") or evento["texto_crudo"]
            evento["confianza_cnn_caracteres"] = resultado.get("confianza_promedio")
            evento["formato_valido"] = bool((resultado.get("formato") or {}).get("valido"))
            evento["cantidad_caracteres_segmentados"] = int(resultado.get("cantidad_caracteres_segmentados", 0) or 0)
            evento["estado"] = "leido_con_cnn" if evento["texto_corregido_formato"] else "sin_texto"
        except Exception as exc:
            evento["estado"] = f"error_lector_cnn: {exc}"
    else:
        evento["estado"] = "candidatos_insuficientes"


def _propagar_texto_eventos_video(ruta_video: Path, historial_frames: dict, eventos: list[dict], fps: float, size: tuple[int, int]) -> None:
    if not historial_frames:
        return
    ruta_tmp = ruta_video.with_name(f"{ruta_video.stem}_tmp{ruta_video.suffix}")
    writer = cv2.VideoWriter(str(ruta_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    eventos_por_id = {int(evento["event_id"]): evento for evento in eventos}
    for frame_index in sorted(historial_frames):
        info = historial_frames[frame_index]
        frame = info["frame"]
        for det in info.get("detecciones", []):
            evento = eventos_por_id.get(int(det["evento_id"]))
            if not evento:
                continue
            texto = evento.get("texto_corregido_formato")
            conf_cnn = evento.get("confianza_cnn_caracteres")
            if texto:
                etiqueta = f"{texto} | CNN {conf_cnn:.2f}" if conf_cnn is not None else texto
                if not evento.get("formato_valido"):
                    etiqueta = f"{texto} | formato dudoso"
            else:
                conf = (evento.get("mejor_candidato") or {}).get("conf_yolo")
                etiqueta = f"Placa detectada | analizando {conf:.2f}" if conf is not None else "Placa detectada | analizando"
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
            y_text = max(25, y1 - 8)
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 6), (0, 0, 0), -1)
            cv2.putText(frame, etiqueta, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(frame)
    writer.release()
    if ruta_tmp.exists():
        ruta_tmp.replace(ruta_video)


def _fila_evento_video_anotado(evento: dict, ruta_video: Path) -> dict:
    mejor = evento.get("mejor_candidato") or {}
    return {
        "event_id": evento.get("event_id"),
        "frame_inicio": evento.get("frame_inicio"),
        "frame_fin": evento.get("frame_fin"),
        "timestamp_inicio": evento.get("timestamp_inicio"),
        "timestamp_fin": evento.get("timestamp_fin"),
        "frame_mejor_recorte": mejor.get("frame_index"),
        "bbox_mejor_recorte": json.dumps(mejor.get("bbox")),
        "confianza_yolo": mejor.get("conf_yolo"),
        "puntaje_recorte": mejor.get("puntaje_total"),
        "sharpness": mejor.get("nitidez"),
        "aspect_ratio": mejor.get("aspect_ratio"),
        "area_relativa": mejor.get("area_relativa"),
        "texto_reconocido_crudo": evento.get("texto_crudo"),
        "texto_corregido_formato": evento.get("texto_corregido_formato"),
        "confianza_cnn_caracteres": evento.get("confianza_cnn_caracteres"),
        "formato_valido": evento.get("formato_valido"),
        "cantidad_caracteres_segmentados": evento.get("cantidad_caracteres_segmentados"),
        "ruta_mejor_recorte": evento.get("ruta_mejor_recorte"),
    }


def _resumen_evento_video_anotado(evento: dict) -> dict:
    fila = _fila_evento_video_anotado(evento, Path(""))
    fila.pop("ruta_video_anotado", None)
    fila["estado"] = evento.get("estado")
    fila["candidatos"] = len(evento.get("candidatos") or [])
    return fila


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _config_recorte_placa(config: dict | None) -> dict:
    salida = DEFAULT_PLATE_CROP_SELECTION.copy()
    if config:
        salida.update({k: v for k, v in config.items() if v is not None})
    salida["buffer_frames"] = max(int(salida.get("buffer_frames", 15)), 1)
    return salida


def calcular_puntaje_recorte_placa(frame, bbox: list[int], recorte, conf_yolo: float, config: dict | None = None) -> dict:
    cfg = _config_recorte_placa(config)
    alto_frame, ancho_frame = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    ancho_bbox = max(x2 - x1, 1)
    alto_bbox = max(y2 - y1, 1)
    area_relativa = (ancho_bbox * alto_bbox) / max(ancho_frame * alto_frame, 1)
    aspect_ratio = ancho_bbox / alto_bbox
    cerca_borde = (
        x1 <= int(cfg["border_margin_px"])
        or y1 <= int(cfg["border_margin_px"])
        or x2 >= ancho_frame - int(cfg["border_margin_px"])
        or y2 >= alto_frame - int(cfg["border_margin_px"])
    )

    if recorte is None or recorte.size == 0:
        nitidez = 0.0
        contraste = 0.0
        alto_recorte = 0
        ancho_recorte = 0
    else:
        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if len(recorte.shape) == 3 else recorte
        nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
        contraste = float(gris.std())
        alto_recorte, ancho_recorte = recorte.shape[:2]

    centro_aspect = (float(cfg["min_aspect_ratio"]) + float(cfg["max_aspect_ratio"])) / 2
    rango_aspect = max((float(cfg["max_aspect_ratio"]) - float(cfg["min_aspect_ratio"])) / 2, 0.1)
    score_aspect = max(0.0, 1.0 - abs(aspect_ratio - centro_aspect) / rango_aspect)

    min_area = float(cfg["min_area_relative"])
    max_area = float(cfg["max_area_relative"])
    centro_area = (min_area + max_area) / 2
    rango_area = max((max_area - min_area) / 2, 0.0001)
    score_area = max(0.0, 1.0 - abs(area_relativa - centro_area) / rango_area)

    score_nitidez = min(nitidez / max(float(cfg["min_sharpness"]) * 4, 1.0), 1.0)
    score_contraste = min(contraste / 80.0, 1.0)
    score_tamano = min((ancho_recorte * alto_recorte) / (120 * 35), 1.0)
    penalizacion = 0.0
    if cerca_borde:
        penalizacion += 0.25
    if aspect_ratio < float(cfg["min_aspect_ratio"]) or aspect_ratio > float(cfg["max_aspect_ratio"]):
        penalizacion += 0.35
    if area_relativa < min_area or area_relativa > max_area:
        penalizacion += 0.30
    if nitidez < float(cfg["min_sharpness"]):
        penalizacion += 0.25
    if ancho_recorte < 50 or alto_recorte < 18:
        penalizacion += 0.25

    puntaje = (
        0.35 * float(conf_yolo)
        + 0.25 * score_nitidez
        + 0.18 * score_aspect
        + 0.12 * score_area
        + 0.05 * score_tamano
        + 0.05 * score_contraste
        - penalizacion
    )
    return {
        "puntaje_total": round(max(puntaje, 0.0), 4),
        "area_relativa": round(area_relativa, 6),
        "aspect_ratio": round(aspect_ratio, 4),
        "nitidez": round(nitidez, 4),
        "contraste": round(contraste, 4),
        "cerca_borde": bool(cerca_borde),
        "ancho_recorte": int(ancho_recorte),
        "alto_recorte": int(alto_recorte),
    }


def _registrar_candidato_recorte(
    estado: dict,
    frame_limpio,
    frame_visual,
    bbox: list[int],
    recorte,
    conf_yolo: float,
    numero_frame: int,
    fps: float,
    config: dict | None,
) -> None:
    cfg = _config_recorte_placa(config)
    if not cfg.get("enabled", True) or recorte is None or recorte.size == 0:
        return
    estado["_plate_crop_selection_cfg"] = cfg
    metricas = calcular_puntaje_recorte_placa(frame_limpio, bbox, recorte, conf_yolo, cfg)
    candidato = {
        "frame_index": int(numero_frame),
        "timestamp": round(numero_frame / fps, 4) if fps > 0 else 0.0,
        "bbox": [int(v) for v in bbox],
        "conf_yolo": round(float(conf_yolo), 4),
        "recorte": recorte.copy(),
        "frame_limpio": frame_limpio.copy(),
        "frame_bbox": frame_visual.copy(),
        **metricas,
    }
    estado["buffer_recortes_placa"].append(candidato)
    estado["ultimo_candidatos_recorte"] = _serializar_candidatos_recorte(estado["buffer_recortes_placa"])
    if len(estado["buffer_recortes_placa"]) >= int(cfg["buffer_frames"]):
        _seleccionar_mejor_recorte_buffer(estado)


def _serializar_candidatos_recorte(candidatos: list[dict]) -> list[dict]:
    campos = [
        "frame_index",
        "timestamp",
        "conf_yolo",
        "bbox",
        "area_relativa",
        "aspect_ratio",
        "nitidez",
        "cerca_borde",
        "ancho_recorte",
        "alto_recorte",
        "puntaje_total",
    ]
    return [{campo: c.get(campo) for campo in campos} for c in candidatos[-20:]]


def _seleccionar_mejor_recorte_buffer(estado: dict) -> None:
    candidatos = estado.get("buffer_recortes_placa") or []
    if not candidatos:
        return

    mejor = max(candidatos, key=lambda item: item.get("puntaje_total", 0.0))
    cfg = _config_recorte_placa(estado.get("_plate_crop_selection_cfg"))
    frame_idx = int(mejor["frame_index"])
    ultimo_frame = estado.get("ultimo_frame_mejor_recorte_placa")
    ultimo_puntaje = estado.get("ultimo_puntaje_recorte_placa")
    cooldown_frames = int(cfg.get("cooldown_frames", 45))
    mejora_minima = float(cfg.get("min_score_improvement", 0.03))
    dentro_cooldown = ultimo_frame is not None and frame_idx - int(ultimo_frame) < cooldown_frames
    mejora_insuficiente = ultimo_puntaje is not None and float(mejor.get("puntaje_total", 0.0)) < float(ultimo_puntaje) + mejora_minima
    if dentro_cooldown and mejora_insuficiente:
        estado["ultimo_candidatos_recorte"] = _serializar_candidatos_recorte(candidatos)
        estado["buffer_recortes_placa"] = []
        return

    salida_dir = Path("reports") / "evidencias" / "mejores_recortes_placa"
    salida_dir.mkdir(parents=True, exist_ok=True)
    ruta_frame = salida_dir / f"mejor_frame_{frame_idx:06d}.jpg"
    ruta_bbox = salida_dir / f"mejor_frame_bbox_{frame_idx:06d}.jpg"
    ruta_recorte = salida_dir / f"mejor_recorte_{frame_idx:06d}.jpg"
    cv2.imwrite(str(ruta_frame), mejor["frame_limpio"])
    cv2.imwrite(str(ruta_bbox), mejor["frame_bbox"])
    cv2.imwrite(str(ruta_recorte), mejor["recorte"])

    csv_path = salida_dir / "metricas_mejores_recortes.csv"
    existe = csv_path.exists()
    campos = [
        "frame_index",
        "timestamp",
        "conf_yolo",
        "bbox",
        "area_relativa",
        "aspect_ratio",
        "nitidez",
        "cerca_borde",
        "ancho_recorte",
        "alto_recorte",
        "puntaje_total",
        "fue_seleccionado",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        if not existe:
            writer.writeheader()
        for candidato in candidatos:
            writer.writerow(
                {
                    "frame_index": candidato["frame_index"],
                    "timestamp": candidato["timestamp"],
                    "conf_yolo": candidato["conf_yolo"],
                    "bbox": json.dumps(candidato["bbox"]),
                    "area_relativa": candidato["area_relativa"],
                    "aspect_ratio": candidato["aspect_ratio"],
                    "nitidez": candidato["nitidez"],
                    "cerca_borde": candidato["cerca_borde"],
                    "ancho_recorte": candidato["ancho_recorte"],
                    "alto_recorte": candidato["alto_recorte"],
                    "puntaje_total": candidato["puntaje_total"],
                    "fue_seleccionado": candidato is mejor,
                }
            )

    info = {k: mejor.get(k) for k in campos if k != "fue_seleccionado"}
    info.update(
        {
            "ruta_frame": str(ruta_frame),
            "ruta_frame_bbox": str(ruta_bbox),
            "ruta_recorte": str(ruta_recorte),
            "motivo": "Mayor puntaje total dentro de la ventana temporal.",
        }
    )
    estado["mejor_recorte_placa"] = mejor["recorte"].copy()
    estado["mejor_recorte_placa_info"] = info
    estado["ruta_mejor_recorte_placa"] = str(ruta_recorte)
    estado["ruta_mejor_frame_recorte_placa"] = str(ruta_frame)
    estado["ruta_mejor_frame_bbox_placa"] = str(ruta_bbox)
    estado["ultimo_frame_mejor_recorte_placa"] = frame_idx
    estado["ultimo_puntaje_recorte_placa"] = float(mejor.get("puntaje_total", 0.0))
    estado["ultimo_candidatos_recorte"] = _serializar_candidatos_recorte(candidatos)
    estado["buffer_recortes_placa"] = []


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
        "Rapida (sin espera)": 0,
        "Rapida": 0,
        "Normal": 1,
    }
    return delay_base * factores.get(velocidad_reproduccion, 0)


def aplicar_rotacion(frame, rotacion: str):
    opcion = (rotacion or "Sin rotación").strip().lower()
    if "90" in opcion and "derecha" in opcion:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if "90" in opcion and "izquierda" in opcion:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "180" in opcion:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def _procesar_frame_monitoreo(
    frame_original,
    detector: PlateDetector,
    numero_frame: int,
    distancia_lineas_m: float,
    limite_velocidad_kmh: float,
    posicion_linea_1: float,
    posicion_linea_2: float,
    nombre_fuente: str,
    frecuencia_deteccion: int,
    evidencia_dir: Path,
    placas_dir: Path,
    estado_persistencia: dict,
    conf_min: float,
    persistencia_frames: int,
    fps: float,
    plate_crop_selection: dict | None = None,
    inference_size: int = 640,
    guardar_debug: bool = False,
) -> dict:
    frame_limpio = frame_original.copy()
    frame_visual = frame_original.copy()
    mensaje_detector = detector.estado
    ultima_deteccion = None
    ultimo_recorte_placa = None
    ultimo_frame_deteccion = None
    detecciones_frame = 0
    detecciones_brutas = 0
    motivos_rechazo = ""
    eventos_placa = estado_persistencia["eventos_placa"]
    estado_placa = "Pendiente"

    detecciones_validas = []
    tiempo_yolo_ms = 0.0
    resolucion_inferencia = f"{frame_limpio.shape[1]}x{frame_limpio.shape[0]}"
    yolo_ejecutado = False
    velocidad = _obtener_speed_tracker(
        estado_persistencia,
        frame_visual.shape[0],
        distancia_lineas_m,
        fps,
        posicion_linea_1,
        posicion_linea_2,
    ).resumen()
    if frecuencia_deteccion > 0 and numero_frame % frecuencia_deteccion == 0:
        yolo_ejecutado = True
        resultado_detector = detector.detectar_en_frame(frame_limpio, conf_min=conf_min, imgsz=inference_size)
        mensaje_detector = resultado_detector["mensaje"]
        detecciones_brutas = resultado_detector.get("detecciones_brutas", 0)
        tiempo_yolo_ms = float(resultado_detector.get("tiempo_yolo_ms", 0.0) or 0.0)
        resolucion_inferencia = resultado_detector.get("resolucion_inferencia", resolucion_inferencia)
        if guardar_debug:
            _guardar_debug_evento_detecciones(numero_frame, resultado_detector.get("debug_detecciones", []))
        motivos_rechazo = _resumir_motivos_rechazo(resultado_detector.get("debug_detecciones", []))
        detecciones_validas = _filtrar_detecciones_zona_superior(
            resultado_detector["detecciones"],
            frame_limpio.shape[0],
        )

    _dibujar_marcas_monitoreo(frame_visual, distancia_lineas_m, limite_velocidad_kmh, nombre_fuente, posicion_linea_1, posicion_linea_2)

    if detecciones_validas:
        detecciones_frame = len(detecciones_validas)
        deteccion = max(detecciones_validas, key=lambda item: item["confianza"])
        bbox_nueva = deteccion["bbox"]
        bbox_anterior = estado_persistencia["ultima_bbox_valida"]
        bbox_suavizada = _suavizar_bbox(bbox_nueva, bbox_anterior)

        if not estado_persistencia["bbox_persistente_activa"]:
            eventos_placa += 1
            _iniciar_evento_placa(estado_persistencia, eventos_placa, numero_frame)

        estado_persistencia["ultima_bbox_valida"] = bbox_suavizada
        estado_persistencia["ultima_confianza_valida"] = float(deteccion["confianza"])
        estado_persistencia["frames_desde_ultima_deteccion"] = 0
        estado_persistencia["frames_sin_deteccion"] = 0
        estado_persistencia["frame_ultimo_evento"] = numero_frame
        estado_persistencia["eventos_placa"] = eventos_placa
        estado_persistencia["bbox_persistente_activa"] = True
        estado_placa = "Detectada"

        x1, y1, x2, y2 = bbox_suavizada
        confianza = float(deteccion["confianza"])
        _dibujar_bbox_placa(frame_visual, bbox_suavizada, f"placa {confianza:.2f}", (0, 180, 0))
        _registrar_candidato_recorte(
            estado_persistencia,
            frame_limpio,
            frame_visual,
            bbox_suavizada,
            deteccion["recorte_placa"],
            confianza,
            numero_frame,
            fps,
            plate_crop_selection,
        )
        _actualizar_mejor_evento_placa(
            estado_persistencia,
            confianza,
            bbox_suavizada,
            frame_visual,
            deteccion["recorte_placa"],
            numero_frame,
        )
        velocidad = estado_persistencia["speed_tracker"].actualizar(bbox_suavizada, numero_frame)
        _dibujar_centro_placa(frame_visual, bbox_suavizada)
        _guardar_evidencia_velocidad(frame_visual, velocidad)

        if numero_frame - estado_persistencia["ultimo_frame_recorte"] >= GUARDAR_RECORTE_CADA_N_FRAMES:
            recorte_path = placas_dir / f"placa_{numero_frame:06d}_{eventos_placa:03d}.jpg"
            cv2.imwrite(str(recorte_path), deteccion["recorte_placa"])
            ultimo_recorte_placa = str(recorte_path)
            estado_persistencia["ultimo_recorte_placa"] = str(recorte_path)
            estado_persistencia["ultimo_frame_recorte"] = numero_frame
        else:
            ultimo_recorte_placa = estado_persistencia["ultimo_recorte_placa"]

        ultima_deteccion = {
            "bbox": bbox_suavizada,
            "confianza": confianza,
            "recorte_placa": ultimo_recorte_placa,
            "frame_deteccion": estado_persistencia.get("ultimo_frame_deteccion"),
        }

        if ultimo_recorte_placa and guardar_debug:
            frame_deteccion_path = placas_dir / f"frame_deteccion_{numero_frame:06d}.jpg"
            cv2.imwrite(str(frame_deteccion_path), frame_visual)
            ultimo_frame_deteccion = str(frame_deteccion_path)
            estado_persistencia["ultimo_frame_deteccion"] = ultimo_frame_deteccion
            cv2.imwrite(str(evidencia_dir / "ultimo_frame_con_deteccion.jpg"), frame_visual)
            ultima_deteccion["frame_deteccion"] = ultimo_frame_deteccion
        else:
            ultimo_frame_deteccion = estado_persistencia["ultimo_frame_deteccion"]
    else:
        estado_persistencia["frames_desde_ultima_deteccion"] += 1
        estado_persistencia["frames_sin_deteccion"] += 1
        if (
            estado_persistencia["ultima_bbox_valida"] is not None
            and estado_persistencia["frames_desde_ultima_deteccion"] <= persistencia_frames
        ):
            estado_placa = "Mantenida"
            confianza = estado_persistencia["ultima_confianza_valida"] or 0.0
            _dibujar_bbox_placa(
                frame_visual,
                estado_persistencia["ultima_bbox_valida"],
                f"placa mantenida {confianza:.2f}",
                (0, 220, 220),
            )
            ultima_deteccion = {
                "bbox": estado_persistencia["ultima_bbox_valida"],
                "confianza": confianza,
                "recorte_placa": estado_persistencia["ultimo_recorte_placa"],
                "frame_deteccion": estado_persistencia["ultimo_frame_deteccion"],
            }
            ultimo_recorte_placa = estado_persistencia["ultimo_recorte_placa"]
            ultimo_frame_deteccion = estado_persistencia["ultimo_frame_deteccion"]
        else:
            if estado_persistencia["evento_activo"] and estado_persistencia["frames_sin_deteccion"] > persistencia_frames:
                _cerrar_evento_placa(estado_persistencia, numero_frame)
            estado_persistencia["bbox_persistente_activa"] = False
            if frecuencia_deteccion > 0 and numero_frame % frecuencia_deteccion == 0 and detector.model is not None:
                mensaje_detector = "No se detecto placa valida fuera de la zona superior excluida."

    _dibujar_info_velocidad(frame_visual, velocidad)

    return {
        "frame_visual": frame_visual,
        "mensaje_detector": mensaje_detector,
        "estado_placa": estado_placa,
        "detecciones_frame": detecciones_frame,
        "detecciones_brutas": detecciones_brutas,
        "detecciones_validas": detecciones_frame,
        "motivos_rechazo": motivos_rechazo,
        "tiempo_yolo_ms": tiempo_yolo_ms,
        "resolucion_inferencia": resolucion_inferencia,
        "yolo_ejecutado": yolo_ejecutado,
        "eventos_placa": eventos_placa,
        "placas_detectadas": eventos_placa,
        "frames_desde_ultima_deteccion": estado_persistencia["frames_desde_ultima_deteccion"],
        "ultima_deteccion": ultima_deteccion,
        "ultimo_recorte_placa": ultimo_recorte_placa,
        "ultimo_frame_deteccion": ultimo_frame_deteccion,
        "mejor_recorte_placa": estado_persistencia.get("ruta_mejor_recorte_placa"),
        "mejor_recorte_placa_info": estado_persistencia.get("mejor_recorte_placa_info"),
        "ultimos_candidatos_recorte": estado_persistencia.get("ultimo_candidatos_recorte", []),
        "estado_persistencia": estado_persistencia,
        "evento_activo": estado_persistencia["evento_activo"],
        "evento_id": estado_persistencia["evento_id"],
        "mejor_confianza_evento": estado_persistencia["mejor_confianza_evento"],
        "frame_mejor_evento": estado_persistencia["frame_mejor_evento"],
        "frames_sin_deteccion": estado_persistencia["frames_sin_deteccion"],
        "ruta_mejor_recorte_evento": estado_persistencia["ruta_mejor_recorte_evento"],
        "velocidad": velocidad,
    }


def _iniciar_evento_placa(estado: dict, evento_id: int, numero_frame: int) -> None:
    estado["evento_activo"] = True
    estado["evento_id"] = evento_id
    estado["mejor_confianza_evento"] = None
    estado["mejor_bbox_evento"] = None
    estado["mejor_frame_evento"] = None
    estado["mejor_recorte_evento"] = None
    estado["frame_inicio_evento"] = numero_frame
    estado["frame_mejor_evento"] = None
    estado["frame_ultimo_evento"] = numero_frame
    estado["frames_sin_deteccion"] = 0
    estado["ruta_mejor_frame_evento"] = None
    estado["ruta_mejor_recorte_evento"] = None


def _actualizar_mejor_evento_placa(
    estado: dict,
    confianza: float,
    bbox: list[int],
    frame_visual,
    recorte_placa,
    numero_frame: int,
) -> None:
    if not estado["evento_activo"]:
        _iniciar_evento_placa(estado, estado["eventos_placa"], numero_frame)

    mejor_confianza = estado["mejor_confianza_evento"]
    if mejor_confianza is None or confianza > mejor_confianza:
        estado["mejor_confianza_evento"] = confianza
        estado["mejor_bbox_evento"] = bbox
        estado["mejor_frame_evento"] = frame_visual.copy()
        estado["mejor_recorte_evento"] = recorte_placa.copy()
        estado["frame_mejor_evento"] = numero_frame


def _cerrar_evento_placa(estado: dict, frame_fin: int) -> None:
    if not estado["evento_activo"] or estado["mejor_frame_evento"] is None:
        estado["evento_activo"] = False
        return

    eventos_dir = Path("reports") / "evidencias" / "eventos_placa"
    eventos_dir.mkdir(parents=True, exist_ok=True)
    evento_id = int(estado["evento_id"])
    ruta_frame = eventos_dir / f"evento_{evento_id:04d}_mejor_frame.jpg"
    ruta_recorte = eventos_dir / f"evento_{evento_id:04d}_mejor_recorte.jpg"
    cv2.imwrite(str(ruta_frame), estado["mejor_frame_evento"])
    cv2.imwrite(str(ruta_recorte), estado["mejor_recorte_evento"])
    estado["ruta_mejor_frame_evento"] = str(ruta_frame)
    estado["ruta_mejor_recorte_evento"] = str(ruta_recorte)

    csv_path = eventos_dir / "eventos_placa.csv"
    existe = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as archivo:
        campos = [
            "evento_id",
            "frame_inicio",
            "frame_mejor",
            "frame_fin",
            "mejor_confianza",
            "mejor_bbox",
            "ruta_mejor_frame",
            "ruta_mejor_recorte",
            "estado_evento",
        ]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        if not existe:
            writer.writeheader()
        writer.writerow(
            {
                "evento_id": evento_id,
                "frame_inicio": estado["frame_inicio_evento"],
                "frame_mejor": estado["frame_mejor_evento"],
                "frame_fin": frame_fin,
                "mejor_confianza": f"{estado['mejor_confianza_evento']:.4f}",
                "mejor_bbox": estado["mejor_bbox_evento"],
                "ruta_mejor_frame": str(ruta_frame),
                "ruta_mejor_recorte": str(ruta_recorte),
                "estado_evento": "cerrado",
            }
        )

    estado["evento_activo"] = False


def _suavizar_bbox(bbox_nueva: list[int], bbox_anterior: list[int] | None) -> list[int]:
    if bbox_anterior is None:
        return [int(valor) for valor in bbox_nueva]
    return [
        int(BBOX_SUAVIZADO_ALPHA * nuevo + (1 - BBOX_SUAVIZADO_ALPHA) * anterior)
        for nuevo, anterior in zip(bbox_nueva, bbox_anterior)
    ]


def _dibujar_bbox_placa(frame, bbox: list[int], etiqueta: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, etiqueta, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _obtener_speed_tracker(
    estado: dict,
    alto_frame: int,
    distancia_metros: float,
    fps: float,
    posicion_linea_1: float,
    posicion_linea_2: float,
) -> SpeedTracker:
    linea_1_y = int(alto_frame * posicion_linea_1)
    linea_2_y = int(alto_frame * posicion_linea_2)
    tracker = estado.get("speed_tracker")
    if tracker is None:
        _limpiar_evidencia_velocidad()
        tracker = SpeedTracker(linea_1_y, linea_2_y, distancia_metros, fps)
        estado["speed_tracker"] = tracker
    return tracker


def _resumen_velocidad_vacio(distancia_metros: float, fps: float) -> dict:
    return {
        "estado": "esperando_linea_1",
        "frame_cruce_linea_1": None,
        "frame_cruce_linea_2": None,
        "tiempo_cruce_linea_1": None,
        "tiempo_cruce_linea_2": None,
        "tiempo_entre_lineas": None,
        "distancia_metros": distancia_metros,
        "fps": fps,
        "velocidad_kmh": None,
        "centro_x": None,
        "centro_y": None,
    }


def _dibujar_centro_placa(frame, bbox: list[int]) -> None:
    x1, y1, x2, y2 = bbox
    centro_x = int((x1 + x2) / 2)
    centro_y = int((y1 + y2) / 2)
    cv2.circle(frame, (centro_x, centro_y), 5, (255, 255, 255), -1)
    cv2.circle(frame, (centro_x, centro_y), 7, (0, 0, 0), 1)


def _dibujar_info_velocidad(frame, velocidad: dict) -> None:
    velocidad_kmh = velocidad.get("velocidad_kmh")
    texto = "Velocidad: Pendiente" if velocidad_kmh is None else f"Velocidad: {velocidad_kmh:.2f} km/h"
    cv2.putText(frame, texto, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Estado velocidad: {velocidad.get('estado', 'pendiente')}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _guardar_evidencia_velocidad(frame, velocidad: dict) -> None:
    velocidad_dir = Path("reports") / "evidencias" / "velocidad"
    velocidad_dir.mkdir(parents=True, exist_ok=True)

    if velocidad.get("frame_cruce_linea_1") is not None and not (velocidad_dir / "frame_cruce_linea_1.jpg").exists():
        cv2.imwrite(str(velocidad_dir / "frame_cruce_linea_1.jpg"), frame)

    if velocidad.get("frame_cruce_linea_2") is not None and not (velocidad_dir / "frame_cruce_linea_2.jpg").exists():
        cv2.imwrite(str(velocidad_dir / "frame_cruce_linea_2.jpg"), frame)

    if velocidad.get("velocidad_kmh") is not None:
        cv2.imwrite(str(velocidad_dir / "frame_velocidad_calculada.jpg"), frame)
        datos = {
            "distancia_metros": velocidad.get("distancia_metros"),
            "fps": velocidad.get("fps"),
            "frame_linea_1": velocidad.get("frame_cruce_linea_1"),
            "frame_linea_2": velocidad.get("frame_cruce_linea_2"),
            "tiempo_segundos": velocidad.get("tiempo_entre_lineas"),
            "velocidad_kmh": velocidad.get("velocidad_kmh"),
        }
        with open(velocidad_dir / "velocidad_evento.json", "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False, indent=2)


def _limpiar_evidencia_velocidad() -> None:
    velocidad_dir = Path("reports") / "evidencias" / "velocidad"
    velocidad_dir.mkdir(parents=True, exist_ok=True)
    for nombre in [
        "frame_cruce_linea_1.jpg",
        "frame_cruce_linea_2.jpg",
        "frame_velocidad_calculada.jpg",
        "velocidad_evento.json",
    ]:
        ruta = velocidad_dir / nombre
        if ruta.exists():
            try:
                ruta.unlink()
            except PermissionError:
                pass


def _guardar_debug_evento_detecciones(numero_frame: int, debug_detecciones: list[dict]) -> None:
    if not debug_detecciones:
        return
    debug_dir = Path("reports") / "evidencias" / "debug_detecciones"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "detecciones_debug.csv"
    existe = debug_path.exists()
    with open(debug_path, "a", newline="", encoding="utf-8") as archivo:
        campos = ["frame", "confianza", "bbox", "aceptada", "motivo_rechazo"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        if not existe:
            writer.writeheader()
        for item in debug_detecciones:
            writer.writerow(
                {
                    "frame": numero_frame,
                    "confianza": f"{item['confianza']:.4f}",
                    "bbox": item["bbox"],
                    "aceptada": "si" if item["aceptada"] else "no",
                    "motivo_rechazo": item["motivo_rechazo"],
                }
            )


def _resumir_motivos_rechazo(debug_detecciones: list[dict]) -> str:
    motivos = sorted({item["motivo_rechazo"] for item in debug_detecciones if item.get("motivo_rechazo")})
    return ", ".join(motivos)


def _filtrar_detecciones_zona_superior(detecciones: list[dict], alto_frame: int) -> list[dict]:
    limite_superior = alto_frame * EXCLUIR_ZONA_SUPERIOR_PORCENTAJE
    detecciones_validas = []
    for deteccion in detecciones:
        _, y1, _, y2 = deteccion["bbox"]
        centro_y = (y1 + y2) / 2
        if centro_y >= limite_superior:
            detecciones_validas.append(deteccion)
    return detecciones_validas


def _dibujar_marcas_monitoreo(
    frame,
    distancia_lineas_m: float,
    limite_velocidad_kmh: float,
    nombre_fuente: str,
    posicion_linea_1: float = 0.45,
    posicion_linea_2: float = 0.65,
) -> None:
    alto_frame, ancho_frame = frame.shape[:2]
    linea_1_y = int(alto_frame * posicion_linea_1)
    linea_2_y = int(alto_frame * posicion_linea_2)

    cv2.line(frame, (0, linea_1_y), (ancho_frame, linea_1_y), (0, 255, 255), 2)
    cv2.line(frame, (0, linea_2_y), (ancho_frame, linea_2_y), (0, 80, 255), 2)
    cv2.putText(frame, "Linea 1", (20, max(30, linea_1_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Linea 2", (20, max(30, linea_2_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
    cv2.putText(
        frame,
        f"Distancia configurada: {distancia_lineas_m:.1f} m",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Limite de velocidad: {limite_velocidad_kmh:.1f} km/h",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Fuente: {nombre_fuente}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def procesar_entrada(*args, **kwargs) -> dict:
    return procesar_imagen_prueba(*args, **kwargs)

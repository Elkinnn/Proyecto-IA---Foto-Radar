from datetime import datetime
from pathlib import Path
import csv
import json
import time

import cv2

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd
from src.fuzzy_system import clasificar_velocidad
from src.notifier import enviar_notificacion_sancion
from src.plate_detector import PlateDetector, dibujar_deteccion
from src.plate_reader import PlateReader
from src.report_generator import guardar_reporte
from src.speed_estimator import SpeedTracker, estimar_velocidad


EXCLUIR_ZONA_SUPERIOR_PORCENTAJE = 0.20
GUARDAR_RECORTE_CADA_N_FRAMES = 15
BBOX_SUAVIZADO_ALPHA = 0.6


def _registrar_resultado(resultado: dict, config: dict) -> dict:
    ruta_bd = config["database"]["path"]
    clasificacion = resultado["clasificacion_difusa"]

    if clasificacion["sancion"]:
        resultado["notificacion"] = enviar_notificacion_sancion(resultado.get("vehiculo"), resultado, config)

    ruta_reporte = guardar_reporte(resultado, config["paths"]["reports_dir"])
    resultado["ruta_reporte"] = ruta_reporte

    evento = {
        "fecha_hora": resultado["fecha_hora"],
        "placa": resultado["texto_placa"],
        "velocidad": resultado["velocidad_kmh"],
        "estado": clasificacion["estado"],
        "sancion": "SI" if clasificacion["sancion"] else "NO",
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

    detector = PlateDetector(config["models"]["plate_detector_path"])
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
        "sancion_generada": clasificacion["sancion"],
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
    frame_callback=None,
    progreso_callback=None,
    detener_callback=None,
) -> dict:
    captura = cv2.VideoCapture(fuente)
    detector = PlateDetector()

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
    frames_procesados = 0
    ultimo_frame = None

    evidencia_dir = Path("reports") / "evidencias" / evidencia_subdir
    evidencia_dir.mkdir(parents=True, exist_ok=True)
    primer_frame_evidencia = evidencia_dir / "primer_frame_procesado.jpg"
    ultimo_frame_evidencia = evidencia_dir / "ultimo_frame_procesado.jpg"
    placas_dir = Path("reports") / "evidencias" / "placas_detectadas"
    placas_dir.mkdir(parents=True, exist_ok=True)
    estado_persistencia = _crear_estado_persistencia()
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
    mejor_confianza_evento = None
    frame_mejor_evento = None
    ruta_mejor_recorte_evento = None

    objetivo_frames = max_frames if max_frames > 0 else total_frames
    if total_frames > 0 and max_frames > 0:
        objetivo_frames = min(max_frames, total_frames)

    while captura.isOpened():
        if detener_callback and detener_callback():
            break

        ok, frame = captura.read()
        if not ok:
            break
        frame = aplicar_rotacion(frame, rotacion)
        alto_actual, ancho_actual = frame.shape[:2]

        frames_procesados += 1
        estado_frame = _procesar_frame_monitoreo(
            frame,
            detector,
            frames_procesados,
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
        )
        frame = estado_frame["frame_visual"]
        estado_persistencia = estado_frame["estado_persistencia"]
        detecciones_frame = estado_frame["detecciones_frame"]
        detecciones_brutas = estado_frame["detecciones_brutas"]
        detecciones_validas = estado_frame["detecciones_validas"]
        motivos_rechazo = estado_frame["motivos_rechazo"]
        eventos_placa = estado_frame["eventos_placa"]
        estado_placa = estado_frame["estado_placa"]
        mensaje_detector = estado_frame["mensaje_detector"]
        ultima_deteccion = estado_frame["ultima_deteccion"] or ultima_deteccion
        ultimo_recorte_placa = estado_frame["ultimo_recorte_placa"] or ultimo_recorte_placa
        ultimo_frame_deteccion = estado_frame["ultimo_frame_deteccion"] or ultimo_frame_deteccion
        mejor_confianza_evento = estado_frame["mejor_confianza_evento"] if estado_frame["mejor_confianza_evento"] is not None else mejor_confianza_evento
        frame_mejor_evento = estado_frame["frame_mejor_evento"] or frame_mejor_evento
        ruta_mejor_recorte_evento = estado_frame["ruta_mejor_recorte_evento"] or ruta_mejor_recorte_evento

        if frames_procesados == 1:
            cv2.imwrite(str(primer_frame_evidencia), frame)
        ultimo_frame = frame.copy()

        if frame_callback:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            estado_callback = {
                "frame_actual": frames_procesados,
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
                "evento_activo": estado_frame["evento_activo"],
                "evento_id": estado_frame["evento_id"],
                "mejor_confianza_evento": estado_frame["mejor_confianza_evento"],
                "frame_mejor_evento": estado_frame["frame_mejor_evento"],
                "frames_sin_deteccion": estado_frame["frames_sin_deteccion"],
                "ruta_mejor_recorte_evento": estado_frame["ruta_mejor_recorte_evento"],
                "velocidad": estado_frame["velocidad"],
            }
            try:
                frame_callback(frame_rgb, frames_procesados, estado_callback)
            except TypeError:
                frame_callback(frame_rgb, frames_procesados)

        if progreso_callback:
            if objetivo_frames and objetivo_frames > 0:
                progreso_callback(min(frames_procesados / objetivo_frames, 1.0))
            else:
                progreso_callback(0.0)

        delay = _calcular_delay_reproduccion(fps, velocidad_reproduccion)
        if delay > 0:
            time.sleep(delay)

        if max_frames > 0 and frames_procesados >= max_frames:
            break

    captura.release()

    if estado_persistencia["evento_activo"]:
        _cerrar_evento_placa(estado_persistencia, frames_procesados)
        ruta_mejor_recorte_evento = estado_persistencia["ruta_mejor_recorte_evento"] or ruta_mejor_recorte_evento

    if ultimo_frame is not None:
        cv2.imwrite(str(ultimo_frame_evidencia), ultimo_frame)

    return {
        "estado": "finalizado",
        "mensaje_estado": f"Monitoreo desde {nombre_fuente.lower()} finalizado. Flujo visual listo; deteccion, OCR y velocidad real quedan para la siguiente etapa.",
        "frames_procesados": frames_procesados,
        "fps": fps,
        "ancho": ultimo_frame.shape[1] if ultimo_frame is not None else ancho,
        "alto": ultimo_frame.shape[0] if ultimo_frame is not None else alto,
        "total_frames": total_frames,
        "duracion_segundos": duracion,
        "segundos_procesados": _calcular_segundos_procesados(frames_procesados, fps),
        "modo_procesamiento": _obtener_modo_procesamiento(max_frames),
        "fuente": nombre_fuente,
        "rotacion": rotacion,
        "distancia_lineas_m": distancia_lineas_m,
        "posicion_linea_1": posicion_linea_1,
        "posicion_linea_2": posicion_linea_2,
        "limite_velocidad_kmh": limite_velocidad_kmh,
        "frecuencia_deteccion": frecuencia_deteccion,
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
    }


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
    velocidad = _obtener_speed_tracker(
        estado_persistencia,
        frame_visual.shape[0],
        distancia_lineas_m,
        fps,
        posicion_linea_1,
        posicion_linea_2,
    ).resumen()
    if frecuencia_deteccion > 0 and numero_frame % frecuencia_deteccion == 0:
        resultado_detector = detector.detectar_en_frame(frame_limpio, conf_min=conf_min)
        mensaje_detector = resultado_detector["mensaje"]
        detecciones_brutas = resultado_detector.get("detecciones_brutas", 0)
        _guardar_debug_evento_detecciones(numero_frame, resultado_detector.get("debug_detecciones", []))
        motivos_rechazo = _resumir_motivos_rechazo(resultado_detector.get("debug_detecciones", []))
        detecciones_validas = _filtrar_detecciones_zona_superior(
            resultado_detector["detecciones"],
            frame_limpio.shape[0],
        )

    _dibujar_marcas_monitoreo(frame_visual, distancia_lineas_m, limite_velocidad_kmh, nombre_fuente, posicion_linea_1, posicion_linea_2)

    if detecciones_validas:
        detecciones_frame = len(detecciones_validas)
        frame_deteccion_path = placas_dir / f"frame_deteccion_{numero_frame:06d}.jpg"
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
            "frame_deteccion": str(frame_deteccion_path),
        }

        cv2.imwrite(str(frame_deteccion_path), frame_visual)
        ultimo_frame_deteccion = str(frame_deteccion_path)
        estado_persistencia["ultimo_frame_deteccion"] = ultimo_frame_deteccion
        cv2.imwrite(str(evidencia_dir / "ultimo_frame_con_deteccion.jpg"), frame_visual)
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
        "eventos_placa": eventos_placa,
        "placas_detectadas": eventos_placa,
        "frames_desde_ultima_deteccion": estado_persistencia["frames_desde_ultima_deteccion"],
        "ultima_deteccion": ultima_deteccion,
        "ultimo_recorte_placa": ultimo_recorte_placa,
        "ultimo_frame_deteccion": ultimo_frame_deteccion,
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

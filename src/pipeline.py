from datetime import datetime
from pathlib import Path

import cv2

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd
from src.fuzzy_system import clasificar_velocidad
from src.notifier import enviar_notificacion_sancion
from src.plate_detector import PlateDetector, dibujar_deteccion
from src.plate_reader import PlateReader
from src.report_generator import guardar_reporte
from src.speed_estimator import calcular_velocidad_kmh, estimar_velocidad


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
    fuente_video: str | int,
    config: dict,
    frame_callback=None,
    detener_callback=None,
    cada_n_frames: int = 15,
    max_frames: int | None = None,
) -> dict:
    ruta_bd = config["database"]["path"]
    inicializar_bd(ruta_bd)

    detector = PlateDetector(config["models"]["plate_detector_path"])
    lector = PlateReader()
    captura = cv2.VideoCapture(fuente_video)

    if not captura.isOpened():
        return {
            "estado": "error",
            "mensaje": "No se pudo abrir la fuente de video/camara.",
            "eventos": [],
            "frames_procesados": 0,
        }

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    eventos = []
    frames_procesados = 0
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_disponible = detector.model is not None
    mensaje_detector = detector.estado

    while captura.isOpened():
        if detener_callback and detener_callback():
            break

        ok, frame = captura.read()
        if not ok:
            break

        frames_procesados += 1
        alto, ancho = frame.shape[:2]
        linea_1_y = int(alto * 0.45)
        linea_2_y = int(alto * 0.70)

        cv2.line(frame, (0, linea_1_y), (ancho, linea_1_y), (0, 255, 255), 2)
        cv2.line(frame, (0, linea_2_y), (ancho, linea_2_y), (0, 80, 255), 2)
        cv2.putText(frame, "Linea virtual 1", (20, linea_1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Linea virtual 2", (20, linea_2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        if detector_disponible and frames_procesados % cada_n_frames == 0:
            ruta_frame = output_dir / f"frame_monitoreo_{frames_procesados}.jpg"
            cv2.imwrite(str(ruta_frame), frame)
            deteccion = detector.detectar(str(ruta_frame))
            mensaje_detector = deteccion["mensaje"]

            if deteccion["detectada"]:
                lectura = lector.leer(deteccion, "automatico")
                texto_placa = lectura["texto"]
                if texto_placa:
                    # Punto preparado para tracking real: el tiempo debe venir del cruce entre lineas.
                    tiempo_estimado = cada_n_frames / fps
                    velocidad_kmh = calcular_velocidad_kmh(
                        float(config["speed"]["default_distance_meters"]),
                        tiempo_estimado,
                    )
                    clasificacion = clasificar_velocidad(velocidad_kmh, config)
                    vehiculo = buscar_vehiculo_por_placa(texto_placa, ruta_bd)
                    evento = {
                        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
                        "ruta_archivo": str(fuente_video),
                        "ruta_archivo_procesado": str(ruta_frame),
                        "deteccion": deteccion,
                        "placa_detectada": True,
                        "texto_placa": texto_placa,
                        "lectura": lectura,
                        "velocidad_kmh": velocidad_kmh,
                        "clasificacion_difusa": clasificacion,
                        "vehiculo": vehiculo,
                        "sancion_generada": clasificacion["sancion"],
                        "notificacion": None,
                        "ruta_reporte": None,
                        "id_evento": None,
                    }
                    eventos.append(_registrar_resultado(evento, config))

        if frame_callback:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_callback(frame_rgb, frames_procesados, mensaje_detector)

        if max_frames and frames_procesados >= max_frames:
            break

    captura.release()
    return {
        "estado": "finalizado",
        "mensaje": mensaje_detector if detector_disponible else "Detector de placa aun no entrenado. Monitoreo visual disponible sin eventos automaticos.",
        "eventos": eventos,
        "frames_procesados": frames_procesados,
        "detector_disponible": detector_disponible,
    }


def procesar_entrada(*args, **kwargs) -> dict:
    return procesar_imagen_prueba(*args, **kwargs)

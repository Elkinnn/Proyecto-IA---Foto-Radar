from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import inicializar_bd, listar_eventos, listar_vehiculos
from src.pipeline import (
    procesar_camara_monitoreo,
    procesar_frame_video_monitoreo,
    procesar_imagen_prueba,
    procesar_video_monitoreo,
)
from src.utils import cargar_config, guardar_archivo_subido


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
    if reiniciar_revision:
        st.session_state.frame_actual = 0
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_imagen_procesada = None
        st.session_state.placas_detectadas_acumuladas = 0
        st.session_state.estado_persistencia = None

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
    elif iniciar:
        st.session_state.monitoreo_activo = True
        st.session_state.monitoreo_pausado = False
        st.session_state.ultimo_resultado = None
        st.session_state.ultima_imagen_procesada = None

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

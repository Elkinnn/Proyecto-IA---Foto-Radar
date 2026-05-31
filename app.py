from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import inicializar_bd, listar_eventos, listar_vehiculos
from src.pipeline import procesar_imagen_prueba, procesar_video_monitoreo
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
        st.info("El sistema mostrara eventos cuando exista deteccion, OCR y cruce entre lineas virtuales.")
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


def pestaña_monitoreo(config: dict) -> None:
    st.header("Monitoreo por video/camara")

    control, visor = st.columns([0.32, 0.68])
    with control:
        fuente = st.radio("Fuente principal", ["Video de prueba", "Camara en vivo"])
        video = None
        indice_camara = 0

        if fuente == "Video de prueba":
            video = st.file_uploader("Cargar video", type=["mp4", "avi", "mov", "mkv"], key="video_monitoreo")
        else:
            indice_camara = st.number_input("Indice de camara", min_value=0, value=0, step=1)

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
        cada_n_frames = st.slider("Frecuencia de deteccion", 5, 60, 15, help="Frames entre llamadas al detector.")
        max_frames = st.slider("Frames maximos por ejecucion", 30, 900, 240)

        iniciar = st.button("Iniciar monitoreo", type="primary", use_container_width=True)
        detener = st.button("Detener monitoreo", use_container_width=True)

    if detener:
        st.session_state["detener_monitoreo"] = True

    with visor:
        frame_placeholder = st.empty()
        estado_placeholder = st.empty()
        st.caption("Las lineas virtuales se dibujan sobre el frame procesado.")
        resultado_placeholder = st.container()

    if iniciar:
        st.session_state["detener_monitoreo"] = False
        config["speed"]["default_distance_meters"] = distancia_metros
        config["speed"]["campus_speed_limit_kmh"] = limite_velocidad

        if fuente == "Video de prueba":
            if not video:
                st.warning("Carga un video de prueba para iniciar el monitoreo.")
                return
            fuente_video = guardar_archivo_subido(video, config["paths"]["input_dir"])
        else:
            fuente_video = int(indice_camara)

        def actualizar_frame(frame_rgb, numero_frame: int, mensaje: str) -> None:
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            estado_placeholder.info(f"Frame {numero_frame} | {mensaje}")

        with st.spinner("Monitoreo en ejecucion..."):
            resumen = procesar_video_monitoreo(
                fuente_video,
                config,
                frame_callback=actualizar_frame,
                detener_callback=lambda: st.session_state.get("detener_monitoreo", False),
                cada_n_frames=cada_n_frames,
                max_frames=max_frames,
            )

        eventos = resumen.get("eventos", [])
        estado_placeholder.info(f"{resumen['mensaje']} | Frames procesados: {resumen['frames_procesados']}")
        with resultado_placeholder:
            panel_resultados(eventos[-1] if eventos else None)

    else:
        with resultado_placeholder:
            panel_resultados(None)


def pestaña_pruebas(config: dict) -> None:
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


def pestaña_base_datos(config: dict) -> None:
    st.header("Base de datos")
    ruta_bd = config["database"]["path"]

    vehiculos = listar_vehiculos(ruta_bd)
    eventos = listar_eventos(ruta_bd)

    st.subheader("Vehiculos registrados")
    st.dataframe(pd.DataFrame(vehiculos), use_container_width=True, hide_index=True)

    st.subheader("Eventos recientes")
    st.dataframe(pd.DataFrame(eventos), use_container_width=True, hide_index=True)


def pestaña_evidencias(config: dict) -> None:
    st.header("Evidencias")
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    archivos = sorted(reports_dir.glob("*.json"), key=lambda ruta: ruta.stat().st_mtime, reverse=True)

    if not archivos:
        st.info("Aun no existen evidencias guardadas.")
        return

    seleccionado = st.selectbox("Evidencia", archivos, format_func=lambda ruta: ruta.name)
    st.code(seleccionado.read_text(encoding="utf-8"), language="json")


def pestaña_configuracion(config: dict) -> None:
    st.header("Configuracion")
    st.json(config)
    st.caption("Los cambios persistentes se realizan editando config.yaml.")


def main() -> None:
    config = cargar_config()
    inicializar_bd(config["database"]["path"])

    st.title("Fotorradar Ecuador IA")
    st.caption("Consola de monitoreo por video/camara para placas ecuatorianas.")

    tabs = st.tabs(["Monitoreo", "Pruebas", "Base de datos", "Evidencias", "Configuracion"])
    with tabs[0]:
        pestaña_monitoreo(config)
    with tabs[1]:
        pestaña_pruebas(config)
    with tabs[2]:
        pestaña_base_datos(config)
    with tabs[3]:
        pestaña_evidencias(config)
    with tabs[4]:
        pestaña_configuracion(config)


if __name__ == "__main__":
    main()

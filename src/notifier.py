from pathlib import Path
import json


def generar_notificacion_simulada(evento: dict, vehiculo: dict | None, resultado_difuso: dict) -> dict | None:
    if resultado_difuso.get("nivel_infraccion") == "Sin infracción":
        return None

    vehiculo = vehiculo or {}
    placa = evento.get("placa") or vehiculo.get("placa") or "SIN_PLACA"
    propietario = vehiculo.get("propietario") or "Propietario no registrado"
    destinatario = vehiculo.get("correo")
    asunto = f"Notificación de velocidad - Placa {placa}"
    mensaje = (
        f"Estimado/a {propietario},\n\n"
        f"Se ha registrado un evento de velocidad asociado al vehículo de placa {placa}.\n\n"
        "Datos del vehículo:\n"
        f"Marca: {vehiculo.get('marca', 'Sin registro')}\n"
        f"Modelo: {vehiculo.get('modelo', 'Sin registro')}\n"
        f"Color: {vehiculo.get('color', 'Sin registro')}\n"
        f"Estado: {vehiculo.get('estado', 'Sin registro')}\n\n"
        "Datos del evento:\n"
        f"Fecha y hora: {evento.get('fecha_hora', 'Sin fecha')}\n"
        f"Velocidad registrada: {float(evento.get('velocidad_kmh', 0.0)):.2f} km/h\n"
        f"Límite permitido: {float(evento.get('limite_kmh', resultado_difuso.get('limite_kmh', 0.0))):.2f} km/h\n"
        f"Clasificación: {resultado_difuso.get('estado', 'Pendiente')}\n"
        f"Nivel de infracción: {resultado_difuso.get('nivel_infraccion', 'Pendiente')}\n"
        f"Sanción: {resultado_difuso.get('sancion', 'Pendiente')}\n"
        f"Horas de suspensión: {resultado_difuso.get('horas_suspension', 0)}\n\n"
        "Mensaje:\n"
        f"{resultado_difuso.get('mensaje', 'Sin mensaje')}\n\n"
        "Esta notificación fue generada en modo simulado como parte del sistema Fotorradar Ecuador IA."
    )

    return {
        "destinatario": destinatario,
        "asunto": asunto,
        "mensaje": mensaje,
        "modo": "simulado",
        "enviado": False,
        "estado": "generada",
    }


def guardar_notificacion_simulada(notificacion: dict, evento_id: int) -> dict:
    notificaciones_dir = Path("reports") / "evidencias" / "notificaciones"
    notificaciones_dir.mkdir(parents=True, exist_ok=True)

    ruta_txt = notificaciones_dir / f"notificacion_evento_{evento_id}.txt"
    ruta_json = notificaciones_dir / f"notificacion_evento_{evento_id}.json"

    ruta_txt.write_text(notificacion["mensaje"], encoding="utf-8")
    ruta_json.write_text(json.dumps(notificacion, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ruta_txt": str(ruta_txt),
        "ruta_json": str(ruta_json),
    }


def enviar_notificacion_sancion(vehiculo: dict | None, resultado: dict, config: dict) -> dict:
    clasificacion = resultado.get("clasificacion_difusa") or {}
    evento = {
        "placa": resultado.get("texto_placa"),
        "fecha_hora": resultado.get("fecha_hora"),
        "velocidad_kmh": resultado.get("velocidad_kmh"),
        "limite_kmh": config.get("speed", {}).get("campus_speed_limit_kmh"),
    }
    notificacion = generar_notificacion_simulada(evento, vehiculo, clasificacion)
    if not notificacion:
        return {
            "enviado": False,
            "modo": "simulado",
            "estado": "no_generada",
            "mensaje": "No se generó notificación porque no existe infracción.",
        }
    return notificacion

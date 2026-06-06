"""Sistema de notificaciones por correo del Fotorradar.

Responsabilidades:
- Validar el correo destino (incluye correos institucionales, p. ej. @uni.edu.ec).
- Aplicar un "candado de calidad" que evita enviar notificaciones de placas
  incompletas (media placa) o lecturas poco confiables.
- Construir el mensaje de la sancion usando la logica difusa (por ahora con una
  velocidad/distancia "quemada" de demostracion).
- Enviar el correo por SMTP real; si no hay credenciales configuradas, cae en un
  modo simulado que guarda el correo y los adjuntos en disco (no rompe la demo).
"""

from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
import json
import mimetypes
import os
import re
import smtplib
import ssl


from src.utils import cargar_variables_entorno

cargar_variables_entorno()
# Cuando se integre la velocidad real, basta con pasar otra velocidad a
# construir_resultado_difuso_demo().
VELOCIDAD_DEMO_KMH = 47.0
LIMITE_DEMO_KMH = 30.0

_EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

NOTIFICACIONES_DIR = Path("reports") / "evidencias" / "notificaciones"

DEFAULT_CALIDAD = {
    "exigir_formato_valido": True,
    "confianza_minima": 0.55,
    "aspect_min": 1.8,
    "aspect_max": 5.0,
    "nitidez_minima": 30.0,
    "longitud_min": 6,
    "longitud_max": 7,
    "permitir_envio_si_no_apto": False,
}


def validar_correo(correo: str | None) -> bool:
    """Valida cualquier correo bien formado, incluidos los institucionales."""
    if not correo:
        return False
    return bool(_EMAIL_REGEX.match(correo.strip()))


def construir_resultado_difuso(
    velocidad_kmh: float,
    limite_kmh: float | None = None,
    config: dict | None = None,
) -> dict:
    """Clasificacion difusa real para correo y panel (entrada: km/h, salida: multa)."""
    from src.fuzzy_system import clasificar_velocidad

    return clasificar_velocidad(float(velocidad_kmh), limite_kmh, config)


def construir_resultado_difuso_demo(
    velocidad_kmh: float = VELOCIDAD_DEMO_KMH,
    limite_kmh: float = LIMITE_DEMO_KMH,
    config: dict | None = None,
) -> dict:
    """Compatibilidad: usa clasificacion difusa con config si se provee."""
    return construir_resultado_difuso(velocidad_kmh, limite_kmh, config)


def _config_calidad(config: dict | None) -> dict:
    calidad = dict(DEFAULT_CALIDAD)
    if config:
        notif = (config.get("notificaciones") or {}).get("calidad") or {}
        calidad.update({k: v for k, v in notif.items() if v is not None})
    return calidad


def evaluar_calidad_evento(metricas: dict, config: dict | None = None) -> dict:
    """Candado de calidad: decide si un evento merece enviar correo.

    `metricas` puede incluir:
      - placa (str), formato_valido (bool), confianza_final (float)
      - aspect_ratio (float), nitidez (float), cerca_borde (bool)
      - cantidad_caracteres (int)
    Devuelve {apto: bool, razones: [str], detalles: {...}}.
    """
    cfg = _config_calidad(config)
    razones: list[str] = []

    placa = (metricas.get("placa") or "").strip().upper()
    longitud = len(placa)
    formato_valido = bool(metricas.get("formato_valido"))
    confianza = metricas.get("confianza_final")
    aspect_ratio = metricas.get("aspect_ratio")
    nitidez = metricas.get("nitidez")
    cerca_borde = bool(metricas.get("cerca_borde"))
    cantidad = metricas.get("cantidad_caracteres")

    if not placa or placa in {"PENDIENTE", "SIN_PLACA"}:
        razones.append("No hay una placa reconocida.")

    if not (int(cfg["longitud_min"]) <= longitud <= int(cfg["longitud_max"])):
        razones.append(
            f"La placa tiene {longitud} caracteres; se esperan entre "
            f"{cfg['longitud_min']} y {cfg['longitud_max']} (posible placa incompleta)."
        )

    if cfg.get("exigir_formato_valido") and not formato_valido:
        razones.append("El formato de la placa no es valido (LLL + digitos).")

    if confianza is not None and float(confianza) < float(cfg["confianza_minima"]):
        razones.append(
            f"Confianza de lectura {float(confianza):.2f} < minima {float(cfg['confianza_minima']):.2f}."
        )

    if aspect_ratio is not None and not (
        float(cfg["aspect_min"]) <= float(aspect_ratio) <= float(cfg["aspect_max"])
    ):
        razones.append(
            f"Relacion de aspecto {float(aspect_ratio):.2f} fuera de rango de placa "
            f"({cfg['aspect_min']}-{cfg['aspect_max']}); posible recorte parcial."
        )

    if cerca_borde:
        razones.append("El recorte esta pegado al borde del frame (placa posiblemente cortada).")

    if nitidez is not None and float(nitidez) < float(cfg["nitidez_minima"]):
        razones.append(f"Nitidez {float(nitidez):.1f} < minima {float(cfg['nitidez_minima']):.1f}.")

    if cantidad is not None and int(cantidad) < int(cfg["longitud_min"]):
        razones.append(f"Solo se segmentaron {int(cantidad)} caracteres.")

    apto = len(razones) == 0
    return {
        "apto": apto,
        "razones": razones,
        "detalles": {
            "placa": placa,
            "longitud": longitud,
            "formato_valido": formato_valido,
            "confianza_final": confianza,
            "aspect_ratio": aspect_ratio,
            "nitidez": nitidez,
            "cerca_borde": cerca_borde,
            "cantidad_caracteres": cantidad,
        },
    }


def construir_mensaje_sancion(placa: str, resultado_difuso: dict, contexto: dict | None = None) -> str:
    contexto = contexto or {}
    sancion_aplica = bool(resultado_difuso.get("sancion_aplica"))
    fecha = contexto.get("fecha_hora") or datetime.now().isoformat(timespec="seconds")

    encabezado = (
        "Estimado/a usuario,\n\n"
        f"El sistema Fotorradar Ecuador IA registro un evento asociado a la placa {placa}.\n\n"
    )

    datos_evento = (
        "Datos del evento:\n"
        f"  Fecha y hora: {fecha}\n"
        f"  Placa reconocida: {placa}\n"
        f"  Velocidad registrada: {float(resultado_difuso.get('velocidad_kmh', 0.0)):.2f} km/h\n"
        f"  Limite permitido: {float(resultado_difuso.get('limite_kmh', 0.0)):.2f} km/h\n"
        f"  Clasificacion: {resultado_difuso.get('estado', 'Pendiente')}\n\n"
    )

    if sancion_aplica:
        cuerpo_sancion = (
            "Resultado de la evaluacion (logica difusa):\n"
            f"  Nivel de infraccion: {resultado_difuso.get('nivel_infraccion', 'Pendiente')}\n"
            f"  Multa: {resultado_difuso.get('multa_texto', 'Pendiente')}\n"
            f"  Sancion: {resultado_difuso.get('sancion', 'Pendiente')}\n"
            f"  Horas de suspension: {resultado_difuso.get('horas_suspension', 0)}\n\n"
            f"Detalle: {resultado_difuso.get('mensaje', '')}\n\n"
        )
    else:
        cuerpo_sancion = (
            "Resultado de la evaluacion (logica difusa):\n"
            "  No corresponde sancion para esta velocidad.\n\n"
            f"Detalle: {resultado_difuso.get('mensaje', '')}\n\n"
        )

    cierre = (
        "Se adjunta la evidencia fotografica del evento (mejor frame y recorte de la placa).\n\n"
        "Nota: la velocidad y la distancia de este aviso son valores de demostracion. "
        "La lectura de la placa y la imagen corresponden al evento real detectado.\n\n"
        "Sistema Fotorradar Ecuador IA."
    )

    return encabezado + datos_evento + cuerpo_sancion + cierre


def _resolver_config_smtp(config: dict | None) -> dict:
    notif = (config or {}).get("notificaciones") or {}
    email_legacy = (config or {}).get("email") or {}

    host = os.environ.get("FOTORRADAR_SMTP_HOST") or notif.get("smtp_host") or email_legacy.get("smtp_host")
    port = os.environ.get("FOTORRADAR_SMTP_PORT") or notif.get("smtp_port") or email_legacy.get("smtp_port") or 587
    usuario = os.environ.get("FOTORRADAR_SMTP_USER") or notif.get("smtp_user")
    password = os.environ.get("FOTORRADAR_SMTP_PASSWORD") or notif.get("smtp_password")
    if password is not None:
        password = str(password).strip().strip('"').strip("'").replace(" ", "")
    remitente = (
        os.environ.get("FOTORRADAR_SMTP_FROM")
        or notif.get("remitente")
        or usuario
        or email_legacy.get("sender")
    )
    seguridad = (os.environ.get("FOTORRADAR_SMTP_SECURITY") or notif.get("smtp_seguridad") or "starttls").lower()
    port = int(port)
    if port == 465:
        seguridad = "ssl"
    elif port == 587 and seguridad == "ssl":
        seguridad = "starttls"
    return {
        "host": host,
        "port": port,
        "usuario": usuario,
        "password": password,
        "remitente": remitente,
        "seguridad": seguridad,
    }


def _adjuntar_imagenes(mensaje: EmailMessage, adjuntos: list[str] | None) -> list[str]:
    adjuntados = []
    for ruta in adjuntos or []:
        if not ruta:
            continue
        ruta_path = Path(ruta)
        if not ruta_path.exists():
            continue
        tipo, _ = mimetypes.guess_type(str(ruta_path))
        maintype, subtype = (tipo.split("/", 1) if tipo else ("application", "octet-stream"))
        mensaje.add_attachment(
            ruta_path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=ruta_path.name,
        )
        adjuntados.append(str(ruta_path))
    return adjuntados


def guardar_notificacion_simulada(notificacion: dict, evento_id) -> dict:
    NOTIFICACIONES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_txt = NOTIFICACIONES_DIR / f"notificacion_evento_{evento_id}.txt"
    ruta_json = NOTIFICACIONES_DIR / f"notificacion_evento_{evento_id}.json"
    ruta_txt.write_text(notificacion.get("mensaje", ""), encoding="utf-8")
    ruta_json.write_text(json.dumps(notificacion, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ruta_txt": str(ruta_txt), "ruta_json": str(ruta_json)}


def enviar_correo(
    destinatario: str,
    asunto: str,
    mensaje: str,
    adjuntos: list[str] | None = None,
    config: dict | None = None,
    evento_id="manual",
) -> dict:
    """Envia el correo por SMTP. Si no hay credenciales, guarda en modo simulado."""
    if not validar_correo(destinatario):
        return {
            "enviado": False,
            "modo": "error",
            "estado": "correo_invalido",
            "destinatario": destinatario,
            "asunto": asunto,
            "mensaje": mensaje,
            "error": "El correo destino no es valido.",
        }

    smtp = _resolver_config_smtp(config)
    adjuntos = adjuntos or []
    hay_credenciales = bool(smtp["host"] and smtp["usuario"] and smtp["password"] and smtp["remitente"])

    base = {
        "destinatario": destinatario,
        "asunto": asunto,
        "mensaje": mensaje,
        "adjuntos": adjuntos,
    }

    if not hay_credenciales:
        rutas = guardar_notificacion_simulada({**base, "modo": "simulado"}, evento_id)
        return {
            **base,
            **rutas,
            "enviado": False,
            "modo": "simulado",
            "estado": "generada_sin_credenciales",
            "mensaje_estado": (
                "No hay credenciales SMTP configuradas. El correo se genero en modo simulado "
                "(revise reports/evidencias/notificaciones/). Configure FOTORRADAR_SMTP_* para envio real."
            ),
        }

    correo = EmailMessage()
    correo["From"] = smtp["remitente"]
    correo["To"] = destinatario
    correo["Subject"] = asunto
    correo.set_content(mensaje)
    adjuntados = _adjuntar_imagenes(correo, adjuntos)

    try:
        if smtp["seguridad"] == "ssl":
            contexto_ssl = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp["host"], smtp["port"], context=contexto_ssl, timeout=30) as servidor:
                servidor.login(smtp["usuario"], smtp["password"])
                servidor.send_message(correo)
        else:
            with smtplib.SMTP(smtp["host"], smtp["port"], timeout=30) as servidor:
                servidor.ehlo()
                servidor.starttls(context=ssl.create_default_context())
                servidor.ehlo()
                servidor.login(smtp["usuario"], smtp["password"])
                servidor.send_message(correo)
    except Exception as exc:
        rutas = guardar_notificacion_simulada({**base, "modo": "simulado", "error": str(exc)}, evento_id)
        return {
            **base,
            **rutas,
            "enviado": False,
            "modo": "error",
            "estado": "fallo_envio",
            "error": str(exc),
            "mensaje_estado": f"No se pudo enviar el correo: {exc}. Se guardo una copia simulada.",
        }

    return {
        **base,
        "adjuntos": adjuntados,
        "enviado": True,
        "modo": "smtp",
        "estado": "enviado",
        "mensaje_estado": f"Correo enviado a {destinatario}.",
    }


def notificar_evento_placa(
    destinatario: str,
    placa: str,
    metricas_calidad: dict,
    adjuntos: list[str] | None = None,
    config: dict | None = None,
    evento_id="evento",
    contexto: dict | None = None,
    velocidad_kmh: float | None = None,
    limite_kmh: float | None = None,
    forzar_envio: bool = False,
) -> dict:
    """Orquesta: candado de calidad -> mensaje difuso -> envio del correo.

    Devuelve un dict con el resultado del envio y la evaluacion de calidad.
    """
    cfg_calidad = _config_calidad(config)
    metricas_calidad = {**metricas_calidad, "placa": placa}
    evaluacion = evaluar_calidad_evento(metricas_calidad, config)

    permitir_no_apto = bool(forzar_envio or cfg_calidad.get("permitir_envio_si_no_apto"))
    if not evaluacion["apto"] and not permitir_no_apto:
        return {
            "enviado": False,
            "modo": "descartado",
            "estado": "evento_no_apto",
            "destinatario": destinatario,
            "evaluacion_calidad": evaluacion,
            "mensaje_estado": "Evento descartado por el candado de calidad; no se envia correo.",
        }

    notif_cfg = (config or {}).get("notificaciones") or {}
    velocidad_medida = velocidad_kmh
    if velocidad_medida is None:
        velocidad_medida = (contexto or {}).get("velocidad_kmh")

    limite = limite_kmh
    if limite is None:
        limite = (contexto or {}).get("limite_kmh")
    if limite is None:
        limite = float((config or {}).get("speed", {}).get("campus_speed_limit_kmh", LIMITE_DEMO_KMH))

    if velocidad_medida is None:
        difuso = {
            "velocidad_kmh": None,
            "limite_kmh": float(limite),
            "estado": "Sin medicion",
            "categoria_fuzzy": None,
            "nivel_infraccion": "No evaluada",
            "sancion": "No aplica",
            "multa_usd": 0.0,
            "multa_texto": "Sin multa (velocidad no medida)",
            "horas_suspension": 0,
            "sancion_aplica": False,
            "mensaje": "No se registro velocidad en el evento; no se calcula multa.",
            "grados": {},
            "metodo": "sin_velocidad",
            "velocidad_es_medida": False,
        }
    else:
        difuso = construir_resultado_difuso(float(velocidad_medida), float(limite), config)
        difuso["velocidad_es_medida"] = True
    mensaje = construir_mensaje_sancion(placa, difuso, contexto)
    asunto = f"Fotorradar Ecuador IA - Notificacion de placa {placa}"

    resultado = enviar_correo(
        destinatario=destinatario,
        asunto=asunto,
        mensaje=mensaje,
        adjuntos=adjuntos,
        config=config,
        evento_id=evento_id,
    )
    resultado["evaluacion_calidad"] = evaluacion
    resultado["clasificacion_difusa"] = difuso
    return resultado

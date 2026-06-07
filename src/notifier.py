"""Sistema de notificaciones por correo del Fotorradar.

Responsabilidades:
- Validar el correo destino (incluye correos institucionales, p. ej. @uni.edu.ec).
- Aplicar un "candado de calidad" que evita enviar notificaciones de placas
  incompletas (media placa) o lecturas poco confiables.
- Construir el mensaje de la sancion usando exclusivamente la velocidad medida
  y la clasificacion difusa del mismo evento.
- Enviar el correo por SMTP real; si no hay credenciales configuradas, cae en un
  modo simulado que guarda el correo y los adjuntos en disco (no rompe la demo).
"""

from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from html import escape
import json
import mimetypes
import os
import re
import smtplib
import ssl


from src.utils import cargar_variables_entorno

cargar_variables_entorno()

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


def _texto_velocidad_kmh(valor, *, pendiente: str = "No medida") -> str:
    if valor is None:
        return pendiente
    try:
        return f"{float(valor):.2f} km/h"
    except (TypeError, ValueError):
        return pendiente


def _texto_numero(valor, sufijo: str, decimales: int = 2, pendiente: str = "No disponible") -> str:
    if valor is None:
        return pendiente
    try:
        return f"{float(valor):.{decimales}f} {sufijo}".strip()
    except (TypeError, ValueError):
        return pendiente


def _texto_porcentaje(valor, pendiente: str = "No disponible") -> str:
    if valor is None:
        return pendiente
    try:
        return f"{float(valor):.0%}"
    except (TypeError, ValueError):
        return pendiente


def construir_mensaje_sancion(placa: str, resultado_difuso: dict, contexto: dict | None = None) -> str:
    contexto = contexto or {}
    sancion_aplica = bool(resultado_difuso.get("sancion_aplica"))
    fecha = contexto.get("fecha_hora") or "No disponible"

    encabezado = (
        "Estimado/a usuario,\n\n"
        f"El sistema Fotorradar Ecuador IA registro un evento asociado a la placa {placa}.\n\n"
    )

    datos_evento = (
        "Datos del evento:\n"
        f"  Fecha y hora: {fecha}\n"
        f"  Placa reconocida: {placa}\n"
        f"  Velocidad registrada: {_texto_velocidad_kmh(resultado_difuso.get('velocidad_kmh'))}\n"
        f"  Limite permitido: {_texto_velocidad_kmh(resultado_difuso.get('limite_kmh'), pendiente='Pendiente')}\n"
        f"  Distancia calibrada: {_texto_numero(contexto.get('distancia_metros'), 'm')}\n"
        f"  Tiempo entre lineas: {_texto_numero(contexto.get('tiempo_entre_lineas'), 's', 4)}\n"
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
        "La placa, velocidad, distancia, sancion y evidencia corresponden al mismo evento "
        "procesado automaticamente por el sistema.\n\n"
        "Sistema Fotorradar Ecuador IA."
    )

    return encabezado + datos_evento + cuerpo_sancion + cierre


def construir_mensaje_sancion_html(
    placa: str,
    resultado_difuso: dict,
    contexto: dict | None = None,
    *,
    incluir_evidencia_embebida: bool = True,
) -> str:
    """Plantilla HTML responsive construida con los datos reales del evento."""
    contexto = contexto or {}
    sancion_aplica = bool(resultado_difuso.get("sancion_aplica"))
    color_estado = "#b42318" if sancion_aplica else "#067647"
    fondo_estado = "#fef3f2" if sancion_aplica else "#ecfdf3"
    placa_html = escape(str(placa))
    fecha = escape(str(contexto.get("fecha_hora") or "No disponible"))
    evento_id = escape(str(contexto.get("evento_id") or "No disponible"))
    fuente = escape(str(contexto.get("fuente") or "Monitoreo"))
    velocidad = escape(_texto_velocidad_kmh(resultado_difuso.get("velocidad_kmh")))
    limite = escape(_texto_velocidad_kmh(resultado_difuso.get("limite_kmh"), pendiente="No disponible"))
    distancia = escape(_texto_numero(contexto.get("distancia_metros"), "m"))
    tiempo = escape(_texto_numero(contexto.get("tiempo_entre_lineas"), "s", 4))
    confianza_ocr = escape(_texto_porcentaje(contexto.get("confianza_ocr")))
    confianza_yolo = escape(_texto_porcentaje(contexto.get("confianza_yolo")))
    nivel = escape(str(resultado_difuso.get("nivel_infraccion") or "No evaluada"))
    estado = escape(str(resultado_difuso.get("estado") or "No evaluada"))
    multa = escape(str(resultado_difuso.get("multa_texto") or "Sin multa"))
    sancion = escape(str(resultado_difuso.get("sancion") or "No aplica"))
    detalle = escape(str(resultado_difuso.get("mensaje") or ""))
    horas = escape(str(resultado_difuso.get("horas_suspension", 0)))
    evidencia = (
        '<img src="cid:evidencia_evento" alt="Evidencia fotografica del evento" '
        'style="display:block;width:100%;max-width:620px;height:auto;border-radius:10px;border:1px solid #d0d5dd;">'
        if incluir_evidencia_embebida
        else '<div style="padding:24px;text-align:center;color:#667085;background:#f2f4f7;border-radius:10px;">Evidencia adjunta al correo</div>'
    )
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Notificacion Fotorradar Ecuador IA</title>
</head>
<body style="margin:0;padding:0;background:#eef2f6;font-family:Arial,Helvetica,sans-serif;color:#101828;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#eef2f6;padding:28px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:680px;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 8px 28px rgba(16,24,40,.12);">
        <tr>
          <td style="padding:26px 32px;background:#12324b;color:#ffffff;border-bottom:5px solid #2f80ed;">
            <div style="font-size:12px;letter-spacing:1.4px;text-transform:uppercase;color:#b9d8f2;">Sistema institucional de monitoreo</div>
            <div style="font-size:25px;font-weight:700;margin-top:7px;">Fotorradar Ecuador IA</div>
            <div style="font-size:14px;color:#d7e7f4;margin-top:6px;">Notificacion automatica de evento vehicular</div>
          </td>
        </tr>
        <tr>
          <td style="padding:30px 32px 12px;">
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
              <tr>
                <td style="vertical-align:top;">
                  <div style="font-size:13px;color:#667085;">Placa reconocida</div>
                  <div style="font-size:32px;font-weight:800;letter-spacing:2px;color:#12324b;margin-top:5px;">{placa_html}</div>
                </td>
                <td align="right" style="vertical-align:top;">
                  <span style="display:inline-block;padding:8px 12px;border-radius:999px;background:{fondo_estado};color:{color_estado};font-size:13px;font-weight:700;">{nivel}</span>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:12px 32px;">
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
              <tr>
                <td width="48%" style="padding:20px;background:#eff8ff;border:1px solid #b2ddff;border-radius:10px;">
                  <div style="font-size:12px;color:#175cd3;text-transform:uppercase;font-weight:700;">Velocidad registrada</div>
                  <div style="font-size:31px;font-weight:800;color:#1849a9;margin-top:8px;">{velocidad}</div>
                  <div style="font-size:13px;color:#475467;margin-top:7px;">Limite configurado: {limite}</div>
                </td>
                <td width="4%"></td>
                <td width="48%" style="padding:20px;background:{fondo_estado};border:1px solid {color_estado};border-radius:10px;">
                  <div style="font-size:12px;color:{color_estado};text-transform:uppercase;font-weight:700;">Resultado difuso</div>
                  <div style="font-size:21px;font-weight:800;color:{color_estado};margin-top:8px;">{estado}</div>
                  <div style="font-size:13px;color:#475467;margin-top:7px;">{multa}</div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:12px 32px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:12px;">Datos reales del evento</div>
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border:1px solid #eaecf0;border-radius:10px;">
              <tr><td style="padding:11px 14px;color:#667085;border-bottom:1px solid #eaecf0;">Fecha y hora</td><td align="right" style="padding:11px 14px;font-weight:700;border-bottom:1px solid #eaecf0;">{fecha}</td></tr>
              <tr><td style="padding:11px 14px;color:#667085;border-bottom:1px solid #eaecf0;">Evento / fuente</td><td align="right" style="padding:11px 14px;font-weight:700;border-bottom:1px solid #eaecf0;">#{evento_id} · {fuente}</td></tr>
              <tr><td style="padding:11px 14px;color:#667085;border-bottom:1px solid #eaecf0;">Distancia entre lineas</td><td align="right" style="padding:11px 14px;font-weight:700;border-bottom:1px solid #eaecf0;">{distancia}</td></tr>
              <tr><td style="padding:11px 14px;color:#667085;border-bottom:1px solid #eaecf0;">Tiempo entre cruces</td><td align="right" style="padding:11px 14px;font-weight:700;border-bottom:1px solid #eaecf0;">{tiempo}</td></tr>
              <tr><td style="padding:11px 14px;color:#667085;border-bottom:1px solid #eaecf0;">Confianza lector CNN</td><td align="right" style="padding:11px 14px;font-weight:700;border-bottom:1px solid #eaecf0;">{confianza_ocr}</td></tr>
              <tr><td style="padding:11px 14px;color:#667085;">Confianza detector YOLO</td><td align="right" style="padding:11px 14px;font-weight:700;">{confianza_yolo}</td></tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:12px 32px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:12px;">Multa y sancion determinada</div>
            <div style="padding:18px;background:#f9fafb;border-left:4px solid {color_estado};border-radius:8px;">
              <div style="font-weight:700;color:#344054;">{sancion}</div>
              <div style="font-size:14px;color:#475467;margin-top:7px;">{multa} · Suspension: {horas} h</div>
              <div style="font-size:13px;color:#667085;margin-top:10px;">{detalle}</div>
            </div>
          </td>
        </tr>
        <tr>
          <td style="padding:12px 32px 30px;">
            <div style="font-size:17px;font-weight:700;margin-bottom:12px;">Evidencia fotografica seleccionada</div>
            {evidencia}
            <div style="font-size:12px;color:#667085;margin-top:9px;">Se utiliza el mejor frame disponible asociado al mismo evento vehicular.</div>
          </td>
        </tr>
        <tr>
          <td style="padding:20px 32px;background:#12324b;color:#d7e7f4;text-align:center;font-size:12px;line-height:1.6;">
            Mensaje generado automaticamente por Fotorradar Ecuador IA.<br>
            Los datos mostrados provienen del procesamiento real del evento #{evento_id}.
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


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


def _incrustar_evidencia_html(mensaje: EmailMessage, ruta_evidencia: str | None) -> str | None:
    if not ruta_evidencia:
        return None
    ruta = Path(ruta_evidencia)
    if not ruta.exists():
        return None
    tipo, _ = mimetypes.guess_type(str(ruta))
    if not tipo or not tipo.startswith("image/"):
        return None
    maintype, subtype = tipo.split("/", 1)
    html_part = mensaje.get_payload()[-1]
    html_part.add_related(
        ruta.read_bytes(),
        maintype=maintype,
        subtype=subtype,
        cid="<evidencia_evento>",
        filename=ruta.name,
    )
    return str(ruta)


def guardar_notificacion_simulada(notificacion: dict, evento_id) -> dict:
    NOTIFICACIONES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_txt = NOTIFICACIONES_DIR / f"notificacion_evento_{evento_id}.txt"
    ruta_html = NOTIFICACIONES_DIR / f"notificacion_evento_{evento_id}.html"
    ruta_json = NOTIFICACIONES_DIR / f"notificacion_evento_{evento_id}.json"
    ruta_txt.write_text(notificacion.get("mensaje", ""), encoding="utf-8")
    ruta_html.write_text(notificacion.get("mensaje_html", ""), encoding="utf-8")
    ruta_json.write_text(json.dumps(notificacion, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ruta_txt": str(ruta_txt), "ruta_html": str(ruta_html), "ruta_json": str(ruta_json)}


def enviar_correo(
    destinatario: str,
    asunto: str,
    mensaje: str,
    mensaje_html: str | None = None,
    adjuntos: list[str] | None = None,
    evidencia_principal: str | None = None,
    contexto_evento: dict | None = None,
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
            "mensaje_html": mensaje_html,
            "error": "El correo destino no es valido.",
        }

    smtp = _resolver_config_smtp(config)
    adjuntos = adjuntos or []
    hay_credenciales = bool(smtp["host"] and smtp["usuario"] and smtp["password"] and smtp["remitente"])

    base = {
        "destinatario": destinatario,
        "asunto": asunto,
        "mensaje": mensaje,
        "mensaje_html": mensaje_html,
        "adjuntos": adjuntos,
        "evidencia_principal": evidencia_principal,
        "contexto_evento": dict(contexto_evento or {}),
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
    evidencia_embebida = None
    if mensaje_html:
        correo.add_alternative(mensaje_html, subtype="html")
        evidencia_embebida = _incrustar_evidencia_html(correo, evidencia_principal)
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
        "evidencia_embebida": evidencia_embebida,
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
    """Orquesta un correo solo cuando el evento tiene datos reales completos.

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

    velocidad_medida = velocidad_kmh
    if velocidad_medida is None:
        velocidad_medida = (contexto or {}).get("velocidad_kmh")
    if velocidad_medida is None:
        return {
            "enviado": False,
            "modo": "bloqueado",
            "estado": "velocidad_no_medida",
            "destinatario": destinatario,
            "evaluacion_calidad": evaluacion,
            "mensaje_estado": "No se envia correo: el evento no tiene una velocidad real calculada.",
        }

    limite = limite_kmh
    if limite is None:
        limite = (contexto or {}).get("limite_kmh")
    if limite is None:
        limite = (config or {}).get("speed", {}).get("campus_speed_limit_kmh")
    if limite is None:
        return {
            "enviado": False,
            "modo": "bloqueado",
            "estado": "limite_no_configurado",
            "destinatario": destinatario,
            "evaluacion_calidad": evaluacion,
            "mensaje_estado": "No se envia correo: no existe limite de velocidad configurado para evaluar el evento.",
        }
    if not (contexto or {}).get("fecha_hora"):
        return {
            "enviado": False,
            "modo": "bloqueado",
            "estado": "fecha_evento_no_disponible",
            "destinatario": destinatario,
            "evaluacion_calidad": evaluacion,
            "mensaje_estado": "No se envia correo: el evento no tiene fecha y hora registradas.",
        }
    evidencia_principal = (contexto or {}).get("evidencia_principal")
    if not evidencia_principal or not Path(str(evidencia_principal)).exists():
        return {
            "enviado": False,
            "modo": "bloqueado",
            "estado": "evidencia_no_disponible",
            "destinatario": destinatario,
            "evaluacion_calidad": evaluacion,
            "mensaje_estado": "No se envia correo: no existe evidencia fotografica real del evento.",
        }

    difuso = construir_resultado_difuso(float(velocidad_medida), float(limite), config)
    difuso["velocidad_es_medida"] = True
    mensaje = construir_mensaje_sancion(placa, difuso, contexto)
    mensaje_html = construir_mensaje_sancion_html(placa, difuso, contexto)
    asunto = f"Fotorradar Ecuador IA - Notificacion de placa {placa}"

    resultado = enviar_correo(
        destinatario=destinatario,
        asunto=asunto,
        mensaje=mensaje,
        mensaje_html=mensaje_html,
        adjuntos=adjuntos,
        evidencia_principal=str(evidencia_principal),
        contexto_evento=contexto,
        config=config,
        evento_id=evento_id,
    )
    resultado["evaluacion_calidad"] = evaluacion
    resultado["clasificacion_difusa"] = difuso
    resultado["contexto_evento"] = dict(contexto or {})
    return resultado

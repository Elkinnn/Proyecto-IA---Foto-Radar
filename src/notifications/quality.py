"""Reglas de calidad para decidir si un evento debe notificarse."""

from __future__ import annotations

import re


EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

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
    return bool(EMAIL_REGEX.match(correo.strip()))


def _config_calidad(config: dict | None) -> dict:
    calidad = dict(DEFAULT_CALIDAD)
    if config:
        notif = (config.get("notificaciones") or {}).get("calidad") or {}
        calidad.update({k: v for k, v in notif.items() if v is not None})
    return calidad


def evaluar_calidad_evento(metricas: dict, config: dict | None = None) -> dict:
    """Candado de calidad: decide si un evento merece enviar correo."""
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


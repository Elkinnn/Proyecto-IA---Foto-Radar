"""Resumen de eventos de monitoreo para UI, fuzzy y notificaciones.

Funciones sin renderizado Streamlit: extraen placa, velocidad, evidencias y
datos reales del evento para mantener `app.py` enfocado en la interfaz.
"""

from datetime import datetime
from pathlib import Path
import json

from src.fuzzy_system import clasificar_velocidad


def _metricas_calidad_desde_resumen(resumen: dict, placa: str) -> dict:
    mejor_info = resumen.get("mejor_recorte_placa_info") or {}
    cantidad = resumen.get("cantidad_caracteres_segmentados_ocr")
    return {
        "placa": placa,
        "formato_valido": bool(resumen.get("formato_consolidado_valido") or resumen.get("formato_ocr_valido")),
        "confianza_final": resumen.get("confianza_final_evento") or resumen.get("confianza_ocr"),
        "aspect_ratio": mejor_info.get("aspect_ratio"),
        "nitidez": mejor_info.get("nitidez"),
        "cerca_borde": mejor_info.get("cerca_borde"),
        "cantidad_caracteres": cantidad if cantidad else len(placa or ""),
    }

def _adjuntos_evento_desde_resumen(resumen: dict) -> list[str]:
    candidatos_frame = [
        resumen.get("ruta_mejor_frame_evento"),
        resumen.get("ruta_snapshot_frame_evento"),
        resumen.get("mejor_frame_bbox_placa"),
        resumen.get("ruta_frame_evento_en_vivo"),
        resumen.get("mejor_frame_recorte_placa"),
        resumen.get("ultimo_frame_deteccion"),
    ]
    candidatos_recorte = [
        resumen.get("ruta_mejor_recorte_evento"),
        resumen.get("mejor_recorte_placa"),
        resumen.get("ruta_recorte_evento_en_vivo"),
        resumen.get("ultimo_recorte_placa"),
    ]

    def _primera_ruta_valida(rutas: list) -> str | None:
        for ruta in rutas:
            if not ruta:
                continue
            ruta_str = str(ruta)
            if Path(ruta_str).exists():
                return ruta_str
        return None

    frame = _primera_ruta_valida(candidatos_frame)
    recorte = _primera_ruta_valida(candidatos_recorte)

    adjuntos: list[str] = []
    if frame:
        adjuntos.append(frame)
    if recorte and recorte not in adjuntos:
        adjuntos.append(recorte)
    return adjuntos

def _mejor_frame_evento_desde_resumen(resumen: dict) -> str | None:
    """Devuelve la evidencia completa seleccionada para el evento."""
    for clave in (
        "ruta_mejor_frame_evento",
        "ruta_snapshot_frame_evento",
        "mejor_frame_bbox_placa",
        "ruta_frame_evento_en_vivo",
    ):
        ruta = resumen.get(clave)
        if ruta and Path(str(ruta)).exists():
            return str(ruta)
    return None

def _contexto_notificacion_desde_resumen(resumen: dict, adjuntos: list[str], config: dict | None = None) -> dict:
    """Extrae solo datos medidos y evidencias pertenecientes al mismo evento."""
    velocidad = resumen.get("velocidad") or {}
    ultima_deteccion = resumen.get("ultima_deteccion") or {}
    mejor_frame = _mejor_frame_evento_desde_resumen(resumen)
    return {
        "evento_id": resumen.get("evento_id") or resumen.get("eventos_placa"),
        "fecha_hora": resumen.get("fecha_hora_evento") or resumen.get("fecha_hora"),
        "fuente": resumen.get("fuente"),
        "velocidad_kmh": _obtener_velocidad_kmh_desde_resumen(resumen),
        "limite_kmh": _obtener_limite_kmh_resumen(resumen, config),
        "distancia_metros": velocidad.get("distancia_metros") or resumen.get("distancia_lineas_m"),
        "tiempo_entre_lineas": velocidad.get("tiempo_entre_lineas"),
        "frame_linea_1": velocidad.get("frame_cruce_linea_1"),
        "frame_linea_2": velocidad.get("frame_cruce_linea_2"),
        "metodo_medicion": velocidad.get("metodo_medicion"),
        "confianza_ocr": resumen.get("confianza_final_evento") or resumen.get("confianza_ocr"),
        "confianza_yolo": resumen.get("mejor_confianza_evento") or ultima_deteccion.get("confianza"),
        "evidencia_principal": mejor_frame,
    }

def _tiene_recorte_placa_valido(resumen: dict) -> bool:
    ruta = (
        resumen.get("mejor_recorte_placa")
        or resumen.get("ruta_mejor_recorte_evento")
        or resumen.get("ultimo_recorte_placa")
    )
    return bool(ruta and Path(str(ruta)).exists())

_ERRORES_LECTURA_OCR_UI = frozenset(
    {
        "PENDIENTE",
        "ANALIZANDOPLACA...",
        "SINLECTURA",
        "SINPLACA",
        "FORMATO_INVALIDO",
        "SEGMENTACION_INCOMPLETA",
        "SIN_CARACTERES",
        "ERROR_CNN",
        "NO_RECONOCIDO",
        "FORMATODUDOSO",
    }
)

def _es_codigo_error_lectura(texto: str) -> bool:
    normalizado = str(texto or "").strip().upper().replace("-", "").replace(" ", "").replace("_", "")
    return normalizado in _ERRORES_LECTURA_OCR_UI or normalizado.startswith("SINLECTURA")

def _extraer_placa_desde_resumen(resumen: dict) -> str:
    """Mejor lectura OCR disponible (consolidada o individual)."""
    candidatos = (
        resumen.get("placa_consolidada_evento"),
        resumen.get("placa_individual"),
        resumen.get("texto_ocr_corregido"),
        resumen.get("texto_ocr_crudo"),
    )
    for candidato in candidatos:
        if not candidato:
            continue
        texto = str(candidato).strip()
        for prefijo in ("Lectura parcial: ", "Formato dudoso: ", "Sin lectura: "):
            if texto.startswith(prefijo):
                texto = texto[len(prefijo) :].strip()
        if _es_codigo_error_lectura(texto):
            continue
        normalizado = texto.upper().replace("-", "").replace(" ", "")
        if normalizado and normalizado not in _ERRORES_LECTURA_OCR_UI:
            return normalizado
    return ""

def _extraer_placa_individual_desde_resumen(resumen: dict) -> str:
    """Lectura que corresponde al recorte y predicciones CNN mostrados."""
    predicciones = resumen.get("predicciones_caracteres_ocr") or []
    texto_predicciones = "".join(
        str(pred.get("caracter_predicho") or "").strip()
        for pred in predicciones
    )
    candidatos = (
        texto_predicciones,
        resumen.get("placa_individual_ultimo_frame"),
        resumen.get("placa_individual"),
    )
    for candidato in candidatos:
        texto = str(candidato or "").strip().upper().replace("-", "").replace(" ", "")
        if _es_codigo_error_lectura(texto):
            continue
        if len(texto) in (6, 7) and texto.isalnum():
            return texto
    return ""

def _obtener_velocidad_kmh_desde_resumen(resumen: dict) -> float | None:
    if resumen.get("velocidad_kmh") is not None:
        return float(resumen["velocidad_kmh"])
    velocidad = resumen.get("velocidad") or {}
    if velocidad.get("velocidad_kmh") is not None:
        return float(velocidad["velocidad_kmh"])
    return None

def _obtener_limite_kmh_resumen(resumen: dict, config: dict | None) -> float:
    if resumen.get("limite_velocidad_kmh") is not None:
        return float(resumen["limite_velocidad_kmh"])
    if config:
        return float((config.get("speed") or {}).get("campus_speed_limit_kmh", 30.0))
    return 30.0

def _aplicar_multa_difusa_resumen(resumen: dict, config: dict | None) -> dict:
    kmh = _obtener_velocidad_kmh_desde_resumen(resumen)
    if kmh is None:
        return resumen
    salida = dict(resumen)
    limite = _obtener_limite_kmh_resumen(salida, config)
    salida["clasificacion_difusa"] = clasificar_velocidad(kmh, limite, config)
    salida["velocidad_kmh"] = kmh
    salida.setdefault("limite_velocidad_kmh", limite)
    _guardar_resultado_difuso_evento(salida)
    return salida

def _guardar_resultado_difuso_evento(resumen: dict) -> str | None:
    """Persiste una evidencia compacta de inferencia para cada evento medido."""
    resultado = resumen.get("clasificacion_difusa") or {}
    evento_id = resumen.get("evento_id") or resumen.get("eventos_placa")
    if not evento_id or resultado.get("velocidad_kmh") is None:
        return None
    directorio = Path("reports") / "evidencias" / "logica_difusa" / "eventos"
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"evento_{int(evento_id):04d}_inferencia.json"
    payload = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "evento_id": int(evento_id),
        "placa": _extraer_placa_desde_resumen(resumen) or None,
        "fuente": resumen.get("fuente"),
        "resultado": resultado,
    }
    ruta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    resumen["ruta_resultado_difuso"] = str(ruta)
    return str(ruta)


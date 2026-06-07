"""Decisiones puras de cache y consolidacion OCR del monitoreo."""

from src.monitoring.event_summary import _extraer_placa_desde_resumen


def _entrada_ocr_tiene_texto_util(entry: dict) -> bool:
    return bool(_extraer_placa_desde_resumen(entry.get("datos") or {}))

def _resultado_ocr_necesita_reintento(resultado: dict) -> bool:
    texto = (
        resultado.get("texto_postprocesado")
        or resultado.get("texto_corregido_formato")
        or resultado.get("texto_detectado_crudo")
        or resultado.get("texto_crudo")
        or ""
    )
    cantidad = int(resultado.get("cantidad_caracteres_segmentados", 0) or 0)
    predicciones = resultado.get("predicciones_caracteres") or []
    return not str(texto).strip() and cantidad >= 6 and not predicciones

def _puntaje_consolidado_evento(consolidado: dict | None) -> float:
    if not consolidado:
        return -1.0
    texto = str(consolidado.get("texto_final") or "")
    puntaje = float(consolidado.get("confianza_final") or 0.0) * 100.0
    if consolidado.get("formato_valido"):
        puntaje += 25.0
    if len(texto) == 7:
        puntaje += 10.0
    elif len(texto) == 6:
        puntaje += 5.0
    usadas = int(consolidado.get("cantidad_lecturas_usadas") or 0)
    puntaje += min(usadas, 5) * 4.0
    return puntaje

def _debe_usar_nuevo_consolidado(nuevo: dict | None, previo: dict | None) -> bool:
    """Evita conservar una placa vieja cuando la votacion actual ya es valida."""
    if not nuevo:
        return False
    texto_nuevo = str(nuevo.get("texto_final") or "").strip()
    if not texto_nuevo:
        return False
    if not previo or not str(previo.get("texto_final") or "").strip():
        return True
    if nuevo.get("formato_valido"):
        return True
    if previo.get("formato_valido"):
        return False
    return _puntaje_consolidado_evento(nuevo) >= _puntaje_consolidado_evento(previo)

def _lectura_existe_para_ruta(lecturas: list, ruta_recorte: str) -> bool:
    destino = str(ruta_recorte or "")
    if not destino:
        return False
    return any(str(item.get("ruta_recorte") or "") == destino for item in (lecturas or []))


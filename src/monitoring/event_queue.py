"""Agrupacion de eventos/vehiculos para evitar duplicados de correo."""

from src.monitoring.event_summary import _extraer_placa_desde_resumen


def _marcar_duplicados_placa_en_cola(cola: list) -> None:
    """Marca duplicados del mismo paso de vehiculo (ventana de frames)."""
    _consolidar_marcas_paso_vehiculo(cola, None)

def _puntaje_item_evento(item: dict) -> float:
    """Mayor puntaje = mejor candidato para enviar correo de ese paso."""
    res = item.get("resumen") or {}
    puntaje = 0.0
    if (res.get("evaluacion_calidad_evento") or {}).get("apto"):
        puntaje += 10.0
    if res.get("formato_consolidado_valido") or res.get("formato_ocr_valido"):
        puntaje += 5.0
    conf_ocr = res.get("confianza_final_evento") or res.get("confianza_ocr")
    if conf_ocr is not None:
        puntaje += float(conf_ocr) * 3.0
    conf_yolo = (res.get("ultima_deteccion") or {}).get("confianza") or res.get("mejor_confianza_evento")
    if conf_yolo is not None:
        puntaje += float(conf_yolo) * 1.0
    placa = _extraer_placa_desde_resumen(res)
    if placa and len(placa) == 7:
        puntaje += 1.5
    elif placa and len(placa) == 6:
        puntaje += 0.5
    if res.get("estado_ocr") == "error":
        puntaje -= 5.0
    return puntaje

def _agrupar_cola_por_paso_vehiculo(cola: list, ventana_frames: int) -> list[dict]:
    ordenados = sorted(
        [item for item in cola if item.get("frame_mejor") is not None or (item.get("resumen") or {}).get("frame_mejor_evento")],
        key=lambda item: int(item.get("frame_mejor") or (item.get("resumen") or {}).get("frame_mejor_evento") or 0),
    )
    grupos: list[dict] = []
    for item in ordenados:
        frame = int(item.get("frame_mejor") or (item.get("resumen") or {}).get("frame_mejor_evento") or 0)
        evento_id = int(item.get("evento_id") or 0)
        placa = _extraer_placa_desde_resumen(item.get("resumen") or {})
        ubicado = False
        for grupo in grupos:
            mismo_evento = bool(evento_id and evento_id in grupo["eventos"])
            misma_placa = bool(placa and placa in grupo["placas"])
            if (mismo_evento or misma_placa) and abs(frame - int(grupo["frame_fin"])) <= ventana_frames:
                grupo["items"].append(item)
                grupo["frame_fin"] = max(int(grupo["frame_fin"]), frame)
                grupo["frame_inicio"] = min(int(grupo["frame_inicio"]), frame)
                if evento_id:
                    grupo["eventos"].add(evento_id)
                if placa:
                    grupo["placas"].add(placa)
                ubicado = True
                break
        if not ubicado:
            grupos.append(
                {
                    "frame_inicio": frame,
                    "frame_fin": frame,
                    "items": [item],
                    "eventos": {evento_id} if evento_id else set(),
                    "placas": {placa} if placa else set(),
                }
            )
    return grupos

def _grupo_listo_para_envio(grupo: dict, frame_actual: int, cooldown_frames: int) -> bool:
    items = grupo.get("items") or []
    if not items:
        return False
    if any(item.get("estado_ocr") == "pendiente" for item in items):
        return False
    if int(cooldown_frames) <= 0:
        return True
    ultimo_frame = max(
        int(item.get("frame_mejor") or (item.get("resumen") or {}).get("frame_mejor_evento") or 0)
        for item in items
    )
    return int(frame_actual or 0) - ultimo_frame >= int(cooldown_frames)

def _consolidar_marcas_paso_vehiculo(cola: list, config: dict | None) -> None:
    ventana = 150
    if config:
        ventana = int((config.get("notificaciones") or {}).get("ventana_frames_mismo_paso", ventana))
    for item in cola:
        item.pop("es_mejor_del_paso", None)
        item.pop("suprimido_duplicado", None)
    for grupo in _agrupar_cola_por_paso_vehiculo(cola, ventana):
        paso_id = f"paso_{grupo['frame_inicio']}"
        items_listos = [i for i in grupo["items"] if i.get("estado_ocr") == "listo"]
        if not items_listos:
            for item in grupo["items"]:
                item["paso_grupo_id"] = paso_id
            continue
        mejor = max(items_listos, key=_puntaje_item_evento)
        for item in grupo["items"]:
            item["paso_grupo_id"] = paso_id
            if item is mejor:
                item["es_mejor_del_paso"] = True
            else:
                item["suprimido_duplicado"] = True


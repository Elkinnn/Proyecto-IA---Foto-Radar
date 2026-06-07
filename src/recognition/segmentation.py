"""Segmentacion de caracteres de placas ecuatorianas.

Agrupa extraccion de banda, componentes, slots y estrategias de segmentacion
usadas por el lector CNN.
"""

from pathlib import Path
import json

import cv2
import numpy as np

from src.recognition.character_normalization import normalizar_imagen_caracter_para_cnn
from src.recognition.image_utils import _imread_seguro, asegurar_bgr, asegurar_grayscale
from src.recognition.naming import _nombre_seguro
from src.recognition.postprocessing import _tipo_posicion_placa
from src.recognition.preprocessing_filters import (
    _asegurar_caracteres_blancos,
    _quitar_lineas_finas_marco,
    _score_orientacion_caracteres,
)


RECONOCIMIENTO_CARACTERES_DIR = Path("reports") / "evidencias" / "reconocimiento_caracteres"
OCR_DIR = RECONOCIMIENTO_CARACTERES_DIR
PREPROCESADAS_DIR = OCR_DIR / "placas_preprocesadas"
CARACTERES_DIR = OCR_DIR / "caracteres_segmentados"
BANDAS_DIR = OCR_DIR / "bandas_caracteres"
DEBUG_SEGMENTACION_DIR = OCR_DIR / "debug_segmentacion"
DEBUG_SLOTS_ECUADOR_DIR = OCR_DIR / "debug_slots_ecuador"
Y_INICIO_BANDA = 0.34
Y_FIN_BANDA = 0.95
ALTURA_MIN_CARACTER_REL = 0.35
ALTURA_MAX_CARACTER_REL = 0.82
ANCHO_MIN_CARACTER_REL = 0.015
ANCHO_MAX_CARACTER_REL = 0.25
ASPECT_MIN_CARACTER = 0.15
ASPECT_MAX_CARACTER = 1.20
AREA_MIN_CARACTER_REL = 0.005
AREA_MAX_CARACTER_REL = 0.40
CARACTERES_MIN_PLACA = 6
CARACTERES_MAX_PLACA = 7


def segmentar_caracteres(ruta_imagen_procesada: str, nombre_base: str = "placa") -> list[dict]:
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    imagen = _imread_seguro(str(ruta_imagen_procesada), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return []

    alto_img, ancho_img = imagen.shape[:2]
    candidatos = []

    for binaria in (imagen, cv2.bitwise_not(imagen)):
        morfologia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contornos, _ = cv2.findContours(morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            area = w * h
            if area < (ancho_img * alto_img) * 0.001:
                continue
            if h < alto_img * 0.18 or h > alto_img * 0.98:
                continue
            if w < ancho_img * 0.008 or w > ancho_img * 0.35:
                continue
            proporcion = w / max(h, 1)
            if proporcion < 0.08 or proporcion > 1.45:
                continue
            candidatos.append((x, y, w, h))

    candidatos = _eliminar_bboxes_duplicadas(candidatos)

    candidatos.sort(key=lambda bbox: bbox[0])
    nombre = _nombre_seguro(ruta_imagen_procesada, nombre_base)
    caracteres = []

    for idx, (x, y, w, h) in enumerate(candidatos, start=1):
        margen = 3
        x1 = max(x - margen, 0)
        y1 = max(y - margen, 0)
        x2 = min(x + w + margen, ancho_img)
        y2 = min(y + h + margen, alto_img)
        recorte = binaria[y1:y2, x1:x2]
        ruta_caracter = CARACTERES_DIR / f"{nombre}_char_{idx:02d}.jpg"
        cv2.imwrite(str(ruta_caracter), recorte)
        caracteres.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "ruta_caracter": str(ruta_caracter),
                "ancho": int(x2 - x1),
                "alto": int(y2 - y1),
            }
        )

    return caracteres

def extraer_banda_caracteres(imagen_preprocesada, nombre_base: str = "placa") -> dict:
    BANDAS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(imagen_preprocesada, (str, Path)):
        imagen = _imread_seguro(str(imagen_preprocesada), cv2.IMREAD_GRAYSCALE)
    else:
        imagen = imagen_preprocesada

    if imagen is None:
        return {
            "ruta_banda": None,
            "banda": None,
            "bbox_banda": None,
            "estado": "error",
            "mensaje": "No se pudo cargar la imagen preprocesada para extraer la banda.",
        }

    imagen = _asegurar_caracteres_blancos(imagen)
    alto, ancho = imagen.shape[:2]
    metodo_banda = "mitad_inferior_directa"
    componentes_grandes = []
    mediana_altura_componentes = 0.0
    fix_borde_inferior_aplicado = False
    y2_original_borde_inferior = None
    y2_ajustado_borde_inferior = None

    # FIX-TOP: detectar la banda en TODA la altura del recorte. Antes se descartaba
    # a ciegas el 35% superior asumiendo la cabecera "ECUADOR"; cuando el recorte de
    # YOLO es ajustado eso decapitaba los caracteres. Ahora la cabecera se excluye
    # solo si realmente existe (texto pequeno de baja altura), porque los caracteres
    # principales son los componentes mas altos.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(imagen, 8)
    componentes = []
    area_min = max(40, int(alto * ancho * 0.0008))
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < area_min:
            continue
        aspect = w / max(h, 1)
        if aspect > 4.5 and h < alto * 0.12:
            continue
        if aspect < 0.10 and h > alto * 0.70:
            continue
        componentes.append((x, y, w, h, area))

    y1 = int(alto * 0.25)
    y2 = int(alto * 0.95)

    if len(componentes) >= 3:
        alturas = np.array([h for _, _, _, h, _ in componentes], dtype=np.float32)
        mediana_altura_componentes = float(np.median(alturas))
        altura_max = float(np.max(alturas))
        umbral_altura = max(0.55 * altura_max, alto * 0.28)
        componentes_grandes = [
            (x, y, w, h)
            for x, y, w, h, _ in componentes
            if h >= umbral_altura
        ]
        if len(componentes_grandes) >= 3:
            y_min = min(y for _, y, _, _ in componentes_grandes)
            y_max = max(y + h for _, y, _, h in componentes_grandes)
            altura_chars = max(y_max - y_min, 1)
            # padding superior generoso para conservar trazos altos del caracter.
            pad_sup = max(2, int(altura_chars * 0.12))
            pad_inf = max(2, int(altura_chars * 0.08))
            y1 = max(0, y_min - pad_sup)
            y2 = min(alto, y_max + pad_inf)
            metodo_banda = "componentes_caracteres_global"

    # Validar altura de banda; si es absurda, usar fallback amplio que NO recorta arriba.
    altura_banda_rel = (y2 - y1) / max(alto, 1)
    if altura_banda_rel < 0.15 or altura_banda_rel > 0.85:
        y1 = int(alto * 0.22)
        y2 = int(alto * 0.95)
        metodo_banda = "fallback_amplio_validacion_altura"

    # FIX-BORDE-INFERIOR: evitar incluir el borde metalico inferior del recorte YOLO.
    y2_original_borde_inferior = int(y2)
    y2_techo = int(alto * 0.97)
    if y2 > y2_techo and (y2_techo - y1) > int(alto * 0.18):
        y2 = y2_techo
        fix_borde_inferior_aplicado = True
    y2_ajustado_borde_inferior = int(y2)

    margen_x = max(int(ancho * 0.035), 6)
    banda = imagen[y1:y2, :].copy()
    banda = cv2.copyMakeBorder(banda, 2, 2, margen_x, margen_x, cv2.BORDER_CONSTANT, value=0)
    nombre = _nombre_seguro(str(nombre_base), "placa")
    ruta_banda = BANDAS_DIR / f"{nombre}_banda_caracteres.jpg"
    cv2.imwrite(str(ruta_banda), banda)

    return {
        "ruta_banda": str(ruta_banda),
        "banda": banda,
        "bbox_banda": [0, y1, ancho, y2],
        "metodo_banda": metodo_banda,
        "componentes_grandes": len(componentes_grandes),
        "mediana_altura_componentes": round(float(mediana_altura_componentes), 4),
        "fix_borde_inferior_aplicado": bool(fix_borde_inferior_aplicado),
        "y2_original": int(y2_original_borde_inferior),
        "y2_ajustado": int(y2_ajustado_borde_inferior),
        "estado": "ok",
        "mensaje": f"Banda principal de caracteres extraida con metodo {metodo_banda}.",
    }

def segmentar_caracteres_v2(ruta_imagen_procesada: str, nombre_base: str = "placa") -> dict:
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_SEGMENTACION_DIR.mkdir(parents=True, exist_ok=True)
    imagen = _imread_seguro(str(ruta_imagen_procesada), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return {
            "caracteres": [],
            "contornos_rechazados": [],
            "cantidad_aceptados": 0,
            "cantidad_rechazados": 0,
            "ruta_banda": None,
            "ruta_debug": None,
            "estado": "error",
            "mensaje": "No se pudo cargar la imagen procesada para segmentacion v2.",
        }

    nombre = _nombre_seguro(ruta_imagen_procesada, nombre_base)
    banda_info = extraer_banda_caracteres(imagen, nombre)
    banda = banda_info.get("banda")
    if banda is None:
        return {
            "caracteres": [],
            "contornos_rechazados": [],
            "cantidad_aceptados": 0,
            "cantidad_rechazados": 0,
            "ruta_banda": banda_info.get("ruta_banda"),
            "ruta_debug": None,
            "estado": "error",
            "mensaje": banda_info.get("mensaje"),
        }

    alto_banda, ancho_banda = banda.shape[:2]
    binaria = _asegurar_caracteres_blancos(banda)
    binaria = _limpiar_bordes_largos_banda(binaria)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    debug = asegurar_bgr(banda)
    aceptados = []
    rechazados = []
    area_banda = alto_banda * ancho_banda

    bboxes_componentes = _bboxes_componentes_conectados(binaria)
    bboxes_proyeccion = _bboxes_por_proyeccion_vertical(binaria)
    bboxes = _fusionar_bboxes(bboxes_componentes + bboxes_proyeccion)
    bboxes = _dividir_bboxes_demasiado_anchas(binaria, bboxes)

    for x, y, w, h in bboxes:
        area = w * h
        aspect = w / max(h, 1)
        motivo = ""

        if area < area_banda * AREA_MIN_CARACTER_REL:
            motivo = "area_pequena"
        elif area > area_banda * AREA_MAX_CARACTER_REL:
            motivo = "area_grande"
        elif h < alto_banda * ALTURA_MIN_CARACTER_REL and not (h > alto_banda * 0.28 and (x < ancho_banda * 0.12 or x + w > ancho_banda * 0.88)):
            motivo = "altura_pequena"
        elif h > alto_banda * ALTURA_MAX_CARACTER_REL:
            motivo = "altura_grande"
        elif w < ancho_banda * ANCHO_MIN_CARACTER_REL:
            motivo = "ancho_pequeno"
        elif w > ancho_banda * ANCHO_MAX_CARACTER_REL:
            motivo = "posible_caracter_unido"
        elif aspect < ASPECT_MIN_CARACTER:
            motivo = "aspecto_muy_delgado"
        elif aspect > ASPECT_MAX_CARACTER:
            motivo = "aspecto_muy_ancho"
        elif h < alto_banda * 0.45 and w < ancho_banda * 0.07:
            motivo = "posible_guion_o_ruido"

        item = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "ruta_caracter": "",
            "ancho": int(w),
            "alto": int(h),
            "area": int(area),
            "aceptado": not motivo,
            "motivo_rechazo": motivo,
        }

        if motivo:
            rechazados.append(item)
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(debug, motivo[:14], (x, max(10, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
        else:
            aceptados.append(item)

    aceptados = _eliminar_items_duplicados(aceptados)
    aceptados = _seleccionar_caracteres_principales(aceptados, ancho_banda, alto_banda)
    aceptados.sort(key=lambda item: item["bbox"][0])
    causa_segmentacion = _diagnosticar_segmentacion(aceptados, rechazados, ancho_banda, alto_banda)

    for idx, item in enumerate(aceptados, start=1):
        x, y, w, h = item["bbox"]
        margen_x = max(3, int(w * 0.16))
        margen_y = max(2, int(h * 0.08))
        x1 = max(x - margen_x, 0)
        y1 = max(y - margen_y, 0)
        x2 = min(x + w + margen_x, ancho_banda)
        y2 = min(y + h + margen_y, alto_banda)
        recorte = binaria[y1:y2, x1:x2]
        recorte = _limpiar_lineas_recorte_caracter(recorte)
        ruta_caracter = CARACTERES_DIR / f"{nombre}_v2_char_{idx:02d}.jpg"
        cv2.imwrite(str(ruta_caracter), recorte)
        item["ruta_caracter"] = str(ruta_caracter)
        item["bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        item["ancho"] = int(x2 - x1)
        item["alto"] = int(y2 - y1)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(debug, str(idx), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    ruta_debug = DEBUG_SEGMENTACION_DIR / f"{nombre}_debug_segmentacion_v2.jpg"
    cv2.imwrite(str(ruta_debug), debug)

    return {
        "caracteres": aceptados,
        "contornos_rechazados": rechazados,
        "cantidad_aceptados": len(aceptados),
        "cantidad_rechazados": len(rechazados),
        "motivos_rechazo": _contar_motivos_rechazo(rechazados),
        "ruta_banda": banda_info.get("ruta_banda"),
        "ruta_debug": str(ruta_debug),
        "bbox_banda": banda_info.get("bbox_banda"),
        "componentes_totales": len(bboxes),
        "diagnostico_segmentacion": causa_segmentacion,
        "causa_probable_segmentacion": causa_segmentacion.get("causa_probable"),
        "primer_caracter_posible_cortado": causa_segmentacion.get("primer_caracter_posible_cortado"),
        "exceso_componentes": causa_segmentacion.get("exceso_componentes"),
        "segmentacion_incompleta": len(aceptados) < CARACTERES_MIN_PLACA,
        "estado": "ok" if 6 <= len(aceptados) <= 7 else "advertencia",
        "mensaje": _mensaje_segmentacion_por_cantidad(len(aceptados)),
    }

def segmentar_caracteres_v3_estrategias(ruta_imagen_procesada: str, nombre_base: str = "placa") -> dict:
    COMPARACION_DIR = RECONOCIMIENTO_CARACTERES_DIR / "comparacion_segmentacion"
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_SEGMENTACION_DIR.mkdir(parents=True, exist_ok=True)
    BANDAS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARACION_DIR.mkdir(parents=True, exist_ok=True)
    imagen = _imread_seguro(str(ruta_imagen_procesada), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return {
            "caracteres": [],
            "contornos_rechazados": [],
            "cantidad_aceptados": 0,
            "cantidad_rechazados": 0,
            "ruta_banda": None,
            "ruta_debug": None,
            "estado": "error",
            "mensaje": "No se pudo cargar la imagen procesada para segmentacion por estrategias.",
        }

    nombre = _nombre_seguro(ruta_imagen_procesada, nombre_base)
    carpeta_comp = COMPARACION_DIR / nombre
    carpeta_comp.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(carpeta_comp / "placa_binarizada.png"), imagen)
    bandas = _generar_bandas_candidatas(imagen, nombre, carpeta_comp)
    estrategias = []
    # FIX-3: activar segmentacion guiada de forma temprana si los componentes ya evidencian fallo.
    for banda_info in bandas:
        banda = banda_info.get("banda")
        if banda is None:
            continue
        binaria_previa = _asegurar_caracteres_blancos(banda)
        binaria_previa = _limpiar_bordes_largos_banda(binaria_previa)
        analisis_guiada = _analizar_componentes_para_guiada_directa(binaria_previa)
        if analisis_guiada.get("activar_guiada_directa"):
            estrategia_guiada = _evaluar_segmentacion_guiada_formato_ecuador(banda_info, nombre, carpeta_comp)
            if len(estrategia_guiada.get("aceptados", [])) < 6 and not estrategia_guiada.get("slots_ecuador_aceptado", False):
                estrategias.append(estrategia_guiada)
                continue
            estrategia_guiada["motivo_guiada_directa"] = analisis_guiada.get("motivo_guiada_directa")
            estrategia_guiada["activacion_guiada_directa"] = True
            estrategia_guiada["componentes_validos_guiada_directa"] = analisis_guiada.get("componentes_validos")
            estrategia_guiada["cantidad_guiones_descartados"] = max(
                int(estrategia_guiada.get("cantidad_guiones_descartados", 0)),
                int(analisis_guiada.get("guiones_fisicos_descartados", 0)),
            )
            estrategia_guiada["guion_fisico_descartado"] = bool(estrategia_guiada.get("cantidad_guiones_descartados", 0))
            estrategia_guiada["filtros_componentes"] = _sumar_stats_filtros_componentes(
                estrategia_guiada.get("filtros_componentes", {}),
                analisis_guiada.get("filtros_componentes", {}),
            )
            estrategias_validas = [estrategia_guiada] if estrategia_guiada.get("banda") is not None else []
            if estrategias_validas:
                mejor = estrategia_guiada
                caracteres = _guardar_caracteres_estrategia(mejor, nombre)
                detalle_json = carpeta_comp / "detalle_estrategias.json"
                detalle_json.write_text(
                    json.dumps([_serializar_estrategia_segmentacion(mejor)], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return {
                    "caracteres": caracteres,
                    "contornos_rechazados": mejor.get("rechazados", []),
                    "cantidad_aceptados": len(caracteres),
                    "cantidad_rechazados": len(mejor.get("rechazados", [])),
                    "motivos_rechazo": _contar_motivos_rechazo(mejor.get("rechazados", [])),
                    "ruta_banda": mejor.get("ruta_banda"),
                    "ruta_debug": mejor.get("ruta_debug"),
                    "bbox_banda": mejor.get("bbox_banda"),
                    "componentes_totales": mejor.get("componentes_totales", 0),
                    "diagnostico_segmentacion": mejor.get("diagnostico_segmentacion", {}),
                    "causa_probable_segmentacion": (mejor.get("diagnostico_segmentacion") or {}).get("causa_probable"),
                    "primer_caracter_posible_cortado": (mejor.get("diagnostico_segmentacion") or {}).get("primer_caracter_posible_cortado"),
                    "exceso_componentes": (mejor.get("diagnostico_segmentacion") or {}).get("exceso_componentes"),
                    "segmentacion_incompleta": len(caracteres) < CARACTERES_MIN_PLACA,
                    "estrategia_segmentacion": mejor.get("nombre_estrategia"),
                    "puntaje_segmentacion": mejor.get("puntaje_segmentacion"),
                    "segmentacion_guiada_formato": bool(mejor.get("segmentacion_guiada_formato")),
                    "guion_descartado": bool(mejor.get("guion_descartado")),
                    "guion_fisico_descartado": bool(mejor.get("guion_fisico_descartado")),
                    "cantidad_guiones_descartados": int(mejor.get("cantidad_guiones_descartados", 0)),
                    "componentes_divididos_por_valle": int(mejor.get("componentes_divididos_por_valle", 0)),
                    "filtros_componentes": mejor.get("filtros_componentes", {}),
                    "fallback_slots_usado": bool(mejor.get("fallback_slots_usado")),
                    "debug_slots_ecuador_path": mejor.get("debug_slots_ecuador_path"),
                    "slots_utiles_generados": int(mejor.get("slots_utiles_generados", 0)),
                    "slots_dudosos": mejor.get("slots_dudosos", []),
                    "cantidad_slots_dudosos": int(mejor.get("cantidad_slots_dudosos", 0)),
                    "puntaje_slots_ecuador": mejor.get("puntaje_slots_ecuador"),
                    "motivo_rechazo_slots_ecuador": mejor.get("motivo_rechazo_slots_ecuador", ""),
                    "slots_ecuador_aceptado": bool(mejor.get("slots_ecuador_aceptado", False)),
                    "activacion_guiada_directa": True,
                    "motivo_guiada_directa": mejor.get("motivo_guiada_directa"),
                    "detalle_estrategias": str(detalle_json),
                    "estado": "ok" if CARACTERES_MIN_PLACA <= len(caracteres) <= CARACTERES_MAX_PLACA else "advertencia",
                    "mensaje": f"{_mensaje_segmentacion_por_cantidad(len(caracteres))} Estrategia: {mejor.get('nombre_estrategia')}.",
                }

    for banda_info in bandas:
        for metodo in ["componentes", "proyeccion", "hibrida"]:
            estrategias.append(_evaluar_estrategia_segmentacion(banda_info, metodo, nombre, carpeta_comp))

    estrategias_validas = [item for item in estrategias if item.get("banda") is not None]
    if not estrategias_validas:
        return segmentar_caracteres_v2(ruta_imagen_procesada, nombre_base)

    mejor_normal = max(estrategias_validas, key=lambda item: item.get("puntaje_segmentacion", -999.0))
    cantidad_normal = len(mejor_normal.get("aceptados", []))
    usar_guiada = cantidad_normal not in (6, 7) or float(mejor_normal.get("puntaje_segmentacion", -999.0)) < 62.0
    if usar_guiada:
        for banda_info in bandas:
            estrategias.append(_evaluar_segmentacion_guiada_formato_ecuador(banda_info, nombre, carpeta_comp))
        estrategias_validas = [item for item in estrategias if item.get("banda") is not None]

    mejor = max(estrategias_validas, key=lambda item: item.get("puntaje_segmentacion", -999.0))
    # FIX-D: fallback final por slots si todas las estrategias quedan con menos de 5 chars o puntaje bajo.
    if len(mejor.get("aceptados", [])) < 5 or float(mejor.get("puntaje_segmentacion", -999.0)) < 40.0:
        candidatos_slots = []
        for banda_info in bandas:
            candidatos_slots.append(_evaluar_fallback_division_uniforme(banda_info, nombre, carpeta_comp, 7))
            candidatos_slots.append(_evaluar_fallback_division_uniforme(banda_info, nombre, carpeta_comp, 6))
        candidatos_slots = [item for item in candidatos_slots if item.get("banda") is not None]
        if candidatos_slots:
            mejor_slots = max(candidatos_slots, key=lambda item: item.get("puntaje_segmentacion", -999.0))
            estrategias.extend(candidatos_slots)
            estrategias_validas.extend(candidatos_slots)
            if mejor_slots.get("puntaje_segmentacion", -999.0) >= mejor.get("puntaje_segmentacion", -999.0):
                mejor = mejor_slots
    caracteres = _guardar_caracteres_estrategia(mejor, nombre)
    debug_final = mejor.get("ruta_debug")
    ruta_banda = mejor.get("ruta_banda")
    detalle_json = carpeta_comp / "detalle_estrategias.json"
    detalle_json.write_text(
        json.dumps([_serializar_estrategia_segmentacion(item) for item in estrategias_validas], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "caracteres": caracteres,
        "contornos_rechazados": mejor.get("rechazados", []),
        "cantidad_aceptados": len(caracteres),
        "cantidad_rechazados": len(mejor.get("rechazados", [])),
        "motivos_rechazo": _contar_motivos_rechazo(mejor.get("rechazados", [])),
        "ruta_banda": ruta_banda,
        "ruta_debug": debug_final,
        "bbox_banda": mejor.get("bbox_banda"),
        "componentes_totales": mejor.get("componentes_totales", 0),
        "diagnostico_segmentacion": mejor.get("diagnostico_segmentacion", {}),
        "causa_probable_segmentacion": (mejor.get("diagnostico_segmentacion") or {}).get("causa_probable"),
        "primer_caracter_posible_cortado": (mejor.get("diagnostico_segmentacion") or {}).get("primer_caracter_posible_cortado"),
        "exceso_componentes": (mejor.get("diagnostico_segmentacion") or {}).get("exceso_componentes"),
        "segmentacion_incompleta": len(caracteres) < CARACTERES_MIN_PLACA,
        "estrategia_segmentacion": mejor.get("nombre_estrategia"),
        "puntaje_segmentacion": mejor.get("puntaje_segmentacion"),
        "segmentacion_guiada_formato": bool(mejor.get("segmentacion_guiada_formato")),
        "guion_descartado": bool(mejor.get("guion_descartado")),
        "guion_fisico_descartado": bool(mejor.get("guion_fisico_descartado")),
        "cantidad_guiones_descartados": int(mejor.get("cantidad_guiones_descartados", 0)),
        "componentes_divididos_por_valle": int(mejor.get("componentes_divididos_por_valle", 0)),
        "filtros_componentes": mejor.get("filtros_componentes", {}),
        "fallback_slots_usado": bool(mejor.get("fallback_slots_usado")),
        "debug_slots_ecuador_path": mejor.get("debug_slots_ecuador_path"),
        "slots_utiles_generados": int(mejor.get("slots_utiles_generados", 0)),
        "slots_dudosos": mejor.get("slots_dudosos", []),
        "cantidad_slots_dudosos": int(mejor.get("cantidad_slots_dudosos", 0)),
        "puntaje_slots_ecuador": mejor.get("puntaje_slots_ecuador"),
        "motivo_rechazo_slots_ecuador": mejor.get("motivo_rechazo_slots_ecuador", ""),
        "slots_ecuador_aceptado": bool(mejor.get("slots_ecuador_aceptado", False)),
        "activacion_guiada_directa": bool(mejor.get("activacion_guiada_directa")),
        "motivo_guiada_directa": mejor.get("motivo_guiada_directa", ""),
        "detalle_estrategias": str(detalle_json),
        "estado": "ok" if CARACTERES_MIN_PLACA <= len(caracteres) <= CARACTERES_MAX_PLACA else "advertencia",
        "mensaje": f"{_mensaje_segmentacion_por_cantidad(len(caracteres))} Estrategia: {mejor.get('nombre_estrategia')}.",
    }

def _generar_bandas_candidatas(imagen, nombre: str, carpeta_comp: Path) -> list[dict]:
    imagen = _asegurar_caracteres_blancos(imagen)
    alto, ancho = imagen.shape[:2]
    bandas = []
    dinamica = extraer_banda_caracteres(imagen, nombre)
    if dinamica.get("banda") is not None:
        dinamica["nombre_banda"] = "dinamica"
        bandas.append(dinamica)
    # FIX-A: ninguna banda candidata debe incluir el tercio superior con texto ECUADOR.
    rangos = [
        ("media", 0.35, 0.88),
        ("inferior", 0.42, 0.96),
        ("amplia", 0.35, 0.98),
        ("central_baja", 0.36, 0.94),
    ]
    for etiqueta, y_ini, y_fin in rangos:
        y1 = int(alto * y_ini)
        y2 = int(alto * y_fin)
        banda = imagen[y1:y2, :].copy()
        margen_x = max(int(ancho * 0.04), 6)
        banda = cv2.copyMakeBorder(banda, 2, 2, margen_x, margen_x, cv2.BORDER_CONSTANT, value=0)
        ruta = BANDAS_DIR / f"{nombre}_banda_{etiqueta}.jpg"
        cv2.imwrite(str(ruta), banda)
        cv2.imwrite(str(carpeta_comp / f"banda_{etiqueta}.png"), banda)
        bandas.append(
            {
                "nombre_banda": etiqueta,
                "ruta_banda": str(ruta),
                "banda": banda,
                "bbox_banda": [0, y1, ancho, y2],
                "estado": "ok",
            }
        )
    return bandas

def _evaluar_estrategia_segmentacion(banda_info: dict, metodo: str, nombre: str, carpeta_comp: Path) -> dict:
    banda = banda_info.get("banda")
    if banda is None:
        return {"banda": None, "puntaje_segmentacion": -999.0}
    alto_banda, ancho_banda = banda.shape[:2]
    binaria = _asegurar_caracteres_blancos(banda)
    binaria = _limpiar_bordes_largos_banda(binaria)
    binaria = _eliminar_marco_placa(binaria)  # FIX-MARCO
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    componentes_divididos_por_valle = 0
    cantidad_guiones_descartados = 0
    stats_filtros_componentes = {}
    if metodo == "componentes":
        bboxes = _bboxes_componentes_conectados(binaria)
        bboxes, stats_filtros_componentes = _filtrar_componentes_marco_borde(bboxes, ancho_banda, alto_banda)
    elif metodo == "proyeccion":
        bboxes = _bboxes_por_proyeccion_vertical(binaria)
        bboxes, stats_filtros_componentes = _filtrar_componentes_marco_borde(bboxes, ancho_banda, alto_banda)
    else:
        bboxes_componentes = _bboxes_componentes_conectados(binaria)
        bboxes_componentes, stats_componentes = _filtrar_componentes_marco_borde(bboxes_componentes, ancho_banda, alto_banda)
        bboxes_proyeccion = _bboxes_por_proyeccion_vertical(binaria)
        bboxes_proyeccion, stats_proyeccion = _filtrar_componentes_marco_borde(bboxes_proyeccion, ancho_banda, alto_banda)
        bboxes = _fusionar_bboxes(bboxes_componentes + bboxes_proyeccion)
        bboxes, stats_fusion = _filtrar_componentes_marco_borde(bboxes, ancho_banda, alto_banda)
        stats_filtros_componentes = _sumar_stats_filtros_componentes(stats_componentes, stats_proyeccion, stats_fusion)
        bboxes = _dividir_bboxes_demasiado_anchas(binaria, bboxes)
        bboxes, componentes_divididos_por_valle = _dividir_componentes_anchos_por_valle(binaria, bboxes)
        bboxes, stats_post_division = _filtrar_componentes_marco_borde(bboxes, ancho_banda, alto_banda)
        stats_filtros_componentes = _sumar_stats_filtros_componentes(stats_filtros_componentes, stats_post_division)
    bboxes, cantidad_guiones_descartados = _filtrar_guion_fisico_bboxes(bboxes, alto_banda)
    cantidad_guiones_descartados += int(stats_filtros_componentes.get("componentes_guion_descartados", 0))
    aceptados, rechazados = _filtrar_bboxes_caracteres(bboxes, ancho_banda, alto_banda)
    aceptados = _eliminar_items_duplicados(aceptados)
    aceptados = _seleccionar_caracteres_principales(aceptados, ancho_banda, alto_banda)
    aceptados.sort(key=lambda item: item["bbox"][0])
    diagnostico = _diagnosticar_segmentacion(aceptados, rechazados, ancho_banda, alto_banda)
    puntaje = calcular_puntaje_segmentacion(aceptados, banda, {"rechazados": rechazados, "diagnostico": diagnostico})
    nombre_estrategia = f"{banda_info.get('nombre_banda', 'banda')}_{metodo}"
    debug = _dibujar_debug_estrategia(banda, aceptados, rechazados, carpeta_comp / f"debug_{nombre_estrategia}.jpg")
    return {
        "nombre_estrategia": nombre_estrategia,
        "banda": banda,
        "binaria": binaria,
        "bbox_banda": banda_info.get("bbox_banda"),
        "ruta_banda": banda_info.get("ruta_banda"),
        "ruta_debug": debug,
        "aceptados": aceptados,
        "rechazados": rechazados,
        "componentes_totales": len(bboxes),
        "diagnostico_segmentacion": diagnostico,
        "puntaje_segmentacion": puntaje,
        "guion_fisico_descartado": bool(cantidad_guiones_descartados),
        "cantidad_guiones_descartados": int(cantidad_guiones_descartados),
        "componentes_divididos_por_valle": int(componentes_divididos_por_valle),
        "filtros_componentes": stats_filtros_componentes,
    }

def _evaluar_segmentacion_guiada_formato_ecuador(banda_info: dict, nombre: str, carpeta_comp: Path) -> dict:
    banda = banda_info.get("banda")
    if banda is None:
        return {"banda": None, "puntaje_segmentacion": -999.0}
    alto_banda, ancho_banda = banda.shape[:2]
    binaria = _asegurar_caracteres_blancos(banda)
    binaria = _limpiar_bordes_largos_banda(binaria)
    binaria = _eliminar_marco_placa(binaria)  # FIX-MARCO
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    bboxes_componentes = _bboxes_componentes_conectados(binaria)
    bboxes_componentes, stats_componentes = _filtrar_componentes_marco_borde(bboxes_componentes, ancho_banda, alto_banda)
    bboxes_proyeccion = _bboxes_por_proyeccion_vertical(binaria)
    bboxes_proyeccion, stats_proyeccion = _filtrar_componentes_marco_borde(bboxes_proyeccion, ancho_banda, alto_banda)
    bboxes_originales = _fusionar_bboxes(bboxes_componentes + bboxes_proyeccion)
    bboxes_base, stats_fusion = _filtrar_componentes_marco_borde(bboxes_originales, ancho_banda, alto_banda)
    bboxes_base = _dividir_bboxes_demasiado_anchas(binaria, bboxes_base)
    bboxes_base, componentes_divididos_por_valle = _dividir_componentes_anchos_por_valle(binaria, bboxes_base)
    bboxes_base, stats_post_division = _filtrar_componentes_marco_borde(bboxes_base, ancho_banda, alto_banda)
    bboxes_base, cantidad_guiones_descartados = _filtrar_guion_fisico_bboxes(bboxes_base, alto_banda)
    stats_filtros_componentes = _sumar_stats_filtros_componentes(stats_componentes, stats_proyeccion, stats_fusion, stats_post_division)
    cantidad_guiones_descartados += int(stats_filtros_componentes.get("componentes_guion_descartados", 0))
    bboxes_base = _descartar_guion_y_ruido_guiado(bboxes_base, ancho_banda, alto_banda)
    aceptados_base, rechazados = _filtrar_bboxes_caracteres(bboxes_base, ancho_banda, alto_banda)
    aceptados_base = _seleccionar_caracteres_principales(_eliminar_items_duplicados(aceptados_base), ancho_banda, alto_banda)
    aceptados_base.sort(key=lambda item: item["bbox"][0])

    candidatos = []
    if len(aceptados_base) in (6, 7):
        candidatos.append(("guiada_componentes", aceptados_base, "componentes_filtrados", {}))
    for cantidad_objetivo in (7, 6):
        slots_eval = _evaluar_slots_ecuador(binaria, cantidad_objetivo, nombre, carpeta_comp, banda_info.get("nombre_banda", "banda"))
        candidatos.append((f"guiada_slots_{cantidad_objetivo}", slots_eval["aceptados"], "slots_ecuador", slots_eval))

    mejor_nombre = "segmentacion_guiada_formato_ecuador"
    mejor_aceptados = []
    mejor_puntaje = -999.0
    mejor_modo = ""
    mejor_extra = {}
    for nombre_candidato, aceptados, modo, extra in candidatos:
        puntaje = calcular_puntaje_segmentacion(aceptados, binaria, {"rechazados": rechazados})
        if len(aceptados) in (6, 7):
            puntaje += 12.0
        if modo == "slots_ecuador":
            puntaje = 0.45 * puntaje + 0.55 * float(extra.get("puntaje_slots_ecuador", 0.0))
            if not extra.get("slots_ecuador_aceptado", False) or len(aceptados) < 6:
                puntaje -= 35.0
                if len(aceptados) < 6:
                    puntaje = min(puntaje, -100.0)
        if puntaje > mejor_puntaje:
            mejor_nombre = nombre_candidato
            mejor_aceptados = aceptados
            mejor_puntaje = puntaje
            mejor_modo = modo
            mejor_extra = extra

    diagnostico = _diagnosticar_segmentacion(mejor_aceptados, rechazados, ancho_banda, alto_banda)
    diagnostico["segmentacion_guiada_formato"] = True
    diagnostico["modo_guiado"] = mejor_modo
    diagnostico["guion_descartado"] = len(bboxes_base) < len(bboxes_originales)
    diagnostico["guion_fisico_descartado"] = bool(cantidad_guiones_descartados)
    diagnostico["cantidad_guiones_descartados"] = int(cantidad_guiones_descartados)
    diagnostico["filtros_componentes"] = stats_filtros_componentes
    diagnostico["slots_ecuador"] = (
        {
            "debug_slots_ecuador_path": mejor_extra.get("debug_slots_ecuador_path"),
            "slots_utiles_generados": int(mejor_extra.get("slots_utiles_generados", 0)),
            "cantidad_slots_dudosos": int(mejor_extra.get("cantidad_slots_dudosos", 0)),
            "slots_dudosos": mejor_extra.get("slots_dudosos", []),
            "puntaje_slots_ecuador": mejor_extra.get("puntaje_slots_ecuador"),
            "motivo_rechazo_slots_ecuador": mejor_extra.get("motivo_rechazo_slots_ecuador", ""),
            "slots_ecuador_aceptado": bool(mejor_extra.get("slots_ecuador_aceptado", False)),
        }
        if mejor_modo == "slots_ecuador"
        else {}
    )
    debug = _dibujar_debug_estrategia(
        banda,
        mejor_aceptados,
        rechazados,
        carpeta_comp / f"debug_{banda_info.get('nombre_banda', 'banda')}_{mejor_nombre}.jpg",
    )
    return {
        "nombre_estrategia": f"{banda_info.get('nombre_banda', 'banda')}_{mejor_nombre}",
        "banda": banda,
        "binaria": binaria,
        "bbox_banda": banda_info.get("bbox_banda"),
        "ruta_banda": banda_info.get("ruta_banda"),
        "ruta_debug": debug,
        "aceptados": mejor_aceptados,
        "rechazados": rechazados,
        "componentes_totales": len(bboxes_base),
        "diagnostico_segmentacion": diagnostico,
        "puntaje_segmentacion": round(float(mejor_puntaje), 4),
        "segmentacion_guiada_formato": True,
        "guion_descartado": diagnostico["guion_descartado"],
        "guion_fisico_descartado": bool(cantidad_guiones_descartados),
        "cantidad_guiones_descartados": int(cantidad_guiones_descartados),
        "componentes_divididos_por_valle": int(componentes_divididos_por_valle),
        "filtros_componentes": stats_filtros_componentes,
        "debug_slots_ecuador_path": mejor_extra.get("debug_slots_ecuador_path"),
        "slots_utiles_generados": int(mejor_extra.get("slots_utiles_generados", 0)),
        "slots_dudosos": mejor_extra.get("slots_dudosos", []),
        "cantidad_slots_dudosos": int(mejor_extra.get("cantidad_slots_dudosos", 0)),
        "puntaje_slots_ecuador": mejor_extra.get("puntaje_slots_ecuador"),
        "motivo_rechazo_slots_ecuador": mejor_extra.get("motivo_rechazo_slots_ecuador", ""),
        "slots_ecuador_aceptado": bool(mejor_extra.get("slots_ecuador_aceptado", False)),
    }

def _descartar_guion_y_ruido_guiado(bboxes: list[tuple[int, int, int, int]], ancho_banda: int, alto_banda: int) -> list[tuple[int, int, int, int]]:
    salida = []
    for x, y, w, h in bboxes:
        aspect = w / max(h, 1)
        centro_x = (x + w / 2) / max(ancho_banda, 1)
        es_guion = h < alto_banda * 0.30 and w < ancho_banda * 0.14 and 0.30 <= centro_x <= 0.62
        es_borde = h > alto_banda * 0.82 and w < ancho_banda * 0.025
        es_ruido = w * h < ancho_banda * alto_banda * 0.002 or aspect < 0.05
        if es_guion or es_borde or es_ruido:
            continue
        salida.append((x, y, w, h))
    return salida

def _es_guion_o_ruido_separador(bbox: tuple[int, int, int, int], altura_banda: int) -> bool:
    # FIX-1: filtro critico para guiones fisicos y separadores antes de aceptar caracteres.
    x, y, w, h = bbox
    if h <= 0:
        return True
    aspect = w / max(h, 1)
    altura_rel = h / max(altura_banda, 1)
    es_guion = aspect > 2.5 and altura_rel < 0.30
    es_ruido = (w * h) < 80
    return bool(es_guion or es_ruido)

def _filtrar_guion_fisico_bboxes(bboxes: list[tuple[int, int, int, int]], altura_banda: int) -> tuple[list[tuple[int, int, int, int]], int]:
    # FIX-1: aplicar el descarte fisico antes de filtros de area/aspecto.
    filtrados = []
    descartados = 0
    for bbox in bboxes:
        if _es_guion_o_ruido_separador(bbox, altura_banda):
            descartados += 1
            continue
        filtrados.append(bbox)
    return filtrados, descartados

def _filtrar_componentes_marco_borde(
    bboxes: list[tuple[int, int, int, int]],
    ancho_banda: int,
    alto_banda: int,
) -> tuple[list[tuple[int, int, int, int]], dict]:
    # FIX-C: descartar marco, borde del recorte, fondo enorme, guion fisico y ruido antes de segmentar.
    filtrados = []
    stats = {
        "componentes_borde_descartados": 0,
        "componentes_guion_descartados": 0,
        "componentes_grandes_descartados": 0,
        "componentes_ruido_descartados": 0,
    }
    area_banda = ancho_banda * alto_banda
    margen = 3
    area_minima = max(60, area_banda * 0.001)
    for x, y, w, h in bboxes:
        toca_borde = x <= margen or y <= margen or x + w >= ancho_banda - margen or y + h >= alto_banda - margen
        if toca_borde and (w > ancho_banda * 0.5 or h > alto_banda * 0.7):
            stats["componentes_borde_descartados"] += 1
            continue
        aspect = w / max(h, 1)
        altura_rel = h / max(alto_banda, 1)
        if aspect > 2.5 and altura_rel < 0.30:
            stats["componentes_guion_descartados"] += 1
            continue
        area_comp = w * h
        if area_comp > area_banda * 0.55 or w > ancho_banda * 0.80:
            stats["componentes_grandes_descartados"] += 1
            continue
        if area_comp < area_minima:
            stats["componentes_ruido_descartados"] += 1
            continue
        filtrados.append((x, y, w, h))
    return filtrados, stats

def _sumar_stats_filtros_componentes(*stats_items: dict) -> dict:
    # FIX-C: unificar contadores de filtros para exponer diagnostico sin cambiar retornos existentes.
    acumulado = {
        "componentes_borde_descartados": 0,
        "componentes_guion_descartados": 0,
        "componentes_grandes_descartados": 0,
        "componentes_ruido_descartados": 0,
    }
    for item in stats_items:
        for clave in acumulado:
            acumulado[clave] += int((item or {}).get(clave, 0))
    return acumulado

def _analizar_componentes_para_guiada_directa(binaria) -> dict:
    # FIX-3: detectar temprano casos donde la segmentacion normal suele fallar.
    alto_banda, ancho_banda = binaria.shape[:2]
    bboxes = _bboxes_componentes_conectados(binaria)
    bboxes, stats_componentes = _filtrar_componentes_marco_borde(bboxes, ancho_banda, alto_banda)
    bboxes, descartados = _filtrar_guion_fisico_bboxes(bboxes, alto_banda)
    componentes_validos = [bbox for bbox in bboxes if bbox[3] >= alto_banda * 0.22]
    motivo = ""
    activar = False
    if len(componentes_validos) < 5:
        activar = True
        motivo = "menos_de_5_componentes_validos"
    elif len(componentes_validos) >= 3:
        anchos = np.array([bbox[2] for bbox in componentes_validos], dtype=np.float32)
        alturas = np.array([bbox[3] for bbox in componentes_validos], dtype=np.float32)
        if float(np.max(anchos)) > 1.7 * max(float(np.mean(anchos)), 1.0):
            activar = True
            motivo = "componente_ancho_posible_caracter_unido"
        elif float(np.std(alturas)) / max(float(np.mean(alturas)), 1.0) > 0.40:
            activar = True
            motivo = "alturas_componentes_inestables"
    return {
        "activar_guiada_directa": bool(activar),
        "motivo_guiada_directa": motivo,
        "componentes_validos": len(componentes_validos),
        "guiones_fisicos_descartados": int(descartados + stats_componentes.get("componentes_guion_descartados", 0)),
        "filtros_componentes": stats_componentes,
    }

def _limpiar_banda_para_slots_ecuador(banda_binaria) -> dict:
    # FIX-SLOTS-ECUADOR: limpiar bordes y ruido antes de dividir la banda en slots.
    binaria = _asegurar_caracteres_blancos(banda_binaria)
    alto, ancho = binaria.shape[:2]
    limpia = binaria.copy()
    filas_borde_eliminadas = 0
    columnas_borde_eliminadas = 0

    proyeccion_filas = np.sum(limpia > 0, axis=1)
    for y, valor in enumerate(proyeccion_filas):
        if valor > ancho * 0.55 and (y < alto * 0.20 or y > alto * 0.80):
            limpia[max(0, y - 1) : min(alto, y + 2), :] = 0
            filas_borde_eliminadas += 1

    proyeccion_columnas = np.sum(limpia > 0, axis=0)
    for x, valor in enumerate(proyeccion_columnas):
        if valor > alto * 0.80 and (x < ancho * 0.10 or x > ancho * 0.90):
            limpia[:, max(0, x - 1) : min(ancho, x + 2)] = 0
            columnas_borde_eliminadas += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    limpia = cv2.morphologyEx(limpia, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(limpia, 8)
    depurada = np.zeros_like(limpia)
    area_min = max(8, int(alto * ancho * 0.0008))
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        aspect = w / max(h, 1)
        es_linea_h = aspect > 4.5 and h < alto * 0.16
        es_linea_v = aspect < 0.12 and h > alto * 0.60
        if area >= area_min and not es_linea_h and not es_linea_v:
            depurada[labels == label] = 255
    if cv2.countNonZero(depurada) > 0:
        limpia = depurada

    return {
        "banda_slots_original": binaria,
        "banda_slots_limpia": limpia,
        "filas_borde_eliminadas": int(filas_borde_eliminadas),
        "columnas_borde_eliminadas": int(columnas_borde_eliminadas),
    }

def _calcular_banda_util_slots_ecuador(banda_limpia) -> dict:
    # FIX-SLOTS-ECUADOR: recalcular zona vertical util para evitar marco superior/inferior.
    alto, ancho = banda_limpia.shape[:2]
    componentes = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(banda_limpia, 8)
    area_min = max(10, int(alto * ancho * 0.001))
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        aspect = w / max(h, 1)
        if area < area_min:
            continue
        # FIX-TOP: subir el techo de altura para no descartar caracteres altos.
        if h < alto * 0.22 or h > alto * 0.98:
            continue
        if aspect > 4.0 or aspect < 0.06:
            continue
        componentes.append((x, y, w, h))

    if len(componentes) >= 3:
        y_min = min(y for _, y, _, _ in componentes)
        y_max = max(y + h for _, y, _, h in componentes)
        metodo = "componentes_plausibles"
    else:
        proyeccion = np.sum(banda_limpia > 0, axis=1)
        umbral = max(1, int(ancho * 0.05))
        filas = np.where(proyeccion > umbral)[0]
        if filas.size:
            y_min = int(filas[0])
            y_max = int(filas[-1]) + 1
            metodo = "proyeccion_horizontal"
        else:
            y_min = int(alto * 0.12)
            y_max = int(alto * 0.88)
            metodo = "fallback_central"

    # FIX-TOP: padding superior mayor para conservar trazos altos del caracter.
    padding_sup = max(3, int(alto * 0.08))
    padding_inf = max(3, int(alto * 0.04))
    y1 = max(0, int(y_min) - padding_sup)
    y2 = min(alto, int(y_max) + padding_inf)
    if y2 - y1 < max(8, int(alto * 0.25)):
        centro = (y1 + y2) // 2
        mitad = max(5, int(alto * 0.22))
        y1 = max(0, centro - mitad)
        y2 = min(alto, centro + mitad)
        metodo = f"{metodo}_expandido"
    return {
        "banda_util_slots": banda_limpia[y1:y2, :].copy(),
        "y_min_chars_slots": int(y1),
        "y_max_chars_slots": int(y2),
        "metodo_y_util_slots": metodo,
    }

def _extraer_caracter_util_desde_slot(slot_binario) -> dict:
    # FIX-SLOTS-ECUADOR: extraer solo tinta util del caracter dentro del slot.
    slot = _asegurar_caracteres_blancos(slot_binario)
    alto, ancho = slot.shape[:2]
    limpio = slot.copy()

    # FIX-TOP: quitar solo lineas finas de marco; el scoring posterior ya descarta
    # componentes que sean lineas. Antes se borraba tinta densa del 25% superior y
    # eso decapitaba caracteres con trazo superior (E, F, T, Z, 5, 7).
    limpio = _quitar_lineas_finas_marco(limpio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(limpio, 8)
    candidatos = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(3, int(alto * ancho * 0.004)):
            continue
        aspect = w / max(h, 1)
        es_linea_h = aspect > 3.2 and h < alto * 0.28
        es_linea_v = aspect < 0.12 and h > alto * 0.55
        if es_linea_h or es_linea_v:
            continue
        centro_x = (x + w / 2) / max(ancho, 1)
        score_altura = min(h / max(alto * 0.65, 1), 1.2)
        score_area = min(area / max(alto * ancho * 0.16, 1), 1.0)
        score_centro = max(0.0, 1.0 - abs(centro_x - 0.5) * 1.5)
        score_aspect = 1.0 if 0.10 <= aspect <= 1.35 else 0.45
        candidatos.append((score_altura * 0.35 + score_area * 0.35 + score_centro * 0.20 + score_aspect * 0.10, x, y, w, h, area))

    slot_dudoso = False
    motivo = ""
    if candidatos:
        _, x, y, w, h, _ = max(candidatos, key=lambda item: item[0])
        margen_x = max(1, int(w * 0.08))
        margen_y = max(1, int(h * 0.05))
        x1 = max(0, x - margen_x)
        y1 = max(0, y - margen_y)
        x2 = min(ancho, x + w + margen_x)
        y2 = min(alto, y + h + margen_y)
        caracter = limpio[y1:y2, x1:x2].copy()
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    else:
        coords = cv2.findNonZero(limpio)
        if coords is None:
            slot_dudoso = True
            motivo = "sin_tinta_util"
            caracter = limpio
            bbox = [0, 0, 0, 0]
        else:
            x, y, w, h = cv2.boundingRect(coords)
            caracter = limpio[y : y + h, x : x + w].copy()
            bbox = [int(x), int(y), int(w), int(h)]

    bw, bh = bbox[2], bbox[3]
    if not slot_dudoso:
        if bh < alto * 0.35:
            slot_dudoso = True
            motivo = "altura_util_baja"
        elif bw < ancho * 0.10:
            slot_dudoso = True
            motivo = "ancho_util_bajo"
        elif bh > 0 and bw / max(bh, 1) > 3.2:
            slot_dudoso = True
            motivo = "linea_horizontal"
        elif bw > 0 and bh / max(bw, 1) > 6.0:
            slot_dudoso = True
            motivo = "linea_vertical"

    return {
        "slot_original": slot,
        "slot_limpio": limpio,
        "slot_caracter_util": caracter,
        "bbox_util": bbox,
        "slot_dudoso": bool(slot_dudoso),
        "motivo_slot_dudoso": motivo,
    }

def _bboxes_slots_formato(binaria, cantidad: int) -> list[tuple[int, int, int, int]]:
    alto, ancho = binaria.shape[:2]
    cols = np.sum(binaria > 0, axis=0)
    activas = np.where(cols > max(1, int(alto * 0.04)))[0]
    if activas.size:
        x_ini = max(int(activas[0]) - int(ancho * 0.02), 0)
        x_fin = min(int(activas[-1]) + int(ancho * 0.02), ancho - 1)
    else:
        x_ini, x_fin = int(ancho * 0.04), int(ancho * 0.96)
    ancho_util = max(x_fin - x_ini + 1, cantidad)
    slot_w = ancho_util / cantidad
    bboxes = []
    for idx in range(cantidad):
        sx1 = int(round(x_ini + idx * slot_w))
        sx2 = int(round(x_ini + (idx + 1) * slot_w))
        sx1 = max(0, min(sx1, ancho - 1))
        sx2 = max(sx1 + 1, min(sx2, ancho))
        roi = binaria[:, sx1:sx2]
        coords = cv2.findNonZero(roi)
        if coords is None:
            y1, h = int(alto * 0.12), int(alto * 0.76)
            x_local, w = 0, sx2 - sx1
        else:
            x_local, y1, w, h = cv2.boundingRect(coords)
        margen_x = max(1, int((sx2 - sx1) * 0.08))
        x = max(sx1 + x_local - margen_x, 0)
        y = max(y1 - 2, 0)
        x2 = min(sx1 + x_local + w + margen_x, ancho)
        y2 = min(y1 + h + 2, alto)
        bboxes.append((x, y, max(1, x2 - x), max(1, y2 - y)))
    return bboxes

def _fallback_division_uniforme(banda_bin, n_slots: int = 7) -> list[tuple[int, int, int, int]]:
    # FIX-D: dividir la banda en slots y ajustar cortes a valles reales de proyeccion.
    H, W = banda_bin.shape[:2]
    ancho_slot = max(1, W // max(n_slots, 1))
    proyeccion = np.sum(banda_bin > 0, axis=0).astype(float)
    kernel_suav = max(3, ancho_slot // 4)
    proyeccion = np.convolve(proyeccion, np.ones(kernel_suav) / kernel_suav, mode="same")

    cortes = []
    for i in range(1, n_slots):
        centro_esperado = i * ancho_slot
        margen_busqueda = max(1, ancho_slot // 3)
        inicio = max(0, centro_esperado - margen_busqueda)
        fin = min(W, centro_esperado + margen_busqueda)
        if fin > inicio:
            valle_local = inicio + int(np.argmin(proyeccion[inicio:fin]))
            cortes.append(int(valle_local))
        else:
            cortes.append(int(centro_esperado))

    puntos = [0] + cortes + [W]
    bboxes = []
    for i in range(len(puntos) - 1):
        x_ini = int(puntos[i])
        x_fin = int(puntos[i + 1])
        if x_fin - x_ini <= 5:
            continue
        columna = banda_bin[:, x_ini:x_fin]
        filas_con_tinta = np.where(np.sum(columna > 0, axis=1) > 0)[0]
        if len(filas_con_tinta) > 0:
            y1 = max(0, int(filas_con_tinta[0]) - 2)
            y2 = min(H, int(filas_con_tinta[-1]) + 2)
        else:
            y1, y2 = 0, H
        bboxes.append((x_ini, y1, x_fin - x_ini, max(1, y2 - y1)))
    return bboxes

def _evaluar_fallback_division_uniforme(banda_info: dict, nombre: str, carpeta_comp: Path, n_slots: int) -> dict:
    # FIX-D: evaluar fallback por slots como estrategia de ultimo recurso.
    banda = banda_info.get("banda")
    if banda is None:
        return {"banda": None, "puntaje_segmentacion": -999.0}
    binaria = _asegurar_caracteres_blancos(banda)
    binaria = _limpiar_bordes_largos_banda(binaria)
    binaria = _eliminar_marco_placa(binaria)  # FIX-MARCO
    slots_eval = _evaluar_slots_ecuador(binaria, n_slots, nombre, carpeta_comp, f"{banda_info.get('nombre_banda', 'banda')}_fallback")
    aceptados = slots_eval["aceptados"]
    rechazados = []
    puntaje = calcular_puntaje_segmentacion(aceptados, binaria, {"rechazados": rechazados})
    if len(aceptados) in (6, 7):
        puntaje += 8.0
    puntaje = 0.45 * puntaje + 0.55 * float(slots_eval.get("puntaje_slots_ecuador", 0.0))
    if not slots_eval.get("slots_ecuador_aceptado", False) or len(aceptados) < 6:
        puntaje -= 35.0
        if len(aceptados) < 6:
            puntaje = min(puntaje, -100.0)
    nombre_estrategia = f"{banda_info.get('nombre_banda', 'banda')}_fallback_uniforme_{n_slots}"
    diagnostico = _diagnosticar_segmentacion(aceptados, rechazados, binaria.shape[1], binaria.shape[0])
    diagnostico["fallback_slots_usado"] = True
    diagnostico["n_slots"] = int(n_slots)
    diagnostico["slots_ecuador"] = {
        "debug_slots_ecuador_path": slots_eval.get("debug_slots_ecuador_path"),
        "slots_utiles_generados": int(slots_eval.get("slots_utiles_generados", 0)),
        "cantidad_slots_dudosos": int(slots_eval.get("cantidad_slots_dudosos", 0)),
        "slots_dudosos": slots_eval.get("slots_dudosos", []),
        "puntaje_slots_ecuador": slots_eval.get("puntaje_slots_ecuador"),
        "motivo_rechazo_slots_ecuador": slots_eval.get("motivo_rechazo_slots_ecuador", ""),
        "slots_ecuador_aceptado": bool(slots_eval.get("slots_ecuador_aceptado", False)),
    }
    debug = _dibujar_debug_estrategia(banda, aceptados, rechazados, carpeta_comp / f"debug_{nombre_estrategia}.jpg")
    return {
        "nombre_estrategia": nombre_estrategia,
        "banda": banda,
        "binaria": binaria,
        "bbox_banda": banda_info.get("bbox_banda"),
        "ruta_banda": banda_info.get("ruta_banda"),
        "ruta_debug": debug,
        "aceptados": aceptados,
        "rechazados": rechazados,
        "componentes_totales": len(aceptados),
        "diagnostico_segmentacion": diagnostico,
        "puntaje_segmentacion": round(float(puntaje), 4),
        "fallback_slots_usado": True,
        "n_slots_fallback": int(n_slots),
        "debug_slots_ecuador_path": slots_eval.get("debug_slots_ecuador_path"),
        "slots_utiles_generados": int(slots_eval.get("slots_utiles_generados", 0)),
        "slots_dudosos": slots_eval.get("slots_dudosos", []),
        "cantidad_slots_dudosos": int(slots_eval.get("cantidad_slots_dudosos", 0)),
        "puntaje_slots_ecuador": slots_eval.get("puntaje_slots_ecuador"),
        "motivo_rechazo_slots_ecuador": slots_eval.get("motivo_rechazo_slots_ecuador", ""),
        "slots_ecuador_aceptado": bool(slots_eval.get("slots_ecuador_aceptado", False)),
    }

def _items_desde_bboxes_slots(bboxes: list[tuple[int, int, int, int]], binaria, origen: str) -> list[dict]:
    alto, ancho = binaria.shape[:2]
    items = []
    for idx, (x, y, w, h) in enumerate(bboxes, start=1):
        area = int(w * h)
        items.append(
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "ruta_caracter": "",
                "ancho": int(w),
                "alto": int(h),
                "area": area,
                "aceptado": True,
                "motivo_rechazo": "",
                "origen": origen,
                "posicion_formato": idx,
                "tipo_esperado": _tipo_posicion_placa(idx - 1),
                "slot_relativo": [round(x / max(ancho, 1), 4), round(y / max(alto, 1), 4), round(w / max(ancho, 1), 4), round(h / max(alto, 1), 4)],
            }
        )
    return items

def _guardar_debug_slots_ecuador(nombre: str, etiqueta: str, datos: dict, carpeta_comp: Path | None = None) -> str:
    # FIX-SLOTS-ECUADOR: guardar debug visual de banda, slots y caracteres utiles enviados a CNN.
    carpeta = DEBUG_SLOTS_ECUADOR_DIR / _nombre_seguro(f"{nombre}_{etiqueta}", "slots")
    carpeta.mkdir(parents=True, exist_ok=True)
    imagenes = {
        "banda_slots_original.png": datos.get("banda_slots_original"),
        "banda_slots_limpia.png": datos.get("banda_slots_limpia"),
        "banda_util_slots.png": datos.get("banda_util_slots"),
    }
    for archivo, imagen in imagenes.items():
        if imagen is not None:
            cv2.imwrite(str(carpeta / archivo), imagen)

    banda_util = datos.get("banda_util_slots")
    slots = datos.get("slots_debug", [])
    if banda_util is not None:
        debug_slots = asegurar_bgr(banda_util)
        for idx, slot_info in enumerate(slots, start=1):
            x, y, w, h = slot_info.get("bbox_slot", [0, 0, 0, 0])
            color = (0, 0, 255) if slot_info.get("slot_dudoso") else (0, 180, 0)
            cv2.rectangle(debug_slots, (x, y), (x + w, y + h), color, 1)
            cv2.putText(debug_slots, str(idx), (x + 1, max(10, y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.imwrite(str(carpeta / "slots_sobre_banda_util.png"), debug_slots)

    for idx, slot_info in enumerate(slots, start=1):
        for clave, sufijo in [
            ("slot_original", "slot_original"),
            ("slot_limpio", "slot_limpio"),
            ("slot_caracter_util", "caracter_util"),
        ]:
            imagen = slot_info.get(clave)
            if imagen is not None:
                cv2.imwrite(str(carpeta / f"slot_{idx:02d}_{sufijo}.png"), imagen)
        caracter = slot_info.get("slot_caracter_util")
        if caracter is not None and np.asarray(caracter).size > 0:
            normalizado = normalizar_imagen_caracter_para_cnn(caracter, f"{nombre}_{etiqueta}_slot_{idx:02d}_debug")
            ruta_norm = normalizado.get("ruta_debug_normalizada")
            if ruta_norm and Path(ruta_norm).exists():
                img_norm = _imread_seguro(str(ruta_norm), cv2.IMREAD_GRAYSCALE)
                if img_norm is not None:
                    cv2.imwrite(str(carpeta / f"slot_{idx:02d}_normalizado_32x32.png"), img_norm)

    resumen = {
        "filas_borde_eliminadas": datos.get("filas_borde_eliminadas", 0),
        "columnas_borde_eliminadas": datos.get("columnas_borde_eliminadas", 0),
        "y_min_chars_slots": datos.get("y_min_chars_slots"),
        "y_max_chars_slots": datos.get("y_max_chars_slots"),
        "metodo_y_util_slots": datos.get("metodo_y_util_slots"),
        "slots_utiles_generados": datos.get("slots_utiles_generados", 0),
        "cantidad_slots_dudosos": datos.get("cantidad_slots_dudosos", 0),
        "slots_dudosos": datos.get("slots_dudosos", []),
        "puntaje_slots_ecuador": datos.get("puntaje_slots_ecuador"),
        "motivo_rechazo_slots_ecuador": datos.get("motivo_rechazo_slots_ecuador"),
        "slots_ecuador_aceptado": datos.get("slots_ecuador_aceptado"),
    }
    (carpeta / "debug_slots_ecuador.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    if carpeta_comp is not None:
        (carpeta_comp / f"debug_slots_ecuador_{etiqueta}.txt").write_text(str(carpeta), encoding="utf-8")
    return str(carpeta)

def _evaluar_slots_ecuador(binaria, cantidad: int, nombre: str, carpeta_comp: Path, etiqueta: str) -> dict:
    # FIX-SLOTS-ECUADOR: generar slots sobre banda limpia y recortar solo tinta util por slot.
    limpieza = _limpiar_banda_para_slots_ecuador(binaria)
    banda_limpia = limpieza["banda_slots_limpia"]
    util = _calcular_banda_util_slots_ecuador(banda_limpia)
    banda_util = util["banda_util_slots"]
    alto_util, ancho_util = banda_util.shape[:2]
    slot_w = ancho_util / max(cantidad, 1)
    aceptados = []
    slots_debug = []
    slots_dudosos = []
    alturas_utiles = []
    tinta_util = []
    for idx in range(cantidad):
        x1 = int(round(idx * slot_w))
        x2 = int(round((idx + 1) * slot_w))
        x1 = max(0, min(x1, ancho_util - 1))
        x2 = max(x1 + 1, min(x2, ancho_util))
        slot = banda_util[:, x1:x2]
        extraido = _extraer_caracter_util_desde_slot(slot)
        bx, by, bw, bh = extraido["bbox_util"]
        bbox_global = [int(x1 + bx), int(by), int(bw), int(bh)]
        slot_info = {
            **extraido,
            "indice": idx + 1,
            "bbox_slot": [int(x1), 0, int(x2 - x1), int(alto_util)],
            "bbox_util_global": bbox_global,
        }
        slots_debug.append(slot_info)
        if extraido["slot_dudoso"]:
            slots_dudosos.append({"indice": idx + 1, "motivo": extraido["motivo_slot_dudoso"]})
        if bw > 0 and bh > 0 and not extraido["slot_dudoso"]:
            area = int(bw * bh)
            alturas_utiles.append(bh)
            tinta_util.append(cv2.countNonZero(extraido["slot_caracter_util"]) / max(area, 1))
            aceptados.append(
                {
                    "bbox": bbox_global,
                    "ruta_caracter": "",
                    "ancho": int(bw),
                    "alto": int(bh),
                    "area": area,
                    "aceptado": True,
                    "motivo_rechazo": "",
                    "origen": f"slots_ecuador_{cantidad}",
                    "posicion_formato": idx + 1,
                    "tipo_esperado": _tipo_posicion_placa(idx),
                    "slot_dudoso": False,
                    "motivo_slot_dudoso": "",
                    "bbox_slot": slot_info["bbox_slot"],
                }
            )

    cantidad_dudosos = len(slots_dudosos)
    motivo_rechazo = ""
    aceptado = True
    if cantidad_dudosos > 2:
        aceptado = False
        motivo_rechazo = "mas_de_2_slots_dudosos"
    elif len(aceptados) < 6:
        aceptado = False
        motivo_rechazo = "pocos_slots_con_tinta_util"

    puntaje = 25.0
    if cantidad == 7:
        puntaje += 20.0
    puntaje += min(len(aceptados), cantidad) * 5.0
    puntaje -= cantidad_dudosos * 12.0
    puntaje -= int(limpieza["filas_borde_eliminadas"]) * 0.8
    if alturas_utiles:
        alturas = np.array(alturas_utiles, dtype=np.float32)
        puntaje += max(0.0, 18.0 - float(np.std(alturas)) / max(float(np.mean(alturas)), 1.0) * 30.0)
        if float(np.mean(alturas)) < alto_util * 0.35:
            puntaje -= 20.0
            if not motivo_rechazo:
                motivo_rechazo = "altura_media_util_baja"
    if tinta_util and float(np.mean(tinta_util)) < 0.08:
        puntaje -= 18.0
        if not motivo_rechazo:
            motivo_rechazo = "poca_tinta_real"
    if not aceptado:
        puntaje -= 25.0

    datos_debug = {
        **limpieza,
        **util,
        "slots_debug": slots_debug,
        "slots_utiles_generados": len(aceptados),
        "slots_dudosos": slots_dudosos,
        "cantidad_slots_dudosos": cantidad_dudosos,
        "puntaje_slots_ecuador": round(float(puntaje), 4),
        "motivo_rechazo_slots_ecuador": motivo_rechazo,
        "slots_ecuador_aceptado": bool(aceptado),
    }
    debug_path = _guardar_debug_slots_ecuador(nombre, f"{etiqueta}_{cantidad}", datos_debug, carpeta_comp)
    return {
        "aceptados": aceptados,
        "banda_util": banda_util,
        "debug_slots_ecuador_path": debug_path,
        "slots_utiles_generados": len(aceptados),
        "slots_dudosos": slots_dudosos,
        "cantidad_slots_dudosos": cantidad_dudosos,
        "puntaje_slots_ecuador": round(float(puntaje), 4),
        "motivo_rechazo_slots_ecuador": motivo_rechazo,
        "slots_ecuador_aceptado": bool(aceptado),
        "filas_borde_eliminadas": int(limpieza["filas_borde_eliminadas"]),
        "columnas_borde_eliminadas": int(limpieza["columnas_borde_eliminadas"]),
        "y_min_chars_slots": int(util["y_min_chars_slots"]),
        "y_max_chars_slots": int(util["y_max_chars_slots"]),
        "metodo_y_util_slots": util["metodo_y_util_slots"],
    }

def _filtrar_bboxes_caracteres(bboxes: list[tuple[int, int, int, int]], ancho_banda: int, alto_banda: int) -> tuple[list[dict], list[dict]]:
    aceptados = []
    rechazados = []
    area_banda = ancho_banda * alto_banda
    for x, y, w, h in bboxes:
        area = w * h
        aspect = w / max(h, 1)
        motivo = ""
        if area < area_banda * AREA_MIN_CARACTER_REL:
            motivo = "area_pequena"
        elif area > area_banda * AREA_MAX_CARACTER_REL:
            motivo = "area_grande"
        elif h < alto_banda * 0.28:
            motivo = "altura_pequena"
        # FIX-ALTURA-RUIDO: descartar bbox excesivamente alto y delgado causado por borde inferior.
        elif h > alto_banda * 0.75 and (h / max(w, 1)) > 3.2:
            motivo = "caracter_estirado_posible_ruido_inferior"
        elif h > alto_banda * ALTURA_MAX_CARACTER_REL:
            motivo = "altura_grande"
        elif w < ancho_banda * ANCHO_MIN_CARACTER_REL:
            motivo = "ancho_pequeno"
        elif w > ancho_banda * 0.32:
            motivo = "posible_caracter_unido"
        elif aspect < 0.08:
            motivo = "aspecto_muy_delgado"
        elif aspect > 1.45:
            motivo = "aspecto_muy_ancho"
        elif h < alto_banda * 0.45 and w < ancho_banda * 0.075:
            motivo = "posible_guion_o_ruido"
        item = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "ruta_caracter": "",
            "ancho": int(w),
            "alto": int(h),
            "area": int(area),
            "aceptado": not motivo,
            "motivo_rechazo": motivo,
        }
        (rechazados if motivo else aceptados).append(item)
    return aceptados, rechazados

def calcular_puntaje_segmentacion(caracteres: list[dict], banda, debug_info: dict | None = None) -> float:
    debug_info = debug_info or {}
    cantidad = len(caracteres)
    if not caracteres:
        return -100.0
    alto_banda, ancho_banda = banda.shape[:2]
    alturas = np.array([item["alto"] for item in caracteres], dtype=np.float32)
    centros_y = np.array([item["bbox"][1] + item["bbox"][3] / 2 for item in caracteres], dtype=np.float32)
    anchos = np.array([item["ancho"] for item in caracteres], dtype=np.float32)
    score_cantidad = 45.0 if cantidad in (6, 7) else max(0.0, 30.0 - abs(6.5 - cantidad) * 10.0)
    score_altura = max(0.0, 20.0 - float(np.std(alturas)) / max(float(np.mean(alturas)), 1.0) * 30.0)
    score_alineacion = max(0.0, 15.0 - float(np.std(centros_y)) / max(alto_banda, 1) * 80.0)
    score_tamano = max(0.0, 10.0 - abs(float(np.mean(alturas)) - alto_banda * 0.62) / max(alto_banda, 1) * 25.0)
    ruido = len(debug_info.get("rechazados") or [])
    penalizacion_ruido = min(12.0, ruido * 0.8)
    penalizacion_guion = sum(1 for item in caracteres if item["alto"] < alto_banda * 0.45 and item["ancho"] < ancho_banda * 0.08) * 8.0
    penalizacion_ancho = sum(1 for valor in anchos if valor > ancho_banda * 0.24) * 6.0
    return round(score_cantidad + score_altura + score_alineacion + score_tamano - penalizacion_ruido - penalizacion_guion - penalizacion_ancho, 4)

def _dibujar_debug_estrategia(banda, aceptados: list[dict], rechazados: list[dict], ruta: Path) -> str:
    debug = asegurar_bgr(banda)
    for item in rechazados:
        x, y, w, h = item["bbox"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 1)
    for idx, item in enumerate(aceptados, start=1):
        x, y, w, h = item["bbox"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 180, 0), 2)
        cv2.putText(debug, str(idx), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)
    cv2.imwrite(str(ruta), debug)
    return str(ruta)

def _guardar_caracteres_estrategia(estrategia: dict, nombre: str) -> list[dict]:
    caracteres = []
    binaria = estrategia["binaria"]
    alto_banda, ancho_banda = binaria.shape[:2]
    for idx, item in enumerate(estrategia.get("aceptados", []), start=1):
        x, y, w, h = item["bbox"]
        # FIX-MARGEN-INF: reducir margen inferior para no capturar ruido del borde.
        margen_x = max(2, int(w * 0.06))
        margen_y_sup = max(2, int(h * 0.05))
        margen_y_inf = max(1, int(h * 0.02))
        x1 = max(x - margen_x, 0)
        y1 = max(y - margen_y_sup, 0)
        x2 = min(x + w + margen_x, ancho_banda)
        limite_inf = int(alto_banda * 0.88)
        margen_inferior_reducido = y + h >= limite_inf
        if margen_inferior_reducido:
            y2 = min(y + h, alto_banda)
        else:
            y2 = min(y + h + margen_y_inf, alto_banda)
        # FIX-GUION-SLOT: verificar que el slot contiene un caracter real y no un guion o linea horizontal.
        recorte_check = binaria[y1:y2, x1:x2]
        if recorte_check.size > 0:
            alto_check, ancho_check = recorte_check.shape[:2]
            proyeccion_check = np.sum(recorte_check > 0, axis=1)
            filas_activas = np.sum(proyeccion_check > ancho_check * 0.05)
            umbral_filas_slot = 0.15 if str(item.get("origen", "")).startswith("slots_ecuador") else 0.25
            if filas_activas < alto_check * umbral_filas_slot:
                continue
        recorte = _limpiar_lineas_recorte_caracter(binaria[y1:y2, x1:x2])
        ruta_caracter = CARACTERES_DIR / f"{nombre}_v3_char_{idx:02d}.jpg"
        cv2.imwrite(str(ruta_caracter), recorte)
        nuevo = item.copy()
        nuevo["ruta_caracter"] = str(ruta_caracter)
        nuevo["bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        nuevo["ancho"] = int(x2 - x1)
        nuevo["alto"] = int(y2 - y1)
        nuevo["margen_inferior_reducido"] = bool(margen_inferior_reducido)
        caracteres.append(nuevo)
    return caracteres

def _binarizar_banda_robusta(gris: np.ndarray) -> np.ndarray:
    """Binariza una banda en escala de grises evitando el desborde del marco/fondo.

    El umbral global (Otsu) inunda de blanco el marco metalico y el fondo cuando hay
    iluminacion no uniforme, fusionando los caracteres en un solo bloque. Aqui se
    generan candidatos (Otsu + adaptativa con varias C, en ambas polaridades) y se
    elige el que maximiza el puntaje de orientacion de caracteres.
    """
    gris = asegurar_grayscale(gris)
    g = cv2.GaussianBlur(gris, (3, 3), 0)
    candidatos = []
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidatos += [otsu, cv2.bitwise_not(otsu)]
    block = max(11, (min(g.shape[:2]) // 2) | 1)
    for c in (7, 12):
        adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)
        candidatos += [adapt, cv2.bitwise_not(adapt)]
    return max(candidatos, key=_score_orientacion_caracteres)

def _aislar_bloque_caracteres(binaria: np.ndarray) -> np.ndarray:
    """Quita restos de marco metalico y deja solo el bloque de caracteres.

    Heuristica robusta basada en que (1) el marco toca los bordes y suele ser muy
    ancho/bajo o una barra fina, y (2) los caracteres comparten altura y linea base
    similares, mientras los restos de marco son atipicos en altura o quedan
    descentrados verticalmente.
    """
    if binaria is None or binaria.size == 0:
        return binaria
    h, w = binaria.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    candidatos = []
    for l in range(1, n):
        x = int(stats[l, cv2.CC_STAT_LEFT])
        y = int(stats[l, cv2.CC_STAT_TOP])
        cw = int(stats[l, cv2.CC_STAT_WIDTH])
        ch = int(stats[l, cv2.CC_STAT_HEIGHT])
        toca_izq = x <= 1
        toca_der = (x + cw) >= w - 1
        if cw > 0.30 * w or ch < 0.35 * h:
            continue
        if (toca_izq or toca_der) and cw < 0.04 * w:
            continue
        candidatos.append((l, x, y, cw, ch, y + ch / 2.0))
    if len(candidatos) < 3:
        return binaria
    alturas = sorted(c[4] for c in candidatos)
    h_med = alturas[len(alturas) // 2]
    centros = sorted(c[5] for c in candidatos)
    cy_med = centros[len(centros) // 2]
    keep = np.zeros_like(binaria)
    conservados = 0
    for l, x, y, cw, ch, cy in candidatos:
        if ch < 0.62 * h_med or ch > 1.6 * h_med:
            continue
        if abs(cy - cy_med) > 0.24 * h:
            continue
        keep[labels == l] = 255
        conservados += 1
    return keep if conservados > 0 else binaria

def _cajas_caracteres_bloque(binaria: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Devuelve cajas de caracteres por componentes conectados, dividiendo cajas muy
    anchas (caracteres pegados) y recortando restos de marco en los extremos hasta
    dejar 6-7 (formato de placa Ecuador)."""
    if binaria is None or binaria.size == 0:
        return []
    h, w = binaria.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    cajas = []
    for l in range(1, n):
        x = int(stats[l, cv2.CC_STAT_LEFT])
        y = int(stats[l, cv2.CC_STAT_TOP])
        cw = int(stats[l, cv2.CC_STAT_WIDTH])
        ch = int(stats[l, cv2.CC_STAT_HEIGHT])
        if cw <= 0 or ch <= 0:
            continue
        cajas.append((x, y, cw, ch))
    if not cajas:
        return []
    cajas.sort()
    anchos = sorted(c[2] for c in cajas)
    w_med = anchos[len(anchos) // 2]
    divididas = []
    for x, y, cw, ch in cajas:
        if cw > 1.55 * w_med and cw > 0.10 * w:
            k = max(2, min(int(round(cw / max(w_med, 1))), 3))
            sw = cw // k
            for j in range(k):
                divididas.append((x + j * sw, y, sw, ch))
        else:
            divididas.append((x, y, cw, ch))
    cajas = sorted(divididas)
    while len(cajas) > 7:
        centros = [x + cw / 2.0 for x, _, cw, _ in cajas]
        gaps = [centros[i + 1] - centros[i] for i in range(len(centros) - 1)]
        pitch = sorted(gaps)[len(gaps) // 2]
        if gaps[0] >= gaps[-1] and gaps[0] > 1.4 * pitch:
            cajas = cajas[1:]
        elif gaps[-1] > 1.4 * pitch:
            cajas = cajas[:-1]
        else:
            alturas = [ch for _, _, _, ch in cajas]
            h_med = sorted(alturas)[len(alturas) // 2]
            if abs(cajas[0][3] - h_med) >= abs(cajas[-1][3] - h_med):
                cajas = cajas[1:]
            else:
                cajas = cajas[:-1]
    return [(int(x), int(y), int(cw), int(ch)) for x, y, cw, ch in cajas]

def _recortar_y_guardar_camino_limpio(estrategia: dict, nombre: str) -> list[dict]:
    """Recorta cada caracter de la banda binaria con un margen simple y lo guarda.

    Recorte deliberadamente sencillo (sin limpiezas agresivas) porque la banda ya
    viene limpia tras el aislamiento del bloque; las limpiezas extra deterioran la
    lectura de la CNN.
    """
    binaria = estrategia["binaria"]
    alto_b, ancho_b = binaria.shape[:2]
    caracteres = []
    for idx, item in enumerate(estrategia.get("aceptados", []), start=1):
        x, y, w, h = item["bbox"]
        m = 2
        x1 = max(x - m, 0)
        y1 = max(y - m, 0)
        x2 = min(x + w + m, ancho_b)
        y2 = min(y + h + m, alto_b)
        recorte = binaria[y1:y2, x1:x2]
        if recorte.size == 0:
            continue
        ruta_caracter = CARACTERES_DIR / f"{nombre}_limpio_char_{idx:02d}.jpg"
        cv2.imwrite(str(ruta_caracter), recorte)
        nuevo = item.copy()
        nuevo["ruta_caracter"] = str(ruta_caracter)
        nuevo["bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        nuevo["ancho"] = int(x2 - x1)
        nuevo["alto"] = int(y2 - y1)
        caracteres.append(nuevo)
    return caracteres

def _banda_binaria_desbordada(binaria: np.ndarray) -> bool:
    """Detecta si la binarizacion se desbordo: el primer plano cubre demasiado o un
    unico componente abarca casi toda la banda (marco+fondo fusionados con letras)."""
    if binaria is None or binaria.size == 0:
        return True
    h, w = binaria.shape[:2]
    ratio = cv2.countNonZero(binaria) / float(h * w)
    if ratio > 0.55:
        return True
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    for l in range(1, n):
        cw = int(stats[l, cv2.CC_STAT_WIDTH])
        ch = int(stats[l, cv2.CC_STAT_HEIGHT])
        if cw > 0.7 * w and ch > 0.7 * h:
            return True
    return False

def segmentar_caracteres_camino_limpio(ruta_procesada: str, ruta_gris: str | None, nombre_base: str = "placa") -> dict:
    """Camino de segmentacion dedicado y robusto contra contaminacion de marco.

    Fuente principal: banda binaria del preprocesamiento v3 (la CNN la lee bien).
    Si esa banda se desborda (marco/fondo fusionados con las letras), reintenta con
    binarizacion adaptativa propia desde el gris. Aisla el bloque de caracteres y
    segmenta por componentes. Compite con v3 a nivel de orquestacion.
    """
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    BANDAS_DIR.mkdir(parents=True, exist_ok=True)
    proc = _imread_seguro(str(ruta_procesada), cv2.IMREAD_GRAYSCALE)
    if proc is None:
        return {
            "caracteres": [],
            "cantidad_aceptados": 0,
            "estado": "error",
            "mensaje": "Camino limpio: no se pudo cargar la imagen procesada.",
            "puntaje_segmentacion": -999.0,
            "metodo": "camino_limpio",
        }
    gris = _imread_seguro(str(ruta_gris), cv2.IMREAD_GRAYSCALE) if ruta_gris else None
    nombre = _nombre_seguro(ruta_procesada, nombre_base)
    alto, ancho = proc.shape[:2]
    margen_x = max(int(ancho * 0.04), 6)
    rangos = [("media", 0.35, 0.88), ("inferior", 0.42, 0.96), ("amplia", 0.35, 0.98), ("central_baja", 0.36, 0.94)]
    mejor = None
    for etiqueta, y_ini, y_fin in rangos:
        y1 = int(alto * y_ini)
        y2 = int(alto * y_fin)
        if y2 - y1 <= 1:
            continue
        fuentes = [("v3bin", _asegurar_caracteres_blancos(proc[y1:y2, :]))]
        if gris is not None:
            fuentes.append(("robusta", _binarizar_banda_robusta(gris[y1:y2, :])))
        elif _banda_binaria_desbordada(fuentes[0][1]):
            pass
        for tag, binaria0 in fuentes:
            binaria = cv2.copyMakeBorder(binaria0, 2, 2, margen_x, margen_x, cv2.BORDER_CONSTANT, value=0)
            binaria = _aislar_bloque_caracteres(binaria)
            cajas = _cajas_caracteres_bloque(binaria)
            if not cajas:
                continue
            aceptados = [
                {
                    "bbox": [x, y, cw, ch],
                    "ruta_caracter": "",
                    "ancho": cw,
                    "alto": ch,
                    "area": cw * ch,
                    "aceptado": True,
                    "motivo_rechazo": "",
                    "origen": f"camino_limpio_{tag}",
                }
                for (x, y, cw, ch) in cajas
            ]
            puntaje = calcular_puntaje_segmentacion(aceptados, binaria, {"rechazados": []})
            if len(aceptados) in (6, 7):
                puntaje += 15.0
            candidato = {
                "nombre_estrategia": f"camino_limpio_{etiqueta}_{tag}",
                "banda": binaria,
                "binaria": binaria,
                "aceptados": aceptados,
                "rechazados": [],
                "puntaje_segmentacion": puntaje,
                "ruta_banda": None,
                "bbox_banda": [0, y1, ancho, y2],
            }
            if mejor is None or puntaje > mejor["puntaje_segmentacion"]:
                mejor = candidato
    if mejor is None:
        return {
            "caracteres": [],
            "cantidad_aceptados": 0,
            "estado": "advertencia",
            "mensaje": "Camino limpio: no se aislaron caracteres.",
            "puntaje_segmentacion": -999.0,
            "metodo": "camino_limpio",
        }
    ruta_banda = BANDAS_DIR / f"{nombre}_banda_limpia.jpg"
    cv2.imwrite(str(ruta_banda), mejor["binaria"])
    mejor["ruta_banda"] = str(ruta_banda)
    caracteres = _recortar_y_guardar_camino_limpio(mejor, nombre)
    return {
        "caracteres": caracteres,
        "contornos_rechazados": [],
        "cantidad_aceptados": len(caracteres),
        "cantidad_rechazados": 0,
        "motivos_rechazo": {},
        "ruta_banda": mejor["ruta_banda"],
        "ruta_debug": None,
        "bbox_banda": mejor.get("bbox_banda"),
        "estrategia_segmentacion": mejor["nombre_estrategia"],
        "puntaje_segmentacion": float(mejor["puntaje_segmentacion"]),
        "estado": "ok" if CARACTERES_MIN_PLACA <= len(caracteres) <= CARACTERES_MAX_PLACA else "advertencia",
        "mensaje": f"Camino limpio: {len(caracteres)} caracteres. Estrategia {mejor['nombre_estrategia']}.",
        "metodo": "camino_limpio",
    }

def _elegir_segmentacion(seg_actual: dict, caracteres_actual: list, seg_limpio: dict) -> tuple[dict, list]:
    """Elige entre la segmentacion v3 y el camino limpio.

    Prioriza la cantidad de caracteres dentro del formato Ecuador (6-7). Si ambas o
    ninguna cumplen, decide por mayor puntaje de segmentacion.
    """
    caracteres_limpio = seg_limpio.get("caracteres", []) or []
    n_actual = len(caracteres_actual)
    n_limpio = len(caracteres_limpio)
    p_actual = float(seg_actual.get("puntaje_segmentacion", -999.0) or -999.0)
    p_limpio = float(seg_limpio.get("puntaje_segmentacion", -999.0) or -999.0)
    actual_ok = CARACTERES_MIN_PLACA <= n_actual <= CARACTERES_MAX_PLACA
    limpio_ok = CARACTERES_MIN_PLACA <= n_limpio <= CARACTERES_MAX_PLACA
    if n_limpio == 0:
        return seg_actual, caracteres_actual
    if limpio_ok and not actual_ok:
        elegir_limpio = True
    elif actual_ok and not limpio_ok:
        elegir_limpio = False
    else:
        elegir_limpio = p_limpio > p_actual
    if elegir_limpio:
        seg_limpio["metodo"] = "camino_limpio"
        return seg_limpio, caracteres_limpio
    return seg_actual, caracteres_actual

def _serializar_estrategia_segmentacion(item: dict) -> dict:
    return {
        "nombre_estrategia": item.get("nombre_estrategia"),
        "puntaje_segmentacion": item.get("puntaje_segmentacion"),
        "cantidad": len(item.get("aceptados", [])),
        "rechazados": len(item.get("rechazados", [])),
        "ruta_banda": item.get("ruta_banda"),
        "ruta_debug": item.get("ruta_debug"),
        "diagnostico_segmentacion": item.get("diagnostico_segmentacion", {}),
        "segmentacion_guiada_formato": bool(item.get("segmentacion_guiada_formato")),
        "guion_descartado": bool(item.get("guion_descartado")),
        "guion_fisico_descartado": bool(item.get("guion_fisico_descartado")),
        "cantidad_guiones_descartados": int(item.get("cantidad_guiones_descartados", 0)),
        "componentes_divididos_por_valle": int(item.get("componentes_divididos_por_valle", 0)),
        "filtros_componentes": item.get("filtros_componentes", {}),
        "fallback_slots_usado": bool(item.get("fallback_slots_usado")),
        "n_slots_fallback": item.get("n_slots_fallback"),
        "debug_slots_ecuador_path": item.get("debug_slots_ecuador_path"),
        "slots_utiles_generados": int(item.get("slots_utiles_generados", 0)),
        "slots_dudosos": item.get("slots_dudosos", []),
        "cantidad_slots_dudosos": int(item.get("cantidad_slots_dudosos", 0)),
        "puntaje_slots_ecuador": item.get("puntaje_slots_ecuador"),
        "motivo_rechazo_slots_ecuador": item.get("motivo_rechazo_slots_ecuador", ""),
        "slots_ecuador_aceptado": bool(item.get("slots_ecuador_aceptado", False)),
        "activacion_guiada_directa": bool(item.get("activacion_guiada_directa")),
        "motivo_guiada_directa": item.get("motivo_guiada_directa", ""),
    }

def _seleccionar_caracteres_principales(items: list[dict], ancho_banda: int, alto_banda: int) -> list[dict]:
    centro_y_objetivo = alto_banda * 0.52
    alto_objetivo = alto_banda * 0.70
    puntuados = []
    for item in items:
        x, y, w, h = item["bbox"]
        area = max(int(item.get("area", w * h)), 1)
        centro_y = y + h / 2
        score_altura = max(0.0, 1.0 - abs(h - alto_objetivo) / max(alto_objetivo, 1))
        score_y = max(0.0, 1.0 - abs(centro_y - centro_y_objetivo) / max(alto_banda * 0.5, 1))
        score_area = min(area / max(ancho_banda * alto_banda * 0.08, 1), 1.0)
        score_borde = 0.95 if x <= ancho_banda * 0.04 or x + w >= ancho_banda * 0.96 else 1.0
        score_guion = 0.35 if h < alto_banda * 0.45 and w < ancho_banda * 0.08 else 1.0
        item["_score_caracter"] = (0.45 * score_altura + 0.30 * score_y + 0.20 * score_area + 0.05 * score_borde) * score_guion
        puntuados.append(item)

    if len(puntuados) <= CARACTERES_MAX_PLACA:
        return sorted(puntuados, key=lambda item: item["bbox"][0])

    mejores_7 = sorted(puntuados, key=lambda item: item["_score_caracter"], reverse=True)[:7]
    mejores_7.sort(key=lambda item: item["bbox"][0])
    return mejores_7

def _limpiar_bordes_largos_banda(binaria) -> np.ndarray:
    limpia = binaria.copy()
    alto, ancho = limpia.shape[:2]
    filas = np.sum(limpia > 0, axis=1)
    columnas = np.sum(limpia > 0, axis=0)

    for y, valor in enumerate(filas):
        # FIX-UMBRAL: bajar umbrales para eliminar marco metalico.
        if valor > ancho * 0.45 and (y < alto * 0.32 or y > alto * 0.72):
            y1 = max(0, y - 1)
            y2 = min(alto, y + 2)
            limpia[y1:y2, :] = 0

    for x, valor in enumerate(columnas):
        if valor > alto * 0.50 and (x < ancho * 0.10 or x > ancho * 0.90):
            x1 = max(0, x - 1)
            x2 = min(ancho, x + 2)
            limpia[:, x1:x2] = 0

    return limpia

def _eliminar_marco_placa(binaria: np.ndarray) -> np.ndarray:
    """
    # FIX-MARCO: eliminar el marco rectangular metalico de la placa ecuatoriana antes de segmentar caracteres.
    # El marco aparece como rectangulo blanco en los bordes de la banda binarizada.
    """
    if binaria is None or binaria.size == 0:
        return binaria
    alto, ancho = binaria.shape[:2]
    resultado = binaria.copy()

    # FIX-MARCO: quitar primero el marco rectangular hueco (lazo cerrado que toca
    # varios bordes). Antes solo se quitaban barras finas H o V, por lo que un marco
    # completo sobrevivia, se pegaba a los caracteres y rompia la segmentacion.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(resultado, 8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        toca_izq = x <= 2
        toca_der = (x + w) >= ancho - 2
        toca_sup = y <= 2
        toca_inf = (y + h) >= alto - 2
        bordes_tocados = int(toca_izq) + int(toca_der) + int(toca_sup) + int(toca_inf)
        cubre_bbox = w >= ancho * 0.82 and h >= alto * 0.60
        densidad = area / max(w * h, 1)
        if cubre_bbox and bordes_tocados >= 3 and densidad < 0.45:
            resultado[labels == label] = 0

    margen_h = max(2, int(alto * 0.06))
    margen_w = max(4, int(ancho * 0.03))
    resultado[:margen_h, :] = 0
    resultado[alto - margen_h :, :] = 0
    resultado[:, :margen_w] = 0
    resultado[:, ancho - margen_w :] = 0

    proyeccion_h = np.sum(resultado > 0, axis=1)
    for y in range(alto):
        if proyeccion_h[y] > ancho * 0.55:
            y1 = max(0, y - 1)
            y2 = min(alto, y + 2)
            resultado[y1:y2, :] = 0

    proyeccion_v = np.sum(resultado > 0, axis=0)
    for x in range(ancho):
        if proyeccion_v[x] > alto * 0.55:
            x1 = max(0, x - 1)
            x2 = min(ancho, x + 2)
            resultado[:, x1:x2] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(resultado, 8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        toca_izq = x <= margen_w + 2
        toca_der = (x + w) >= ancho - margen_w - 2
        toca_sup = y <= margen_h + 2
        toca_inf = (y + h) >= alto - margen_h - 2
        es_barra_h = toca_izq and toca_der and h < alto * 0.22
        es_barra_v = toca_sup and toca_inf and w < ancho * 0.10
        if es_barra_h or es_barra_v:
            resultado[labels == label] = 0

    return resultado

def _limpiar_lineas_recorte_caracter(recorte) -> np.ndarray:
    if recorte is None or recorte.size == 0:
        return recorte
    limpio = recorte.copy()
    alto, ancho = limpio.shape[:2]
    filas = np.sum(limpio > 0, axis=1)
    for y, valor in enumerate(filas):
        if valor > ancho * 0.72 and (y < alto * 0.18 or y > alto * 0.82):
            limpio[max(0, y - 1) : min(alto, y + 2), :] = 0
    return limpio

def _bboxes_componentes_conectados(binaria) -> list[tuple[int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    bboxes = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        bboxes.append((x, y, w, h))
    return bboxes

def _bboxes_por_proyeccion_vertical(binaria) -> list[tuple[int, int, int, int]]:
    alto, ancho = binaria.shape[:2]
    columnas = np.sum(binaria > 0, axis=0)
    umbral = max(1, int(alto * 0.08))
    activa = columnas >= umbral
    grupos = []
    inicio = None
    for idx, valor in enumerate(activa):
        if valor and inicio is None:
            inicio = idx
        elif not valor and inicio is not None:
            if idx - inicio >= max(2, int(ancho * 0.008)):
                grupos.append((inicio, idx - 1))
            inicio = None
    if inicio is not None and ancho - inicio >= max(2, int(ancho * 0.008)):
        grupos.append((inicio, ancho - 1))

    bboxes = []
    for x1, x2 in grupos:
        roi = binaria[:, x1 : x2 + 1]
        coords = cv2.findNonZero(roi)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        bboxes.append((x1 + x, y, w, h))
    return bboxes

def _fusionar_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []
    ordenados = sorted(bboxes, key=lambda bbox: bbox[0])
    fusionados = []
    for bbox in ordenados:
        x, y, w, h = bbox
        if not fusionados:
            fusionados.append(bbox)
            continue
        px, py, pw, ph = fusionados[-1]
        separacion = x - (px + pw)
        solape_y = min(y + h, py + ph) - max(y, py)
        if _iou_bbox(bbox, fusionados[-1]) > 0.20 or (separacion <= 2 and solape_y > min(h, ph) * 0.45):
            nx1 = min(px, x)
            ny1 = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            fusionados[-1] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
        else:
            fusionados.append(bbox)
    return fusionados

def _dividir_bboxes_demasiado_anchas(binaria, bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    alto, ancho = binaria.shape[:2]
    salida = []
    ancho_max = ancho * ANCHO_MAX_CARACTER_REL
    for x, y, w, h in bboxes:
        if w <= ancho_max or h < alto * 0.35:
            salida.append((x, y, w, h))
            continue

        roi = binaria[y : y + h, x : x + w]
        columnas = np.sum(roi > 0, axis=0)
        umbral = max(1, int(h * 0.08))
        activa = columnas >= umbral
        grupos = []
        inicio = None
        for idx, valor in enumerate(activa):
            if valor and inicio is None:
                inicio = idx
            elif not valor and inicio is not None:
                if idx - inicio >= max(2, int(w * 0.04)):
                    grupos.append((inicio, idx - 1))
                inicio = None
        if inicio is not None and w - inicio >= max(2, int(w * 0.04)):
            grupos.append((inicio, w - 1))

        partes = []
        for gx1, gx2 in grupos:
            sub = roi[:, gx1 : gx2 + 1]
            coords = cv2.findNonZero(sub)
            if coords is None:
                continue
            sx, sy, sw, sh = cv2.boundingRect(coords)
            if sh < h * 0.35 and sw < w * 0.20:
                continue
            if sw < ancho * ANCHO_MIN_CARACTER_REL:
                continue
            partes.append((x + gx1 + sx, y + sy, sw, sh))

        if 2 <= len(partes) <= 4:
            salida.extend(partes)
        else:
            salida.append((x, y, w, h))
    return _fusionar_bboxes(salida)

def _intentar_dividir_componente_ancho(subimg, bbox_ancho: tuple[int, int, int, int], ancho_promedio_resto: float) -> list[tuple[int, int, int, int]]:
    # FIX-5: dividir caracteres unidos solo si existe un valle claro en la proyeccion vertical.
    x, y, w, h = bbox_ancho
    if w < 1.7 * max(float(ancho_promedio_resto), 1.0):
        return [bbox_ancho]

    inicio = int(w * 0.25)
    fin = int(w * 0.75)
    if fin <= inicio + 1:
        return [bbox_ancho]

    proy = np.sum(subimg[:, inicio:fin] > 0, axis=0).astype(np.float32)
    if proy.size == 0:
        return [bbox_ancho]

    max_proy = float(np.max(proy))
    if max_proy <= 0:
        return [bbox_ancho]

    valle_idx = int(np.argmin(proy))
    valle_valor = float(proy[valle_idx])
    if valle_valor < 0.40 * max_proy:
        punto_corte = inicio + valle_idx
        if punto_corte <= 2 or punto_corte >= w - 2:
            return [bbox_ancho]
        bbox_izq = (x, y, punto_corte, h)
        bbox_der = (x + punto_corte, y, w - punto_corte, h)
        return [bbox_izq, bbox_der]

    return [bbox_ancho]

def _dividir_componentes_anchos_por_valle(binaria, bboxes: list[tuple[int, int, int, int]]) -> tuple[list[tuple[int, int, int, int]], int]:
    # FIX-5: aplicar la division conservadora solo a componentes claramente anchos.
    if len(bboxes) < 2:
        return bboxes, 0
    salida = []
    divididos = 0
    anchos = [bbox[2] for bbox in bboxes]
    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        resto = [valor for pos, valor in enumerate(anchos) if pos != idx]
        ancho_promedio_resto = float(np.mean(resto)) if resto else float(np.mean(anchos))
        subimg = binaria[y : y + h, x : x + w]
        partes = _intentar_dividir_componente_ancho(subimg, bbox, ancho_promedio_resto)
        if len(partes) > 1:
            divididos += 1
        salida.extend(partes)
    return sorted(salida, key=lambda item: item[0]), divididos

def _diagnosticar_segmentacion(aceptados: list[dict], rechazados: list[dict], ancho_banda: int, alto_banda: int) -> dict:
    cantidad = len(aceptados)
    motivos = _contar_motivos_rechazo(rechazados)
    primer_cortado = False
    if aceptados:
        x, y, w, h = aceptados[0]["bbox"]
        primer_cortado = x <= max(3, int(ancho_banda * 0.015)) and h >= alto_banda * 0.35

    if cantidad < CARACTERES_MIN_PLACA:
        causa = "segmentacion_incompleta"
    elif cantidad > CARACTERES_MAX_PLACA:
        causa = "ruido_detectado_como_caracter"
    elif motivos.get("posible_guion_o_ruido", 0) or motivos.get("aspecto_muy_delgado", 0):
        causa = "guion_detectado_como_caracter"
    elif primer_cortado:
        causa = "primer_caracter_cortado"
    elif motivos.get("altura_pequena", 0) > motivos.get("area_pequena", 0):
        causa = "texto_ecuador_interfiere"
    else:
        causa = "sin_error_evidente"

    return {
        "causa_probable": causa,
        "primer_caracter_posible_cortado": bool(primer_cortado),
        "exceso_componentes": cantidad > CARACTERES_MAX_PLACA,
        "menos_de_6": cantidad < CARACTERES_MIN_PLACA,
        "mas_de_7": cantidad > CARACTERES_MAX_PLACA,
        "motivos_rechazo": motivos,
    }

def _mensaje_segmentacion_por_cantidad(cantidad: int) -> str:
    if 6 <= cantidad <= 7:
        return "Segmentacion v2 ejecutada sobre banda principal de caracteres."
    if cantidad < 6:
        return "Segmentacion incompleta: se detectaron menos de 6 caracteres principales."
    return "Segmentacion con exceso de componentes: se conservaron los candidatos principales."

def _eliminar_items_duplicados(items: list[dict]) -> list[dict]:
    ordenados = sorted(items, key=lambda item: item["area"], reverse=True)
    filtrados = []
    for item in ordenados:
        bbox = _xywh_to_tuple(item["bbox"])
        if all(_iou_bbox(bbox, _xywh_to_tuple(existente["bbox"])) < 0.45 for existente in filtrados):
            filtrados.append(item)
    return filtrados

def _xywh_to_tuple(bbox: list[int]) -> tuple[int, int, int, int]:
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

def _contar_motivos_rechazo(rechazados: list[dict]) -> dict:
    conteo = {}
    for item in rechazados:
        motivo = item.get("motivo_rechazo") or "sin_motivo"
        conteo[motivo] = conteo.get(motivo, 0) + 1
    return conteo

def _eliminar_bboxes_duplicadas(candidatos: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ordenados = sorted(candidatos, key=lambda bbox: bbox[2] * bbox[3], reverse=True)
    filtrados = []
    for bbox in ordenados:
        if all(_iou_bbox(bbox, existente) < 0.45 for existente in filtrados):
            filtrados.append(bbox)
    return sorted(filtrados, key=lambda bbox: bbox[0])

def _iou_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


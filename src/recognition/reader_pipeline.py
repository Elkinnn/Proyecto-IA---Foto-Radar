from datetime import datetime
from pathlib import Path
from functools import lru_cache
import json
import traceback

import cv2
import numpy as np

from src.recognition.character_model import cargar_modelo_caracteres
from src.recognition.character_normalization import (
    normalizar_caracter_para_cnn,
    normalizar_imagen_caracter_para_cnn,
)
from src.recognition.image_utils import (
    _imread_seguro,
    asegurar_bgr,
    asegurar_grayscale,
    asegurar_rgb,
    describir_imagen,
)
from src.recognition.naming import (
    _huella_archivo_imagen,
    _nombre_lectura_unica,
    _nombre_seguro,
)
from src.recognition.preprocessing_filters import (
    _asegurar_caracteres_blancos,
    _denoise_componentes_placa,
    _quitar_lineas_finas_marco,
    _quitar_marco_rectangular,
)
from src.recognition.rectification import (
    _candidato_perspectiva_4_puntos,
    _candidato_rotacion_angulo,
    _estimar_angulo_skew_robusto,
    _mejorar_contraste_placa,
    _rotar_imagen_sin_cortar,
)
from src.recognition.segmentation import (
    segmentar_caracteres,
    segmentar_caracteres_camino_limpio,
    segmentar_caracteres_v2,
    segmentar_caracteres_v3_estrategias,
    _aislar_bloque_caracteres,
    _analizar_componentes_para_guiada_directa,
    _banda_binaria_desbordada,
    _bboxes_componentes_conectados,
    _bboxes_por_proyeccion_vertical,
    _bboxes_slots_formato,
    _binarizar_banda_robusta,
    _cajas_caracteres_bloque,
    _calcular_banda_util_slots_ecuador,
    _contar_motivos_rechazo,
    _descartar_guion_y_ruido_guiado,
    _diagnosticar_segmentacion,
    _dibujar_debug_estrategia,
    _dividir_bboxes_demasiado_anchas,
    _dividir_componentes_anchos_por_valle,
    _elegir_segmentacion,
    _eliminar_bboxes_duplicadas,
    _eliminar_items_duplicados,
    _eliminar_marco_placa,
    _es_guion_o_ruido_separador,
    _evaluar_estrategia_segmentacion,
    _evaluar_fallback_division_uniforme,
    _evaluar_segmentacion_guiada_formato_ecuador,
    _evaluar_slots_ecuador,
    _extraer_caracter_util_desde_slot,
    _fallback_division_uniforme,
    _filtrar_bboxes_caracteres,
    _filtrar_componentes_marco_borde,
    _filtrar_guion_fisico_bboxes,
    _fusionar_bboxes,
    _generar_bandas_candidatas,
    _guardar_caracteres_estrategia,
    _guardar_debug_slots_ecuador,
    _intentar_dividir_componente_ancho,
    _iou_bbox,
    _items_desde_bboxes_slots,
    _limpiar_banda_para_slots_ecuador,
    _limpiar_bordes_largos_banda,
    _limpiar_lineas_recorte_caracter,
    _mensaje_segmentacion_por_cantidad,
    _recortar_y_guardar_camino_limpio,
    _seleccionar_caracteres_principales,
    _serializar_estrategia_segmentacion,
    _sumar_stats_filtros_componentes,
    _xywh_to_tuple,
    calcular_puntaje_segmentacion,
    extraer_banda_caracteres,
)
from src.utils import normalizar_placa
from src.recognition.diagnostics import (
    registrar_diagnostico_recorte,
    registrar_reporte_ocr,
)
from src.recognition.postprocessing import (
    DIGITOS_PLACA,
    LETRAS_PLACA,
    _aplicar_mascara_formato,
    _confianzas_enmascaradas_ventana,
    _es_tipo_caracter,
    _mapa_confusion_para_tipo,
    _mejor_caracter_por_tipo,
    _mejor_top3_por_tipo,
    _normalizar_topk_prediccion,
    _opciones_por_posicion,
    _peso_lectura_evento,
    _puntuar_ventana_lectura,
    _tipo_posicion_formato,
    _tipo_posicion_placa,
    _top3_por_tipo,
    comparar_con_esperada,
    consolidar_lecturas_evento_placa,
    evaluar_calidad_lectura_evento,
    guardar_debug_votacion_evento,
    postprocesar_por_formato_ecuador,
    postprocesar_texto_placa,
    validar_formato_placa_ecuador,
)


RECONOCIMIENTO_CARACTERES_DIR = Path("reports") / "evidencias" / "reconocimiento_caracteres"
OCR_DIR = RECONOCIMIENTO_CARACTERES_DIR
PREPROCESADAS_DIR = OCR_DIR / "placas_preprocesadas"
CARACTERES_DIR = OCR_DIR / "caracteres_segmentados"
BANDAS_DIR = OCR_DIR / "bandas_caracteres"
DEBUG_SEGMENTACION_DIR = OCR_DIR / "debug_segmentacion"
DEBUG_RECTIFICACION_DIR = OCR_DIR / "debug_rectificacion"
VOTACION_EVENTOS_DIR = OCR_DIR / "votacion_eventos"
DEBUG_CNN_INPUTS_DIR = OCR_DIR / "debug_cnn_inputs"
DEBUG_SLOTS_ECUADOR_DIR = OCR_DIR / "debug_slots_ecuador"
DIAGNOSTICO_RECORTES_DIR = OCR_DIR / "diagnostico_recortes"
ERRORES_LECTOR_CNN_LOG = OCR_DIR / "errores_lector_cnn.log"
SIN_LECTURA_DEBUG_DIR = OCR_DIR / "sin_lectura_debug"
ERRORES_MONITOREO_CNN_LOG = OCR_DIR / "errores_monitoreo_cnn.log"
ERRORES_MONITOREO_CNN_DIR = OCR_DIR / "errores_monitoreo_cnn"

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


class PlateReader:
    def leer(self, deteccion: dict, modo_ocr: str, placa_manual: str | None = None) -> dict:
        if modo_ocr == "manual_controlado":
            placa = normalizar_placa(placa_manual)
            return {
                "texto": placa,
                "modo": modo_ocr,
                "confianza": 1.0 if placa else 0.0,
            "mensaje": "Lector de caracteres manual/controlado usado temporalmente.",
            }

        if modo_ocr == "automatico":
            return {
                "texto": "",
                "modo": modo_ocr,
                "confianza": 0.0,
                "mensaje": "Lector CNN de caracteres automatico pendiente de integrar con modelo entrenado desde cero.",
            }

        raise ValueError(f"Modo de lector de caracteres no soportado: {modo_ocr}")












def predecir_caracter(ruta_caracter: str, modelo, class_names: list) -> dict:
    normalizacion = normalizar_caracter_para_cnn(ruta_caracter)
    if normalizacion.get("array") is None:
        return {
            "caracter_predicho": "",
            "confianza": 0.0,
            "top3_predicciones": [],
            "probs_por_clase": {},
            "ruta_debug_normalizada": normalizacion.get("ruta_debug_normalizada"),
            "mensaje": normalizacion.get("mensaje", "No se pudo normalizar el caracter."),
        }

    entrada = normalizacion["array"]
    try:
        prediccion = modelo.predict(entrada, verbose=0)[0]
    except Exception as exc:
        _registrar_error_lector_cnn(
            ruta_imagen=ruta_caracter,
            etapa="prediccion_cnn",
            exc=exc,
            contexto={
                "shape_array": list(entrada.shape) if entrada is not None else None,
                "dtype_array": str(entrada.dtype) if entrada is not None else None,
                "min_array": float(entrada.min()) if entrada is not None else None,
                "max_array": float(entrada.max()) if entrada is not None else None,
            },
        )
        return {
            "caracter_predicho": "",
            "confianza": 0.0,
            "top3_predicciones": [],
            "probs_por_clase": {},
            "ruta_debug_normalizada": normalizacion.get("ruta_debug_normalizada"),
            "mensaje": f"Error en prediccion CNN: {exc}",
        }
    indices_top = np.argsort(prediccion)[-3:][::-1]
    top3 = [
        {
            "caracter": str(class_names[int(idx)]),
            "confianza": float(prediccion[int(idx)]),
        }
        for idx in indices_top
    ]
    probs_por_clase = {str(class_names[i]): float(prediccion[i]) for i in range(len(class_names))}
    mejor = top3[0] if top3 else {"caracter": "", "confianza": 0.0}
    return {
        "caracter_predicho": mejor["caracter"],
        "confianza": float(mejor["confianza"]),
        "top3_predicciones": top3,
        "probs_por_clase": probs_por_clase,
        "ruta_debug_normalizada": normalizacion.get("ruta_debug_normalizada"),
        "shape_array": normalizacion.get("shape_array"),
        "dtype_array": normalizacion.get("dtype_array"),
        "min_array": normalizacion.get("min_array"),
        "max_array": normalizacion.get("max_array"),
        "mensaje": "Caracter predicho con CNN propia.",
    }
















def _reconstruir_con_ventana_formato(caracteres_segmentados: list, modelo, class_names: list) -> dict:
    """Predice todos los caracteres, restringe cada posicion al tipo del formato
    Ecuador (3 letras + 3/4 numeros) y, si hay restos de marco (>7 slots), elige la
    ventana de 6 o 7 consecutivos con mejor formato y confianza enmascarada."""
    predicciones = []
    for indice, caracter in enumerate(caracteres_segmentados, start=1):
        prediccion = predecir_caracter(caracter.get("ruta_caracter", ""), modelo, class_names)
        prediccion["indice"] = indice
        prediccion["ruta_caracter"] = caracter.get("ruta_caracter")
        prediccion["bbox"] = caracter.get("bbox")
        predicciones.append(prediccion)

    n = len(predicciones)
    if n < CARACTERES_MIN_PLACA:
        ventana = predicciones
    elif n in (CARACTERES_MIN_PLACA, CARACTERES_MAX_PLACA):
        ventana = predicciones
    else:
        mejor_ventana = predicciones[:CARACTERES_MAX_PLACA]
        mejor_puntaje = -1e9
        for tam in (CARACTERES_MAX_PLACA, CARACTERES_MIN_PLACA):
            if n < tam:
                continue
            for inicio in range(0, n - tam + 1):
                candidata = predicciones[inicio : inicio + tam]
                puntaje = _puntuar_ventana_lectura(candidata)
                if puntaje > mejor_puntaje:
                    mejor_puntaje = puntaje
                    mejor_ventana = candidata
        ventana = mejor_ventana

    _aplicar_mascara_formato(ventana)
    for idx, pred in enumerate(ventana, start=1):
        pred["indice"] = idx
    texto = "".join(p.get("caracter_predicho", "") for p in ventana)
    confs = [float(p.get("confianza", 0.0)) for p in ventana]
    return {
        "texto_detectado": texto,
        "confianza_promedio": sum(confs) / len(confs) if confs else 0.0,
        "predicciones_caracteres": ventana,
    }


def reconstruir_placa_desde_caracteres(caracteres_segmentados: list, modelo, class_names: list) -> dict:
    return _reconstruir_con_ventana_formato(caracteres_segmentados, modelo, class_names)




















































def calcular_puntaje_preprocesamiento_placa(placa_procesada) -> dict:
    try:
        gris = asegurar_grayscale(placa_procesada)
    except Exception as exc:
        return {"puntaje_total": -999.0, "motivo": f"imagen_invalida: {exc}", "componentes": 0}
    alto, ancho = gris.shape[:2]
    if alto < 12 or ancho < 40:
        return {"puntaje_total": -200.0, "motivo": "placa_demasiado_pequena", "componentes": 0}

    aspect = ancho / max(alto, 1)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    contraste = float(gris.std())
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(binaria) > binaria.size * 0.55:
        binaria = cv2.bitwise_not(binaria)
    y1 = int(alto * 0.30)
    y2 = int(alto * 0.96)
    banda = binaria[y1:y2, :]
    banda = _limpiar_bordes_largos_banda(_asegurar_caracteres_blancos(banda))
    bboxes = _fusionar_bboxes(_bboxes_componentes_conectados(banda) + _bboxes_por_proyeccion_vertical(banda))
    bboxes = _dividir_bboxes_demasiado_anchas(banda, bboxes)
    aceptados, rechazados = _filtrar_bboxes_caracteres(bboxes, banda.shape[1], banda.shape[0])
    aceptados = _seleccionar_caracteres_principales(_eliminar_items_duplicados(aceptados), banda.shape[1], banda.shape[0])
    cantidad = len(aceptados)
    anchos = [item["ancho"] for item in aceptados]
    alturas = [item["alto"] for item in aceptados]

    score_aspect = max(0.0, 25.0 - abs(aspect - 3.8) * 8.0)
    score_nitidez = min(nitidez / 18.0, 20.0)
    score_contraste = min(contraste / 3.5, 18.0)
    score_cantidad = 30.0 if cantidad in (6, 7) else max(0.0, 22.0 - abs(6.5 - cantidad) * 8.0)
    score_alineacion = 0.0
    if aceptados:
        centros_y = np.array([item["bbox"][1] + item["bbox"][3] / 2 for item in aceptados], dtype=np.float32)
        score_alineacion = max(0.0, 12.0 - float(np.std(centros_y)) / max(banda.shape[0], 1) * 80.0)

    penalizacion = 0.0
    if aspect < 1.7 or aspect > 7.0:
        penalizacion += 28.0
    if cantidad < 6:
        penalizacion += (6 - cantidad) * 8.0
    if cantidad > 7:
        penalizacion += (cantidad - 7) * 6.0
    if anchos and max(anchos) > banda.shape[1] * 0.30:
        penalizacion += 14.0
    if alturas and min(alturas) < banda.shape[0] * 0.22:
        penalizacion += 6.0
    penalizacion += min(18.0, len(rechazados) * 0.7)

    puntaje = round(score_aspect + score_nitidez + score_contraste + score_cantidad + score_alineacion - penalizacion, 4)
    return {
        "puntaje_total": puntaje,
        "aspect_ratio": round(float(aspect), 4),
        "nitidez": round(nitidez, 4),
        "contraste": round(contraste, 4),
        "componentes": int(cantidad),
        "rechazados": int(len(rechazados)),
        "ancho": int(ancho),
        "alto": int(alto),
    }


def rectificar_placa_para_lector_cnn(crop_placa, nombre_base: str = "placa") -> dict:
    DEBUG_RECTIFICACION_DIR.mkdir(parents=True, exist_ok=True)
    nombre = _nombre_seguro(nombre_base, "placa")
    imagen_bgr = asegurar_bgr(crop_placa)
    # FIX-RECORTE: agregar margen al recorte YOLO para evitar cortes de caracteres y marco parcial.
    alto_r, ancho_r = imagen_bgr.shape[:2]
    margen_extra_y = max(2, int(alto_r * 0.04))
    margen_extra_x = max(3, int(ancho_r * 0.02))
    imagen_bgr = cv2.copyMakeBorder(
        imagen_bgr,
        margen_extra_y,
        margen_extra_y,
        margen_extra_x,
        margen_extra_x,
        cv2.BORDER_REPLICATE,
    )
    alto_original, ancho_original = imagen_bgr.shape[:2]
    aspect_ratio_original = ancho_original / max(alto_original, 1)
    # FIX-4: permitir rectificacion mas flexible cuando el recorte original viene demasiado cerrado.
    rectificacion_flexible = aspect_ratio_original < 2.2
    umbral_mejora_minima = 0.05 if rectificacion_flexible else 0.15
    calidad_minima_rectificacion = 12.0 if rectificacion_flexible else 18.0
    candidatos = []

    def agregar(etiqueta: str, metodo: str, imagen, extra: dict | None = None) -> None:
        if imagen is None or np.asarray(imagen).size == 0:
            return
        bgr = asegurar_bgr(imagen)
        puntaje = calcular_puntaje_preprocesamiento_placa(bgr)
        ruta = DEBUG_RECTIFICACION_DIR / f"{nombre}_{etiqueta}.jpg"
        cv2.imwrite(str(ruta), bgr)
        candidatos.append(
            {
                "etiqueta": etiqueta,
                "metodo": metodo,
                "imagen": bgr,
                "ruta": str(ruta),
                "puntaje": puntaje,
                "extra": extra or {},
            }
        )

    # FIX-DESKEW: enderezar SIEMPRE antes de generar candidatos, usando la linea
    # base de los caracteres. Con la placa recta el marco queda alineado a los ejes
    # (se elimina limpio) y los caracteres dejan de juntarse en diagonal.
    angulo_deskew = _estimar_angulo_skew_robusto(imagen_bgr)
    if angulo_deskew is not None and 3.0 <= abs(angulo_deskew) <= 25.0:
        base_deskew = _rotar_imagen_sin_cortar(imagen_bgr, angulo_deskew)
    else:
        base_deskew = imagen_bgr
        angulo_deskew = 0.0

    agregar("original", "sin_rectificacion", base_deskew, {"angulo_deskew": round(float(angulo_deskew), 4)})
    margen_y = max(4, int(base_deskew.shape[0] * 0.08))
    margen_x = max(8, int(base_deskew.shape[1] * 0.05))
    agregar("margen", "sin_rectificacion", cv2.copyMakeBorder(base_deskew, margen_y, margen_y, margen_x, margen_x, cv2.BORDER_REPLICATE))
    agregar("contraste", "sin_rectificacion", _mejorar_contraste_placa(base_deskew))

    rotada, angulo = _candidato_rotacion_angulo(base_deskew)
    agregar("rotacion", "rotacion_angulo", rotada, {"angulo": round(angulo, 4)})

    perspectiva = _candidato_perspectiva_4_puntos(base_deskew)
    agregar("perspectiva", "perspectiva_4_puntos", perspectiva)

    original = candidatos[0]
    mejor = max(candidatos, key=lambda item: item["puntaje"].get("puntaje_total", -999.0))
    if mejor["metodo"] != "sin_rectificacion":
        puntaje_mejor = mejor["puntaje"].get("puntaje_total", -999.0)
        puntaje_original = original["puntaje"].get("puntaje_total", -999.0)
        mejora_relativa = (puntaje_mejor - puntaje_original) / max(abs(puntaje_original), 1.0)
        if puntaje_mejor < calidad_minima_rectificacion or mejora_relativa < umbral_mejora_minima:
            mejor = original

    ruta_seleccionada = DEBUG_RECTIFICACION_DIR / f"{nombre}_seleccionada.jpg"
    cv2.imwrite(str(ruta_seleccionada), mejor["imagen"])
    resumen = {
        "metodo_rectificacion": mejor["metodo"],
        "confianza_rectificacion": float(mejor["puntaje"].get("puntaje_total", 0.0)),
        "puntaje_rectificacion": mejor["puntaje"],
        "aspect_ratio_original": round(float(aspect_ratio_original), 4),
        "rectificacion_flexible_aplicada": bool(rectificacion_flexible),
        "umbral_mejora_minima": float(umbral_mejora_minima),
        "ruta_original": candidatos[0]["ruta"],
        "ruta_seleccionada": str(ruta_seleccionada),
        "candidatos": [
            {
                "etiqueta": item["etiqueta"],
                "metodo": item["metodo"],
                "ruta": item["ruta"],
                "puntaje": item["puntaje"],
                "extra": item["extra"],
                "seleccionado": item is mejor,
            }
            for item in candidatos
        ],
    }
    return {"placa_rectificada": mejor["imagen"], **resumen}


def preprocesar_placa(ruta_imagen: str, nombre_base: str = "placa") -> dict:
    PREPROCESADAS_DIR.mkdir(parents=True, exist_ok=True)
    imagen = _imread_seguro(str(ruta_imagen))
    if imagen is None:
        return {
            "ruta_imagen_procesada": None,
            "estado": "error",
            "mensaje": "No se pudo cargar el recorte de placa.",
        }

    # nombre_base ya incorpora la huella del recorte cuando proviene del lector
    # completo. Preservarlo evita que otro evento sobrescriba estas evidencias.
    # Las llamadas independientes que dejan el valor por defecto conservan el
    # nombre derivado de su archivo de entrada.
    nombre = (
        _nombre_seguro(ruta_imagen, nombre_base)
        if nombre_base == "placa"
        else _nombre_seguro(nombre_base, "placa")
    )
    rectificacion = rectificar_placa_para_lector_cnn(imagen, nombre)
    preprocesamiento = preprocesar_recorte_placa_para_ocr(rectificacion["placa_rectificada"], nombre)
    preprocesamiento["rectificacion"] = {
        "metodo_rectificacion": rectificacion.get("metodo_rectificacion"),
        "confianza_rectificacion": rectificacion.get("confianza_rectificacion"),
        "puntaje_rectificacion": rectificacion.get("puntaje_rectificacion"),
        "aspect_ratio_original": rectificacion.get("aspect_ratio_original"),
        "rectificacion_flexible_aplicada": rectificacion.get("rectificacion_flexible_aplicada"),
        "umbral_mejora_minima": rectificacion.get("umbral_mejora_minima"),
        "ruta_original": rectificacion.get("ruta_original"),
        "ruta_seleccionada": rectificacion.get("ruta_seleccionada"),
        "candidatos": rectificacion.get("candidatos", []),
    }
    preprocesamiento["mensaje"] = (
        f"{preprocesamiento.get('mensaje', '')} Rectificacion: "
        f"{rectificacion.get('metodo_rectificacion')} "
        f"({float(rectificacion.get('confianza_rectificacion', 0.0)):.2f})."
    ).strip()
    return preprocesamiento


def preprocesar_recorte_placa_para_ocr(crop, nombre_base: str = "placa") -> dict:
    PREPROCESADAS_DIR.mkdir(parents=True, exist_ok=True)
    if crop is None or np.asarray(crop).size == 0:
        return {
            "ruta_imagen_procesada": None,
            "estado": "error",
            "mensaje": "Recorte vacio para reconocimiento de caracteres.",
        }

    gris = asegurar_grayscale(crop)
    alto_original, ancho_original = gris.shape[:2]
    escala = max(2.0, min(4.0, 420 / max(ancho_original, 1)))
    gris = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)

    alto, ancho = gris.shape[:2]
    margen_y = max(int(alto * 0.025), 1)
    gris = gris[margen_y : max(alto - margen_y, margen_y + 1), 0:ancho]
    gris = cv2.copyMakeBorder(
        gris,
        top=max(int(gris.shape[0] * 0.03), 2),
        bottom=max(int(gris.shape[0] * 0.04), 2),
        left=max(int(gris.shape[1] * 0.035), 6),
        right=max(int(gris.shape[1] * 0.035), 6),
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )

    # FIX-B: corregir iluminacion no uniforme antes de binarizar placas inclinadas o con gradiente.
    kernel_w = max(3, ancho // 4)
    kernel_h = max(3, alto // 4)
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    fondo_estimado = cv2.morphologyEx(gris, cv2.MORPH_DILATE, kernel_grande)
    fondo_estimado = np.maximum(fondo_estimado, 1).astype(np.float32)
    gris_normalizado = np.clip((gris.astype(np.float32) / fondo_estimado) * 128.0, 0, 255).astype(np.uint8)

    # FIX-B: CLAHE agresivo y tres binarizaciones combinadas para resistir angulo/iluminacion.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    contraste = clahe.apply(gris_normalizado)
    denoise = cv2.bilateralFilter(contraste, 5, 45, 45)
    suavizada = cv2.GaussianBlur(denoise, (3, 3), 0)
    _, otsu = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptativa = cv2.adaptiveThreshold(
        suavizada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8,
    )
    img_float = suavizada.astype(np.float32)
    media_local = cv2.blur(img_float, (15, 15))
    media_cuadrados = cv2.blur(img_float * img_float, (15, 15))
    std_local = np.sqrt(np.maximum(media_cuadrados - media_local * media_local, 0.0))
    umbral_sauvola = media_local * (1.0 + 0.15 * (std_local / 64.0 - 1.0))
    sauvola = np.where(img_float > umbral_sauvola, 255, 0).astype(np.uint8)
    procesada = cv2.bitwise_or(cv2.bitwise_and(otsu, adaptativa), sauvola)

    blancos = cv2.countNonZero(procesada)
    total = procesada.shape[0] * procesada.shape[1]
    if blancos > total * 0.65:
        procesada = cv2.bitwise_not(procesada)

    # FIX-B: morfologia proporcional al tamano del recorte.
    k = max(1, ancho // 120)
    kernel_ruido = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, k * 2), k))
    procesada = cv2.morphologyEx(procesada, cv2.MORPH_OPEN, kernel_ruido)
    procesada = cv2.morphologyEx(procesada, cv2.MORPH_CLOSE, kernel_horizontal)
    procesada = _asegurar_caracteres_blancos(procesada)
    # FIX-MARCO: quitar el borde rectangular de la placa antes de limpiar speckle.
    procesada = _quitar_marco_rectangular(procesada)
    # FIX-RUIDO: limpiar speckle por componentes para placas borrosas/diagonales.
    procesada = _denoise_componentes_placa(procesada)

    nombre = _nombre_seguro(nombre_base, "placa")
    ruta_gris = PREPROCESADAS_DIR / f"{nombre}_01_gris.jpg"
    ruta_gris_normalizado = PREPROCESADAS_DIR / f"{nombre}_01b_gris_normalizado.jpg"
    ruta_contraste = PREPROCESADAS_DIR / f"{nombre}_02_contraste.jpg"
    ruta_otsu = PREPROCESADAS_DIR / f"{nombre}_03_otsu.jpg"
    ruta_adaptativa = PREPROCESADAS_DIR / f"{nombre}_04_adaptativa.jpg"
    ruta_sauvola = PREPROCESADAS_DIR / f"{nombre}_05_sauvola.jpg"
    ruta_denoise = PREPROCESADAS_DIR / f"{nombre}_02b_denoise.jpg"
    ruta_salida = PREPROCESADAS_DIR / f"{nombre}_preprocesada.jpg"
    cv2.imwrite(str(ruta_gris), gris)
    cv2.imwrite(str(ruta_gris_normalizado), gris_normalizado)
    cv2.imwrite(str(ruta_contraste), contraste)
    cv2.imwrite(str(ruta_denoise), denoise)
    cv2.imwrite(str(ruta_otsu), otsu)
    cv2.imwrite(str(ruta_adaptativa), adaptativa)
    cv2.imwrite(str(ruta_sauvola), sauvola)
    cv2.imwrite(str(ruta_salida), procesada)

    return {
        "ruta_imagen_procesada": str(ruta_salida),
        "rutas_debug": {
            "gris": str(ruta_gris),
            "gris_normalizado": str(ruta_gris_normalizado),
            "contraste": str(ruta_contraste),
            "denoise": str(ruta_denoise),
            "otsu": str(ruta_otsu),
            "adaptativa": str(ruta_adaptativa),
            "sauvola": str(ruta_sauvola),
            "final": str(ruta_salida),
        },
        "estado": "ok",
        "mensaje": "Recorte preprocesado para reconocimiento de caracteres con contraste, reduccion de ruido y binarizacion robusta.",
    }
















































































































def diagnosticar_ocr_recorte(
    placa_esperada: str,
    texto_crudo: str,
    texto_postprocesado: str,
    predicciones_caracteres: list,
    caracteres_segmentados: list,
    formato_valido: bool,
    acierto: bool,
) -> dict:
    esperada = validar_formato_placa_ecuador(placa_esperada).get("texto_normalizado", "")
    cantidad_esperada = len(esperada) if esperada else 0
    cantidad_detectada = len(predicciones_caracteres)
    diagnostico_caracteres = []

    if esperada and cantidad_detectada != cantidad_esperada:
        causa_probable = "mala_segmentacion"
        mensaje = "Segmentacion incompleta o caracteres descartados: la cantidad detectada no coincide con la esperada."
    elif not predicciones_caracteres:
        causa_probable = "mala_segmentacion"
        mensaje = "No hay caracteres segmentados para diagnosticar."
    elif esperada and texto_postprocesado != esperada:
        causa_probable = "mala_prediccion_cnn"
        mensaje = "La cantidad coincide, pero una o mas predicciones no coinciden con la placa esperada."
    elif esperada and texto_crudo != esperada and texto_postprocesado == esperada:
        causa_probable = "postprocesamiento_corrigio"
        mensaje = "La CNN tuvo confusiones, pero el postprocesamiento por formato corrigio el resultado."
    elif not formato_valido:
        causa_probable = "postprocesamiento_insuficiente"
        mensaje = "El texto no cumple formato ecuatoriano despues del postprocesamiento."
    else:
        causa_probable = "sin_error_evidente"
        mensaje = "No se observa error evidente en segmentacion, normalizacion, CNN o postprocesamiento."

    max_items = max(cantidad_detectada, cantidad_esperada)
    for idx in range(max_items):
        pred = predicciones_caracteres[idx] if idx < cantidad_detectada else {}
        esperado = esperada[idx] if idx < cantidad_esperada else ""
        crudo = texto_crudo[idx] if idx < len(texto_crudo) else ""
        post = texto_postprocesado[idx] if idx < len(texto_postprocesado) else ""
        ruta_caracter = pred.get("ruta_caracter") or (
            caracteres_segmentados[idx].get("ruta_caracter") if idx < len(caracteres_segmentados) else ""
        )
        top3 = pred.get("top3_predicciones", []) or []
        top3_chars = [str(item.get("caracter", "")) for item in top3]
        confianza = float(pred.get("confianza", 0.0))

        if not pred:
            estado = "faltante_por_segmentacion"
            causa = "mala_segmentacion"
            observacion = "No existe caracter detectado para esta posicion esperada."
        elif esperado and pred.get("caracter_predicho") == esperado:
            estado = "correcto_crudo"
            causa = "sin_error_evidente"
            observacion = "La CNN acerto esta posicion."
        elif esperado and post == esperado:
            estado = "corregido_postprocesamiento"
            causa = "postprocesamiento_corrigio"
            observacion = "La prediccion cruda no coincidio, pero el postprocesamiento llego a la etiqueta esperada."
        elif esperado and esperado in top3_chars:
            estado = "esperada_en_top3"
            causa = "postprocesamiento_insuficiente"
            observacion = "La etiqueta esperada esta en top3; el postprocesamiento podria elegirla con una regla contextual."
        elif pred and confianza < 0.60:
            estado = "baja_confianza"
            causa = "mala_normalizacion_del_caracter"
            observacion = "Confianza baja; revisar imagen normalizada 32x32 y segmentacion."
        elif esperado and pred.get("caracter_predicho") != esperado:
            estado = "error_cnn"
            causa = "mala_prediccion_cnn"
            observacion = "La cantidad coincide, pero la CNN predijo otra clase."
        else:
            estado = "sin_etiqueta_esperada"
            causa = "diagnostico_limitado"
            observacion = "No hay etiqueta esperada para comparar esta posicion."

        diagnostico_caracteres.append(
            {
                "indice": idx + 1,
                "etiqueta_esperada": esperado,
                "caracter_crudo": crudo,
                "caracter_postprocesado": post,
                "prediccion": pred.get("caracter_predicho", ""),
                "confianza": confianza,
                "top3": top3,
                "ruta_caracter": ruta_caracter,
                "ruta_debug_normalizada": pred.get("ruta_debug_normalizada", ""),
                "estado": estado,
                "causa_probable": causa,
                "observacion": observacion,
            }
        )

    if esperada and cantidad_detectada == cantidad_esperada and not acierto:
        causas = {item.get("causa_probable") for item in diagnostico_caracteres}
        if "mala_normalizacion_del_caracter" in causas:
            causa_probable = "mala_normalizacion_del_caracter"
            mensaje = "La cantidad coincide, pero hay caracteres con baja confianza; revise la imagen normalizada 32x32."
        elif "postprocesamiento_insuficiente" in causas:
            causa_probable = "postprocesamiento_insuficiente"
            mensaje = "La etiqueta esperada aparece en top3 para algun caracter; falta una regla de postprocesamiento mas contextual."
        elif "mala_prediccion_cnn" in causas:
            causa_probable = "mala_prediccion_cnn"
            mensaje = "La segmentacion coincide en longitud, pero la CNN no ubica la etiqueta esperada como mejor clase."

    if not esperada:
        mensaje = "Ingrese placa esperada para diagnostico para comparar caracter por caracter."
        causa_probable = "diagnostico_limitado"

    return {
        "placa_esperada_normalizada": esperada,
        "cantidad_esperada": cantidad_esperada,
        "cantidad_detectada": cantidad_detectada,
        "caracteres_esperados": list(esperada),
        "caracteres_detectados": list(texto_crudo),
        "causa_probable": causa_probable,
        "mensaje": mensaje,
        "diagnostico_caracteres": diagnostico_caracteres,
        "formato_valido": formato_valido,
        "acierto": acierto,
    }


def _leer_placa_desde_recorte_core(
    ruta_imagen: str,
    placa_esperada: str = "",
    metodo_segmentacion: str = "v3_estrategias",
) -> dict:
    nombre_base = _nombre_lectura_unica(ruta_imagen, "placa")
    preprocesamiento = preprocesar_placa(ruta_imagen, nombre_base)
    caracteres = []
    segmentacion = {
        "metodo": metodo_segmentacion,
        "caracteres": [],
        "contornos_rechazados": [],
        "cantidad_aceptados": 0,
        "cantidad_rechazados": 0,
        "motivos_rechazo": {},
        "ruta_banda": None,
        "ruta_debug": None,
        "estado": "pendiente",
        "mensaje": "Segmentacion pendiente.",
    }
    if preprocesamiento.get("ruta_imagen_procesada"):
        if metodo_segmentacion == "v1_contornos_globales":
            caracteres = segmentar_caracteres(preprocesamiento["ruta_imagen_procesada"], nombre_base)
            segmentacion.update(
                {
                    "caracteres": caracteres,
                    "cantidad_aceptados": len(caracteres),
                    "estado": "ok",
                    "mensaje": "Segmentacion v1 ejecutada con contornos globales.",
                }
            )
        elif metodo_segmentacion == "v3_estrategias":
            segmentacion = segmentar_caracteres_v3_estrategias(preprocesamiento["ruta_imagen_procesada"], nombre_base)
            segmentacion["metodo"] = "v3_estrategias"
            caracteres = segmentacion.get("caracteres", [])
            if len(caracteres) < CARACTERES_MIN_PLACA:
                segmentacion_v2 = segmentar_caracteres_v2(preprocesamiento["ruta_imagen_procesada"], f"{nombre_base}_v2fallback")
                if len(segmentacion_v2.get("caracteres", [])) > len(caracteres):
                    segmentacion_v2["fallback_usado"] = "v2_banda_caracteres"
                    segmentacion = segmentacion_v2
                    caracteres = segmentacion.get("caracteres", [])
            # Camino limpio (additivo): binarizacion robusta + aislamiento de marco que
            # compite con v3. Parte del gris (antes del binarizado global que desborda).
            try:
                rutas_debug = preprocesamiento.get("rutas_debug") or {}
                ruta_gris_limpio = rutas_debug.get("contraste") or rutas_debug.get("gris_normalizado") or rutas_debug.get("gris")
                seg_limpio = segmentar_caracteres_camino_limpio(
                    preprocesamiento["ruta_imagen_procesada"], ruta_gris_limpio, f"{nombre_base}_limpio"
                )
                segmentacion, caracteres = _elegir_segmentacion(segmentacion, caracteres, seg_limpio)
            except Exception as exc:
                _registrar_error_lector_cnn(
                    ruta_imagen=ruta_imagen,
                    etapa="segmentacion_camino_limpio",
                    exc=exc,
                    contexto={"ruta_preprocesada": preprocesamiento.get("ruta_imagen_procesada")},
                )
        else:
            segmentacion = segmentar_caracteres_v2(preprocesamiento["ruta_imagen_procesada"], nombre_base)
            segmentacion["metodo"] = "v2_banda_caracteres"
            caracteres = segmentacion.get("caracteres", [])
            if len(caracteres) < CARACTERES_MIN_PLACA:
                try:
                    caracteres_legacy = segmentar_caracteres(preprocesamiento["ruta_imagen_procesada"], f"{nombre_base}_legacy")
                    if len(caracteres_legacy) > len(caracteres):
                        segmentacion["fallback_usado"] = "v1_contornos_globales"
                        segmentacion["mensaje_fallback"] = (
                            f"Segmentacion mejorada devolvio {len(caracteres)} caracteres; "
                            f"fallback legacy devolvio {len(caracteres_legacy)}."
                        )
                        caracteres = caracteres_legacy
                        segmentacion["caracteres"] = caracteres
                        segmentacion["cantidad_aceptados"] = len(caracteres)
                        segmentacion["estado"] = "ok" if len(caracteres) >= CARACTERES_MIN_PLACA else "advertencia"
                        segmentacion["mensaje"] = _mensaje_segmentacion_por_cantidad(len(caracteres))
                except Exception as exc:
                    _registrar_error_lector_cnn(
                        ruta_imagen=ruta_imagen,
                        etapa="segmentacion_fallback",
                        exc=exc,
                        contexto={"ruta_preprocesada": preprocesamiento.get("ruta_imagen_procesada")},
                    )

    carga_modelo, mensaje_modelo = cargar_modelo_caracteres()
    texto_detectado = "NO_RECONOCIDO"
    texto_detectado_crudo = ""
    texto_postprocesado = ""
    texto_normalizado = ""
    confianza_promedio = 0.0
    predicciones_caracteres = []
    cambios_postprocesamiento = []

    if carga_modelo is None:
        estado = "requiere_modelo_caracteres"
        mensaje = "La placa fue preprocesada y segmentada, pero aun no existe un modelo propio de caracteres para reconocer el texto."
    else:
        modelo, class_names = carga_modelo
        reconstruccion = reconstruir_placa_desde_caracteres(caracteres, modelo, class_names)
        texto_detectado_crudo = reconstruccion["texto_detectado"]
        postprocesado = postprocesar_por_formato_ecuador(texto_detectado_crudo, reconstruccion["predicciones_caracteres"])
        texto_postprocesado = postprocesado["texto_postprocesado"]
        texto_detectado = texto_detectado_crudo or "NO_RECONOCIDO"
        texto_normalizado = texto_postprocesado
        confianza_promedio = reconstruccion["confianza_promedio"]
        predicciones_caracteres = reconstruccion["predicciones_caracteres"]
        cambios_postprocesamiento = postprocesado.get("cambios", [])
        estado = "ok" if texto_normalizado else "sin_caracteres_segmentados"
        mensaje = (
            "Reconocimiento de caracteres ejecutado con CNN propia."
            if texto_normalizado
            else "No hay caracteres segmentados suficientes para reconstruir texto."
        )

    formato = validar_formato_placa_ecuador(texto_normalizado)
    comparacion = comparar_con_esperada(texto_normalizado, placa_esperada)
    diagnostico = diagnosticar_ocr_recorte(
        placa_esperada=placa_esperada,
        texto_crudo=texto_detectado_crudo,
        texto_postprocesado=texto_postprocesado,
        predicciones_caracteres=predicciones_caracteres,
        caracteres_segmentados=caracteres,
        formato_valido=bool(formato.get("valido")),
        acierto=bool(comparacion.get("coincide")),
    )
    causa_probable = _causa_probable_ocr(
        preprocesamiento=preprocesamiento,
        segmentacion=segmentacion,
        formato=formato,
        confianza_promedio=confianza_promedio,
    )

    return {
        "ruta_imagen": str(ruta_imagen),
        "id_lectura": nombre_base,
        "huella_recorte": nombre_base.rsplit("_", 1)[-1],
        "preprocesamiento": preprocesamiento,
        "rectificacion": preprocesamiento.get("rectificacion", {}),
        "metodo_rectificacion": (preprocesamiento.get("rectificacion") or {}).get("metodo_rectificacion"),
        "puntaje_rectificacion": (preprocesamiento.get("rectificacion") or {}).get("confianza_rectificacion"),
        "metodo_segmentacion": metodo_segmentacion,
        "segmentacion": segmentacion,
        "caracteres": caracteres,
        "cantidad_caracteres_segmentados": len(caracteres),
        "cantidad_contornos_rechazados": segmentacion.get("cantidad_rechazados", 0),
        "motivos_rechazo": segmentacion.get("motivos_rechazo", {}),
        "ruta_banda": segmentacion.get("ruta_banda"),
        "ruta_debug_segmentacion": segmentacion.get("ruta_debug"),
        "texto_detectado": texto_detectado,
        "texto_detectado_crudo": texto_detectado_crudo,
        "texto_postprocesado": texto_postprocesado,
        "texto_normalizado": texto_normalizado,
        "confianza": confianza_promedio,
        "confianza_promedio": confianza_promedio,
        "predicciones_caracteres": predicciones_caracteres,
        "cambios_postprocesamiento": cambios_postprocesamiento,
        "formato": formato,
        "comparacion": comparacion,
        "diagnostico": diagnostico,
        "causa_probable": causa_probable,
        "estado": estado,
        "mensaje": mensaje,
        "mensaje_modelo": mensaje_modelo,
    }


def leer_placa_desde_recorte(
    ruta_imagen: str,
    placa_esperada: str = "",
    metodo_segmentacion: str = "v3_estrategias",
) -> dict:
    try:
        resultado = _leer_placa_desde_recorte_core(ruta_imagen, placa_esperada, metodo_segmentacion)
        return _normalizar_salida_lector_cnn(resultado)
    except Exception as exc:
        _registrar_error_lector_cnn(
            ruta_imagen=ruta_imagen,
            etapa="salida_interfaz",
            exc=exc,
            contexto={"placa_esperada": placa_esperada, "metodo_segmentacion": metodo_segmentacion},
        )
        return _resultado_error_lector_cnn(ruta_imagen, "salida_interfaz", exc)


def leer_placa_cnn_seguro_desde_monitoreo(crop_o_ruta, contexto: dict | None = None) -> dict:
    contexto = contexto or {}
    ERRORES_MONITOREO_CNN_DIR.mkdir(parents=True, exist_ok=True)
    ruta_recorte = None
    imagen = None
    try:
        if isinstance(crop_o_ruta, (str, Path)):
            ruta_recorte = Path(str(crop_o_ruta))
            imagen = _imread_seguro(str(ruta_recorte), cv2.IMREAD_UNCHANGED)
        else:
            try:
                imagen = np.asarray(crop_o_ruta)
            except Exception:
                imagen = None

        if imagen is None or imagen.size == 0:
            raise ValueError("El recorte enviado desde Monitoreo es None, vacio o no se puede cargar.")

        diagnostico_canales = {"antes": describir_imagen(imagen)}
        imagen = asegurar_bgr(imagen)
        diagnostico_canales["despues_normalizar_bgr"] = describir_imagen(imagen)

        if ruta_recorte is None or not ruta_recorte.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ruta_recorte = ERRORES_MONITOREO_CNN_DIR / f"recorte_monitoreo_{timestamp}.jpg"
            cv2.imwrite(str(ruta_recorte), imagen)

        resultado = leer_placa_desde_recorte(str(ruta_recorte))
        resultado = _normalizar_compatibilidad_monitoreo(resultado)
        resultado["contexto_monitoreo"] = contexto
        resultado["ruta_recorte_monitoreo"] = str(ruta_recorte)
        resultado["diagnostico_canales_monitoreo"] = diagnostico_canales
        if resultado.get("estado_lectura") == "error_cnn" or resultado.get("error"):
            _registrar_diagnostico_monitoreo_cnn(
                funcion="leer_placa_cnn_seguro_desde_monitoreo",
                etapa=resultado.get("etapa_error") or "lectura_interfaz",
                ruta_recorte=ruta_recorte,
                imagen=imagen,
                resultado=resultado,
                contexto=contexto,
            )
        return resultado
    except Exception as exc:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if imagen is not None and imagen.size:
            ruta_fallo = ERRORES_MONITOREO_CNN_DIR / f"recorte_fallo_{timestamp}.jpg"
            cv2.imwrite(str(ruta_fallo), imagen)
            ruta_recorte = ruta_fallo
        resultado = _resultado_error_lector_cnn(str(ruta_recorte or ""), "llamada_desde_monitoreo", exc)
        resultado = _normalizar_compatibilidad_monitoreo(resultado)
        resultado["diagnostico_canales_monitoreo"] = {
            "antes": describir_imagen(imagen),
            "despues_normalizar_bgr": None,
        }
        _registrar_diagnostico_monitoreo_cnn(
            funcion="leer_placa_cnn_seguro_desde_monitoreo",
            etapa="llamada_desde_monitoreo",
            ruta_recorte=ruta_recorte,
            imagen=imagen,
            resultado=resultado,
            contexto=contexto,
        )
        return resultado


def _normalizar_compatibilidad_monitoreo(resultado: dict) -> dict:
    if not resultado.get("estado_lectura"):
        resultado = _normalizar_salida_lector_cnn(resultado)
    segmentacion = resultado.get("segmentacion") or {}
    resultado.update(
        {
            "estrategia_segmentacion": segmentacion.get("estrategia_segmentacion") or segmentacion.get("metodo"),
            "puntaje_segmentacion": segmentacion.get("puntaje_segmentacion"),
            "debug": resultado.get("debug_images") or {},
        }
    )
    return resultado


def _registrar_diagnostico_monitoreo_cnn(
    funcion: str,
    etapa: str,
    ruta_recorte: Path | None,
    imagen,
    resultado: dict,
    contexto: dict,
) -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    registro = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "funcion": funcion,
        "etapa": etapa,
        "ruta_recorte": str(ruta_recorte or ""),
        "recorte_es_none": imagen is None,
        "shape_recorte": list(imagen.shape) if imagen is not None else None,
        "dtype_recorte": str(imagen.dtype) if imagen is not None else None,
        "min_recorte": float(np.min(imagen)) if imagen is not None and imagen.size else None,
        "max_recorte": float(np.max(imagen)) if imagen is not None and imagen.size else None,
        "cantidad_caracteres_segmentados": resultado.get("cantidad_caracteres_segmentados", 0),
        "claves_resultado": sorted(resultado.keys()),
        "estado_lectura": resultado.get("estado_lectura"),
        "motivo": resultado.get("motivo"),
        "error": resultado.get("error"),
        "traceback": resultado.get("traceback"),
        "contexto": contexto,
    }
    with open(ERRORES_MONITOREO_CNN_LOG, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def _normalizar_salida_lector_cnn(resultado: dict) -> dict:
    resultado = resultado or {}
    formato = resultado.get("formato") or {}
    texto_crudo = resultado.get("texto_detectado_crudo") or resultado.get("texto_crudo") or ""
    texto_formato = resultado.get("texto_postprocesado") or resultado.get("texto_corregido_formato") or ""
    confianza = resultado.get("confianza_promedio")
    cantidad = int(resultado.get("cantidad_caracteres_segmentados", 0) or 0)
    error_lector = resultado.get("error")
    if resultado.get("estado") == "requiere_modelo_caracteres":
        error_lector = resultado.get("mensaje_modelo") or resultado.get("mensaje") or "No se pudo cargar la CNN de caracteres."
    elif cantidad > 0 and not texto_crudo and not texto_formato and not resultado.get("predicciones_caracteres"):
        error_lector = (
            resultado.get("mensaje_modelo")
            or "La segmentacion produjo caracteres, pero la CNN no devolvio predicciones."
        )
    estado_lectura, motivo = _determinar_estado_lectura(
        texto_crudo=texto_crudo,
        texto_corregido=texto_formato,
        cantidad_caracteres=cantidad,
        formato_valido=bool(formato.get("valido")),
        error=error_lector,
        causa_probable=resultado.get("causa_probable"),
    )
    ok = bool(resultado.get("ok", True)) and not bool(error_lector) and resultado.get("estado") != "error"
    resultado.update(
        {
            "ok": ok,
            "estado_lectura": estado_lectura,
            "motivo": motivo,
            "motivo_sin_lectura": motivo if estado_lectura in {"sin_caracteres", "segmentacion_incompleta", "error_cnn"} else "",
            "texto_crudo": texto_crudo,
            "texto_corregido_formato": texto_formato,
            "confianza_cnn_caracteres": confianza,
            "formato_valido": bool(formato.get("valido")),
            "error": error_lector,
            "etapa_error": resultado.get("etapa_error"),
            "debug_images": {
                "preprocesada": (resultado.get("preprocesamiento") or {}).get("ruta_imagen_procesada"),
                "rectificacion": (resultado.get("rectificacion") or {}).get("ruta_seleccionada"),
                "banda": resultado.get("ruta_banda"),
                "segmentacion": resultado.get("ruta_debug_segmentacion"),
            },
            "caracteres_segmentados": resultado.get("caracteres", []),
            "estrategia_segmentacion": (resultado.get("segmentacion") or {}).get("estrategia_segmentacion") or (resultado.get("segmentacion") or {}).get("metodo"),
            "puntaje_segmentacion": (resultado.get("segmentacion") or {}).get("puntaje_segmentacion"),
            "debug": {
                "preprocesada": (resultado.get("preprocesamiento") or {}).get("ruta_imagen_procesada"),
                "rectificacion": (resultado.get("rectificacion") or {}).get("ruta_seleccionada"),
                "banda": resultado.get("ruta_banda"),
                "segmentacion": resultado.get("ruta_debug_segmentacion"),
            },
        }
    )
    if estado_lectura != "lectura_completa":
        resultado["ruta_debug_sin_lectura"] = _guardar_debug_sin_lectura(resultado)
    return resultado


def _determinar_estado_lectura(
    texto_crudo: str,
    texto_corregido: str,
    cantidad_caracteres: int,
    formato_valido: bool,
    error: str | None,
    causa_probable: str | None,
) -> tuple[str, str]:
    if error:
        return "error_cnn", str(error)
    if cantidad_caracteres <= 0:
        return "sin_caracteres", causa_probable or "segmentacion_no_detecto_caracteres"
    if cantidad_caracteres < CARACTERES_MIN_PLACA:
        return "lectura_parcial", causa_probable or "segmentacion_incompleta"
    if texto_crudo and not formato_valido:
        return "formato_dudoso", causa_probable or "formato_invalido"
    if texto_corregido and formato_valido:
        return "lectura_completa", "formato_ecuatoriano_valido"
    return "segmentacion_incompleta", causa_probable or "sin_texto_reconocido"


def _guardar_debug_sin_lectura(resultado: dict) -> str:
    SIN_LECTURA_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ruta_imagen = Path(str(resultado.get("ruta_imagen", "recorte")))
    nombre = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{_nombre_seguro(str(ruta_imagen), 'recorte')}"
    salida = SIN_LECTURA_DEBUG_DIR / nombre
    salida.mkdir(parents=True, exist_ok=True)

    debug_images = resultado.get("debug_images") or {}
    rutas_a_copiar = {
        "recorte_original": resultado.get("ruta_imagen"),
        "rectificacion_seleccionada": debug_images.get("rectificacion"),
        "placa_preprocesada": debug_images.get("preprocesada"),
        "banda_caracteres": debug_images.get("banda"),
        "debug_segmentacion": debug_images.get("segmentacion"),
    }
    for etiqueta, ruta in rutas_a_copiar.items():
        if not ruta:
            continue
        origen = Path(str(ruta))
        if origen.exists():
            imagen = _imread_seguro(str(origen), cv2.IMREAD_UNCHANGED)
            if imagen is not None:
                cv2.imwrite(str(salida / f"{etiqueta}{origen.suffix or '.jpg'}"), imagen)

    caracteres = resultado.get("caracteres_segmentados") or resultado.get("caracteres") or []
    for idx, caracter in enumerate(caracteres, start=1):
        ruta_char = caracter.get("ruta_caracter")
        if ruta_char and Path(ruta_char).exists():
            imagen_char = _imread_seguro(str(ruta_char), cv2.IMREAD_UNCHANGED)
            if imagen_char is not None:
                cv2.imwrite(str(salida / f"caracter_{idx:02d}.png"), imagen_char)

    shape_recorte = None
    if resultado.get("ruta_imagen"):
        imagen = _imread_seguro(str(resultado["ruta_imagen"]))
        if imagen is not None:
            shape_recorte = list(imagen.shape)

    datos = {
        "motivo_sin_lectura": resultado.get("motivo_sin_lectura") or resultado.get("motivo"),
        "estado_lectura": resultado.get("estado_lectura"),
        "etapa": resultado.get("etapa_error") or "lector_cnn",
        "metodo_rectificacion": resultado.get("metodo_rectificacion") or (resultado.get("rectificacion") or {}).get("metodo_rectificacion"),
        "puntaje_rectificacion": resultado.get("puntaje_rectificacion") or (resultado.get("rectificacion") or {}).get("confianza_rectificacion"),
        "shape_recorte": shape_recorte,
        "cantidad_caracteres_segmentados": resultado.get("cantidad_caracteres_segmentados", 0),
        "cantidad_caracteres_enviados_cnn": len(resultado.get("predicciones_caracteres") or []),
        "predicciones_si_existen": resultado.get("predicciones_caracteres", []),
        "confianza_si_existe": resultado.get("confianza_cnn_caracteres"),
        "texto_crudo": resultado.get("texto_crudo"),
        "texto_corregido": resultado.get("texto_corregido_formato"),
        "formato_valido": resultado.get("formato_valido"),
        "excepcion": resultado.get("error"),
        "traceback": resultado.get("traceback"),
    }
    (salida / "sin_lectura_debug.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(salida)


def _resultado_error_lector_cnn(ruta_imagen: str, etapa: str, exc: Exception) -> dict:
    mensaje = f"{type(exc).__name__}: {exc}"
    return {
        "ok": False,
        "ruta_imagen": str(ruta_imagen),
        "preprocesamiento": {},
        "segmentacion": {},
        "caracteres": [],
        "caracteres_segmentados": [],
        "cantidad_caracteres_segmentados": 0,
        "texto_detectado": "NO_RECONOCIDO",
        "texto_detectado_crudo": "",
        "texto_crudo": "",
        "texto_postprocesado": "",
        "texto_corregido_formato": "",
        "texto_normalizado": "",
        "confianza": 0.0,
        "confianza_promedio": 0.0,
        "confianza_cnn_caracteres": 0.0,
        "predicciones_caracteres": [],
        "formato": {"valido": False, "mensaje": "No se pudo ejecutar el lector CNN."},
        "formato_valido": False,
        "comparacion": {},
        "diagnostico": {},
        "causa_probable": "error_lector_cnn",
        "estado": "error",
        "mensaje": mensaje,
        "error": mensaje,
        "etapa_error": etapa,
        "traceback": traceback.format_exc(),
        "debug_images": {},
        "mensaje_modelo": "",
    }


def _registrar_error_lector_cnn(ruta_imagen: str, etapa: str, exc: Exception, contexto: dict | None = None) -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "ruta_imagen": str(ruta_imagen),
        "etapa": etapa,
        "error": f"{type(exc).__name__}: {exc}",
        "contexto": contexto or {},
        "traceback": traceback.format_exc(),
    }
    try:
        imagen = _imread_seguro(str(ruta_imagen))
        if imagen is not None:
            info["shape_recorte_original"] = list(imagen.shape)
            info["dtype_recorte_original"] = str(imagen.dtype)
    except Exception:
        pass
    with open(ERRORES_LECTOR_CNN_LOG, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(info, ensure_ascii=False) + "\n")


def _causa_probable_ocr(preprocesamiento: dict, segmentacion: dict, formato: dict, confianza_promedio: float) -> str:
    causa_segmentacion = segmentacion.get("causa_probable_segmentacion")
    if causa_segmentacion and causa_segmentacion != "sin_error_evidente":
        return causa_segmentacion
    cantidad = int(segmentacion.get("cantidad_aceptados", 0) or 0)
    if cantidad < 6:
        return "segmentacion_incompleta"
    if not formato.get("valido"):
        return "formato_invalido"
    if confianza_promedio and confianza_promedio < 0.60:
        return "baja_confianza_cnn_caracteres"
    mensaje_pre = (preprocesamiento.get("mensaje") or "").lower()
    if "borroso" in mensaje_pre:
        return "recorte_borroso"
    return "sin_error_evidente"





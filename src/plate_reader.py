from datetime import datetime
from pathlib import Path
import csv
import json
import re

import cv2
import numpy as np

from src.utils import normalizar_placa


OCR_DIR = Path("reports") / "evidencias" / "ocr"
PREPROCESADAS_DIR = OCR_DIR / "placas_preprocesadas"
CARACTERES_DIR = OCR_DIR / "caracteres_segmentados"
BANDAS_DIR = OCR_DIR / "bandas_caracteres"
DEBUG_SEGMENTACION_DIR = OCR_DIR / "debug_segmentacion"
MODELO_CARACTERES_PATH = Path("models") / "character_reader" / "character_reader.keras"

Y_INICIO_BANDA = 0.30
Y_FIN_BANDA = 0.95
ALTURA_MIN_CARACTER_REL = 0.35
ALTURA_MAX_CARACTER_REL = 0.95
ANCHO_MIN_CARACTER_REL = 0.02
ANCHO_MAX_CARACTER_REL = 0.25
ASPECT_MIN_CARACTER = 0.15
ASPECT_MAX_CARACTER = 1.20
AREA_MIN_CARACTER_REL = 0.005
AREA_MAX_CARACTER_REL = 0.40


class PlateReader:
    def leer(self, deteccion: dict, modo_ocr: str, placa_manual: str | None = None) -> dict:
        if modo_ocr == "manual_controlado":
            placa = normalizar_placa(placa_manual)
            return {
                "texto": placa,
                "modo": modo_ocr,
                "confianza": 1.0 if placa else 0.0,
                "mensaje": "OCR manual/controlado usado temporalmente.",
            }

        if modo_ocr == "automatico":
            return {
                "texto": "",
                "modo": modo_ocr,
                "confianza": 0.0,
                "mensaje": "OCR automatico pendiente de integrar con modelo entrenado desde cero.",
            }

        raise ValueError(f"Modo de OCR no soportado: {modo_ocr}")


def cargar_modelo_caracteres():
    MODELO_CARACTERES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODELO_CARACTERES_PATH.exists():
        return None, "No existe un modelo propio de caracteres. Entrene primero el clasificador de caracteres."
    return None, "Carga de modelo de caracteres pendiente de implementar para el modelo propio."


def _nombre_seguro(ruta_imagen: str, nombre_base: str) -> str:
    stem = Path(ruta_imagen).stem if ruta_imagen else nombre_base
    texto = re.sub(r"[^A-Za-z0-9_-]+", "_", stem or nombre_base)
    return texto[:80]


def preprocesar_placa(ruta_imagen: str, nombre_base: str = "placa") -> dict:
    PREPROCESADAS_DIR.mkdir(parents=True, exist_ok=True)
    imagen = cv2.imread(str(ruta_imagen))
    if imagen is None:
        return {
            "ruta_imagen_procesada": None,
            "estado": "error",
            "mensaje": "No se pudo cargar el recorte de placa.",
        }

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contraste = clahe.apply(gris)
    suavizada = cv2.GaussianBlur(contraste, (3, 3), 0)
    _, otsu = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptativa = cv2.adaptiveThreshold(
        suavizada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    procesada = cv2.bitwise_and(otsu, adaptativa)

    blancos = cv2.countNonZero(procesada)
    total = procesada.shape[0] * procesada.shape[1]
    if blancos > total * 0.65:
        procesada = cv2.bitwise_not(procesada)

    nombre = _nombre_seguro(ruta_imagen, nombre_base)
    ruta_gris = PREPROCESADAS_DIR / f"{nombre}_01_gris.jpg"
    ruta_contraste = PREPROCESADAS_DIR / f"{nombre}_02_contraste.jpg"
    ruta_otsu = PREPROCESADAS_DIR / f"{nombre}_03_otsu.jpg"
    ruta_adaptativa = PREPROCESADAS_DIR / f"{nombre}_04_adaptativa.jpg"
    ruta_salida = PREPROCESADAS_DIR / f"{nombre}_preprocesada.jpg"
    cv2.imwrite(str(ruta_gris), gris)
    cv2.imwrite(str(ruta_contraste), contraste)
    cv2.imwrite(str(ruta_otsu), otsu)
    cv2.imwrite(str(ruta_adaptativa), adaptativa)
    cv2.imwrite(str(ruta_salida), procesada)

    return {
        "ruta_imagen_procesada": str(ruta_salida),
        "rutas_debug": {
            "gris": str(ruta_gris),
            "contraste": str(ruta_contraste),
            "otsu": str(ruta_otsu),
            "adaptativa": str(ruta_adaptativa),
            "final": str(ruta_salida),
        },
        "estado": "ok",
        "mensaje": "Recorte preprocesado correctamente.",
    }


def segmentar_caracteres(ruta_imagen_procesada: str, nombre_base: str = "placa") -> list[dict]:
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    imagen = cv2.imread(str(ruta_imagen_procesada), cv2.IMREAD_GRAYSCALE)
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
        imagen = cv2.imread(str(imagen_preprocesada), cv2.IMREAD_GRAYSCALE)
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

    alto, ancho = imagen.shape[:2]
    y1 = int(alto * Y_INICIO_BANDA)
    y2 = int(alto * Y_FIN_BANDA)
    banda = imagen[y1:y2, 0:ancho].copy()
    nombre = _nombre_seguro(str(nombre_base), "placa")
    ruta_banda = BANDAS_DIR / f"{nombre}_banda_caracteres.jpg"
    cv2.imwrite(str(ruta_banda), banda)

    return {
        "ruta_banda": str(ruta_banda),
        "banda": banda,
        "bbox_banda": [0, y1, ancho, y2],
        "estado": "ok",
        "mensaje": "Banda principal de caracteres extraida.",
    }


def segmentar_caracteres_v2(ruta_imagen_procesada: str, nombre_base: str = "placa") -> dict:
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_SEGMENTACION_DIR.mkdir(parents=True, exist_ok=True)
    imagen = cv2.imread(str(ruta_imagen_procesada), cv2.IMREAD_GRAYSCALE)
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
    if cv2.countNonZero(banda) > (alto_banda * ancho_banda) * 0.55:
        binaria = cv2.bitwise_not(banda)
    else:
        binaria = banda.copy()

    kernel = np.ones((2, 2), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = cv2.cvtColor(banda, cv2.COLOR_GRAY2BGR)
    aceptados = []
    rechazados = []
    area_banda = alto_banda * ancho_banda

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        area = w * h
        aspect = w / max(h, 1)
        motivo = ""

        if x <= 1 or y <= 1 or x + w >= ancho_banda - 1 or y + h >= alto_banda - 1:
            motivo = "pegado_al_borde"
        elif area < area_banda * AREA_MIN_CARACTER_REL:
            motivo = "area_pequena"
        elif area > area_banda * AREA_MAX_CARACTER_REL:
            motivo = "area_grande"
        elif h < alto_banda * ALTURA_MIN_CARACTER_REL:
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
    aceptados.sort(key=lambda item: item["bbox"][0])

    for idx, item in enumerate(aceptados, start=1):
        x, y, w, h = item["bbox"]
        margen = 2
        x1 = max(x - margen, 0)
        y1 = max(y - margen, 0)
        x2 = min(x + w + margen, ancho_banda)
        y2 = min(y + h + margen, alto_banda)
        recorte = binaria[y1:y2, x1:x2]
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
        "estado": "ok",
        "mensaje": "Segmentacion v2 ejecutada sobre banda principal de caracteres.",
    }


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


def validar_formato_placa_ecuador(texto: str) -> dict:
    normalizada = normalizar_placa((texto or "").replace("-", ""))
    valido = bool(re.fullmatch(r"[A-Z]{3}[0-9]{4}", normalizada))
    return {
        "texto_normalizado": normalizada if valido else normalizada,
        "valido": valido,
        "mensaje": "Formato ecuatoriano valido." if valido else "Formato no valido para placa ecuatoriana ABC1234.",
    }


def comparar_con_esperada(texto_normalizado: str, placa_esperada: str) -> dict:
    esperada = validar_formato_placa_ecuador(placa_esperada).get("texto_normalizado", "")
    detectada = validar_formato_placa_ecuador(texto_normalizado).get("texto_normalizado", "")
    if not esperada:
        return {
            "placa_esperada_normalizada": "",
            "coincide": False,
            "mensaje": "No se ingreso placa esperada para comparar.",
        }
    coincide = bool(detectada and detectada == esperada)
    return {
        "placa_esperada_normalizada": esperada,
        "coincide": coincide,
        "mensaje": "La placa coincide con la esperada." if coincide else "La placa detectada no coincide o aun no fue reconocida.",
    }


def leer_placa_desde_recorte(
    ruta_imagen: str,
    placa_esperada: str = "",
    metodo_segmentacion: str = "v2_banda_caracteres",
) -> dict:
    nombre_base = _nombre_seguro(ruta_imagen, "placa")
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
        else:
            segmentacion = segmentar_caracteres_v2(preprocesamiento["ruta_imagen_procesada"], nombre_base)
            segmentacion["metodo"] = "v2_banda_caracteres"
            caracteres = segmentacion.get("caracteres", [])

    modelo, mensaje_modelo = cargar_modelo_caracteres()
    texto_detectado = "NO_RECONOCIDO"
    texto_normalizado = ""
    formato = validar_formato_placa_ecuador(texto_normalizado)
    comparacion = comparar_con_esperada(texto_normalizado, placa_esperada)

    estado = "requiere_modelo_caracteres" if modelo is None else "pendiente_clasificacion"
    mensaje = (
        "La placa fue preprocesada y segmentada, pero aun no existe un modelo propio de caracteres para reconocer el texto."
        if modelo is None
        else mensaje_modelo
    )

    return {
        "ruta_imagen": str(ruta_imagen),
        "preprocesamiento": preprocesamiento,
        "metodo_segmentacion": metodo_segmentacion,
        "segmentacion": segmentacion,
        "caracteres": caracteres,
        "cantidad_caracteres_segmentados": len(caracteres),
        "cantidad_contornos_rechazados": segmentacion.get("cantidad_rechazados", 0),
        "motivos_rechazo": segmentacion.get("motivos_rechazo", {}),
        "ruta_banda": segmentacion.get("ruta_banda"),
        "ruta_debug_segmentacion": segmentacion.get("ruta_debug"),
        "texto_detectado": texto_detectado,
        "texto_normalizado": texto_normalizado,
        "confianza": 0.0,
        "formato": formato,
        "comparacion": comparacion,
        "estado": estado,
        "mensaje": mensaje,
        "mensaje_modelo": mensaje_modelo,
    }


def registrar_reporte_ocr(resultado: dict, fuente_recorte: str, placa_esperada: str = "") -> dict:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    ruta_json = OCR_DIR / "reporte_ocr_experimental.json"
    ruta_csv = OCR_DIR / "reporte_ocr_experimental.csv"
    fila = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "ruta_imagen": resultado.get("ruta_imagen"),
        "fuente_recorte": fuente_recorte,
        "metodo_segmentacion": resultado.get("metodo_segmentacion"),
        "placa_esperada": placa_esperada,
        "texto_detectado": resultado.get("texto_detectado"),
        "texto_normalizado": resultado.get("texto_normalizado"),
        "valida_formato": resultado.get("formato", {}).get("valido"),
        "acierto": resultado.get("comparacion", {}).get("coincide"),
        "cantidad_caracteres_segmentados": resultado.get("cantidad_caracteres_segmentados", 0),
        "estado": resultado.get("estado"),
        "mensaje": resultado.get("mensaje"),
    }

    registros = []
    if ruta_json.exists():
        try:
            registros = json.loads(ruta_json.read_text(encoding="utf-8"))
            if not isinstance(registros, list):
                registros = []
        except json.JSONDecodeError:
            registros = []
    registros.append(fila)
    ruta_json.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

    existe_csv = ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(fila.keys()))
        if not existe_csv:
            writer.writeheader()
        writer.writerow(fila)

    return {
        "ruta_json": str(ruta_json),
        "ruta_csv": str(ruta_csv),
    }

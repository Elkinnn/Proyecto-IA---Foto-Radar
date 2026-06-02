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
DEBUG_CNN_INPUTS_DIR = OCR_DIR / "debug_cnn_inputs"
DIAGNOSTICO_RECORTES_DIR = OCR_DIR / "diagnostico_recortes"
MODELO_CARACTERES_PATH = Path("models") / "character_reader" / "character_cnn.keras"
CLASS_NAMES_PATH = Path("models") / "character_reader" / "class_names.json"

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
    if not CLASS_NAMES_PATH.exists():
        return None, "No existe class_names.json para interpretar las salidas del modelo de caracteres."
    try:
        import tensorflow as tf

        modelo = tf.keras.models.load_model(MODELO_CARACTERES_PATH)
    except ImportError:
        try:
            import keras

            modelo = keras.models.load_model(MODELO_CARACTERES_PATH)
        except ImportError:
            return None, "No esta instalado TensorFlow ni Keras en este entorno. No se puede cargar la CNN de caracteres."
        except Exception as exc:
            return None, f"No se pudo cargar el modelo propio de caracteres con Keras: {exc}"
    except Exception as exc:
        return None, f"No se pudo cargar el modelo propio de caracteres con TensorFlow: {exc}"

    try:
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"No se pudo cargar class_names.json: {exc}"
    return (modelo, class_names), "Modelo propio de caracteres cargado correctamente."


def normalizar_caracter_para_cnn(ruta_caracter: str) -> dict:
    DEBUG_CNN_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    imagen = cv2.imread(str(ruta_caracter), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return {
            "array": None,
            "ruta_debug_normalizada": None,
            "estado": "error",
            "mensaje": "No se pudo cargar el caracter segmentado.",
        }

    suavizada = cv2.GaussianBlur(imagen, (3, 3), 0)
    _, binaria = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blancos = cv2.countNonZero(binaria)
    total = binaria.shape[0] * binaria.shape[1]
    if blancos > total * 0.5:
        binaria = cv2.bitwise_not(binaria)

    coords = cv2.findNonZero(binaria)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binaria = binaria[y : y + h, x : x + w]

    alto, ancho = binaria.shape[:2]
    escala = min(26 / max(ancho, 1), 26 / max(alto, 1))
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    redimensionada = cv2.resize(binaria, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((32, 32), dtype=np.uint8)
    x0 = (32 - nuevo_ancho) // 2
    y0 = (32 - nuevo_alto) // 2
    lienzo[y0 : y0 + nuevo_alto, x0 : x0 + nuevo_ancho] = redimensionada

    nombre = _nombre_seguro(ruta_caracter, "caracter")
    ruta_debug = DEBUG_CNN_INPUTS_DIR / f"{nombre}_cnn_32x32.png"
    cv2.imwrite(str(ruta_debug), lienzo)

    entrada = lienzo.astype("float32") / 255.0
    entrada = entrada.reshape(1, 32, 32, 1)
    return {
        "array": entrada,
        "ruta_debug_normalizada": str(ruta_debug),
        "estado": "ok",
        "mensaje": "Caracter normalizado para CNN con proporcion conservada y padding negro.",
    }


def predecir_caracter(ruta_caracter: str, modelo, class_names: list) -> dict:
    normalizacion = normalizar_caracter_para_cnn(ruta_caracter)
    if normalizacion.get("array") is None:
        return {
            "caracter_predicho": "",
            "confianza": 0.0,
            "top3_predicciones": [],
            "ruta_debug_normalizada": normalizacion.get("ruta_debug_normalizada"),
            "mensaje": normalizacion.get("mensaje", "No se pudo normalizar el caracter."),
        }

    entrada = normalizacion["array"]
    prediccion = modelo.predict(entrada, verbose=0)[0]
    indices_top = np.argsort(prediccion)[-3:][::-1]
    top3 = [
        {
            "caracter": str(class_names[int(idx)]),
            "confianza": float(prediccion[int(idx)]),
        }
        for idx in indices_top
    ]
    mejor = top3[0] if top3 else {"caracter": "", "confianza": 0.0}
    return {
        "caracter_predicho": mejor["caracter"],
        "confianza": float(mejor["confianza"]),
        "top3_predicciones": top3,
        "ruta_debug_normalizada": normalizacion.get("ruta_debug_normalizada"),
        "mensaje": "Caracter predicho con CNN propia.",
    }


def reconstruir_placa_desde_caracteres(caracteres_segmentados: list, modelo, class_names: list) -> dict:
    predicciones = []
    texto = ""
    confianzas = []
    for indice, caracter in enumerate(caracteres_segmentados, start=1):
        prediccion = predecir_caracter(caracter.get("ruta_caracter", ""), modelo, class_names)
        prediccion["indice"] = indice
        prediccion["ruta_caracter"] = caracter.get("ruta_caracter")
        prediccion["bbox"] = caracter.get("bbox")
        predicciones.append(prediccion)
        texto += prediccion.get("caracter_predicho", "")
        confianzas.append(float(prediccion.get("confianza", 0.0)))

    confianza_promedio = sum(confianzas) / len(confianzas) if confianzas else 0.0
    return {
        "texto_detectado": texto,
        "confianza_promedio": confianza_promedio,
        "predicciones_caracteres": predicciones,
    }


def postprocesar_texto_placa(texto: str) -> dict:
    crudo = normalizar_placa((texto or "").replace("-", ""))
    letras = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}
    numeros = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8"}
    corregido = []
    for idx, caracter in enumerate(crudo):
        if idx < 3:
            corregido.append(letras.get(caracter, caracter))
        else:
            corregido.append(numeros.get(caracter, caracter))
    return {
        "texto_detectado_crudo": crudo,
        "texto_postprocesado": "".join(corregido),
    }


def postprocesar_por_formato_ecuador(texto_crudo: str, predicciones_caracteres: list) -> dict:
    crudo = normalizar_placa((texto_crudo or "").replace("-", ""))
    letras_directas = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}
    numeros_directos = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8", "G": "6"}
    caracteres = []
    cambios = []

    for idx, caracter in enumerate(crudo):
        pred = predicciones_caracteres[idx] if idx < len(predicciones_caracteres) else {}
        confianza = float(pred.get("confianza", 0.0))
        top3 = pred.get("top3_predicciones", []) or []
        top3_map = {str(item.get("caracter")): float(item.get("confianza", 0.0)) for item in top3}
        esperado = "letra" if idx < 3 else "numero"
        nuevo = caracter
        motivo = ""

        if esperado == "letra":
            if caracter.isalpha():
                nuevo = caracter
            else:
                candidato = letras_directas.get(caracter)
                if candidato and (candidato in top3_map or confianza < 0.85):
                    nuevo = candidato
                    motivo = f"{caracter}->{candidato} por posicion de letra"
                elif caracter == "6" and "G" in top3_map and confianza < 0.75:
                    nuevo = "G"
                    motivo = "6->G respaldado por top3"
                else:
                    mejor_letra = _mejor_top3_por_tipo(top3, tipo="letra", confianza_actual=confianza)
                    if mejor_letra:
                        nuevo = mejor_letra
                        motivo = f"{caracter}->{mejor_letra} por top3 en posicion de letra"
        else:
            if caracter.isdigit():
                nuevo = caracter
            else:
                candidato = numeros_directos.get(caracter)
                if candidato and (candidato in top3_map or confianza < 0.85):
                    nuevo = candidato
                    motivo = f"{caracter}->{candidato} por posicion numerica"
                else:
                    mejor_numero = _mejor_top3_por_tipo(top3, tipo="numero", confianza_actual=confianza)
                    if mejor_numero:
                        nuevo = mejor_numero
                        motivo = f"{caracter}->{mejor_numero} por top3 en posicion numerica"

        caracteres.append(nuevo)
        if nuevo != caracter:
            cambios.append(
                {
                    "indice": idx + 1,
                    "original": caracter,
                    "corregido": nuevo,
                    "confianza_original": confianza,
                    "motivo": motivo,
                }
            )

    texto = "".join(caracteres)
    formato = validar_formato_placa_ecuador(texto)
    return {
        "texto_detectado_crudo": crudo,
        "texto_postprocesado": texto,
        "formato_valido": formato["valido"],
        "cambios": cambios,
    }


def _mejor_top3_por_tipo(top3: list, tipo: str, confianza_actual: float) -> str:
    for item in top3:
        candidato = str(item.get("caracter", ""))
        confianza = float(item.get("confianza", 0.0))
        if tipo == "letra" and not candidato.isalpha():
            continue
        if tipo == "numero" and not candidato.isdigit():
            continue
        if confianza_actual < 0.80 or confianza >= confianza_actual - 0.20:
            return candidato
    return ""


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
    valido = bool(re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", normalizada))
    return {
        "texto_normalizado": normalizada if valido else normalizada,
        "valido": valido,
        "mensaje": "Formato ecuatoriano valido." if valido else "Formato no valido para placa ecuatoriana ABC123 o ABC1234.",
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
            "OCR experimental ejecutado con CNN propia de caracteres."
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
        "estado": estado,
        "mensaje": mensaje,
        "mensaje_modelo": mensaje_modelo,
    }


def registrar_diagnostico_recorte(resultado: dict, placa_esperada: str = "") -> dict:
    DIAGNOSTICO_RECORTES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_json = DIAGNOSTICO_RECORTES_DIR / "diagnostico_recortes.json"
    ruta_csv = DIAGNOSTICO_RECORTES_DIR / "diagnostico_recortes.csv"
    diagnostico = resultado.get("diagnostico", {})
    predicciones = resultado.get("predicciones_caracteres", [])
    filas_caracteres = []
    for item in diagnostico.get("diagnostico_caracteres", []):
        filas_caracteres.append(
            {
                "indice": item.get("indice"),
                "etiqueta_esperada": item.get("etiqueta_esperada"),
                "prediccion": item.get("prediccion"),
                "confianza": item.get("confianza"),
                "top3": item.get("top3"),
                "caracter_crudo": item.get("caracter_crudo"),
                "caracter_postprocesado": item.get("caracter_postprocesado"),
                "ruta_caracter": item.get("ruta_caracter"),
                "ruta_debug_normalizada": item.get("ruta_debug_normalizada"),
                "estado": item.get("estado"),
                "causa_probable": item.get("causa_probable"),
                "observacion": item.get("observacion"),
            }
        )

    registro = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "ruta_recorte": resultado.get("ruta_imagen"),
        "placa_esperada": placa_esperada,
        "placa_esperada_normalizada": diagnostico.get("placa_esperada_normalizada", ""),
        "texto_crudo": resultado.get("texto_detectado_crudo", ""),
        "texto_postprocesado": resultado.get("texto_postprocesado", ""),
        "formato_valido": resultado.get("formato", {}).get("valido"),
        "acierto": resultado.get("comparacion", {}).get("coincide"),
        "cantidad_esperada": diagnostico.get("cantidad_esperada", 0),
        "cantidad_detectada": diagnostico.get("cantidad_detectada", len(predicciones)),
        "predicciones_por_caracter": filas_caracteres,
        "confianza_por_caracter": [item.get("confianza") for item in filas_caracteres],
        "top3_por_caracter": [item.get("top3") for item in filas_caracteres],
        "causa_probable": diagnostico.get("causa_probable"),
        "mensaje": diagnostico.get("mensaje"),
    }

    registros = []
    if ruta_json.exists():
        try:
            registros = json.loads(ruta_json.read_text(encoding="utf-8"))
            if not isinstance(registros, list):
                registros = []
        except json.JSONDecodeError:
            registros = []
    registros.append(registro)
    ruta_json.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

    fila_csv = {
        "fecha_hora": registro["fecha_hora"],
        "ruta_recorte": registro["ruta_recorte"],
        "placa_esperada": registro["placa_esperada"],
        "texto_crudo": registro["texto_crudo"],
        "texto_postprocesado": registro["texto_postprocesado"],
        "formato_valido": registro["formato_valido"],
        "acierto": registro["acierto"],
        "cantidad_esperada": registro["cantidad_esperada"],
        "cantidad_detectada": registro["cantidad_detectada"],
        "prediccion_por_caracter": json.dumps([item.get("prediccion") for item in filas_caracteres], ensure_ascii=False),
        "confianza_por_caracter": json.dumps(registro["confianza_por_caracter"], ensure_ascii=False),
        "top3_por_caracter": json.dumps(registro["top3_por_caracter"], ensure_ascii=False),
        "causa_probable": registro["causa_probable"],
        "mensaje": registro["mensaje"],
    }
    existe_csv = ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(fila_csv.keys()))
        if not existe_csv:
            writer.writeheader()
        writer.writerow(fila_csv)

    return {
        "ruta_json": str(ruta_json),
        "ruta_csv": str(ruta_csv),
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
        "texto_detectado_crudo": resultado.get("texto_detectado_crudo"),
        "texto_postprocesado": resultado.get("texto_postprocesado"),
        "texto_normalizado": resultado.get("texto_normalizado"),
        "confianza_promedio": resultado.get("confianza_promedio", 0.0),
        "formato_valido": resultado.get("formato", {}).get("valido"),
        "valida_formato": resultado.get("formato", {}).get("valido"),
        "acierto": resultado.get("comparacion", {}).get("coincide"),
        "predicciones_caracteres": json.dumps(resultado.get("predicciones_caracteres", []), ensure_ascii=False),
        "cambios_postprocesamiento": json.dumps(resultado.get("cambios_postprocesamiento", []), ensure_ascii=False),
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

"""Normalizacion de caracteres para la CNN propia.

Convierte caracteres segmentados a entrada 32x32x1, manteniendo proporcion
con padding negro y caracter claro.
"""

from pathlib import Path

import cv2
import numpy as np

from src.recognition.image_utils import _imread_seguro, asegurar_grayscale
from src.recognition.naming import _nombre_seguro
from src.recognition.preprocessing_filters import _quitar_lineas_finas_marco


RECONOCIMIENTO_CARACTERES_DIR = Path("reports") / "evidencias" / "reconocimiento_caracteres"
OCR_DIR = RECONOCIMIENTO_CARACTERES_DIR
DEBUG_CNN_INPUTS_DIR = OCR_DIR / "debug_cnn_inputs"


def normalizar_caracter_para_cnn(ruta_caracter: str) -> dict:
    DEBUG_CNN_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    imagen = _imread_seguro(str(ruta_caracter), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return {
            "array": None,
            "ruta_debug_normalizada": None,
            "estado": "error",
            "mensaje": "No se pudo cargar el caracter segmentado.",
        }

    return normalizar_imagen_caracter_para_cnn(imagen, _nombre_seguro(ruta_caracter, "caracter"))

def normalizar_imagen_caracter_para_cnn(imagen, nombre_base: str = "caracter") -> dict:
    DEBUG_CNN_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if imagen is None or np.asarray(imagen).size == 0:
        return {
            "array": None,
            "ruta_debug_normalizada": None,
            "estado": "error",
            "mensaje": "Imagen de caracter vacia.",
        }

    imagen = asegurar_grayscale(imagen)
    suavizada = cv2.GaussianBlur(imagen, (3, 3), 0)
    _, binaria = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blancos = cv2.countNonZero(binaria)
    total = binaria.shape[0] * binaria.shape[1]
    if blancos > total * 0.5:
        binaria = cv2.bitwise_not(binaria)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    limpia = np.zeros_like(binaria)
    area_min = max(3, int(total * 0.01))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= area_min:
            limpia[labels == label] = 255
    if cv2.countNonZero(limpia) > 0:
        binaria = limpia

    # FIX-TOP: quitar solo lineas finas de marco/guion sin recortar los trazos
    # superiores o inferiores que pertenecen al propio caracter.
    binaria = _quitar_lineas_finas_marco(binaria)
    filas_inferiores_limpiadas = 0

    coords = cv2.findNonZero(binaria)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margen = 2
        x1 = max(x - margen, 0)
        y1 = max(y - margen, 0)
        x2 = min(x + w + margen, binaria.shape[1])
        y2 = min(y + h + margen, binaria.shape[0])
        binaria = binaria[y1:y2, x1:x2]

    alto, ancho = binaria.shape[:2]
    escala = min(24 / max(ancho, 1), 26 / max(alto, 1))
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    redimensionada = cv2.resize(binaria, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((32, 32), dtype=np.uint8)
    x0 = (32 - nuevo_ancho) // 2
    y0 = (32 - nuevo_alto) // 2
    lienzo[y0 : y0 + nuevo_alto, x0 : x0 + nuevo_ancho] = redimensionada

    nombre = _nombre_seguro(nombre_base, "caracter")
    ruta_debug = DEBUG_CNN_INPUTS_DIR / f"{nombre}_cnn_32x32.png"
    cv2.imwrite(str(ruta_debug), lienzo)

    entrada = lienzo.astype("float32") / 255.0
    entrada = entrada.reshape(1, 32, 32, 1)
    return {
        "array": entrada,
        "ruta_debug_normalizada": str(ruta_debug),
        "estado": "ok",
        "mensaje": "Caracter normalizado para CNN con proporcion conservada y padding negro.",
        "shape_array": list(entrada.shape),
        "dtype_array": str(entrada.dtype),
        "min_array": float(entrada.min()),
        "max_array": float(entrada.max()),
        "filas_inferiores_limpiadas": int(filas_inferiores_limpiadas),
    }


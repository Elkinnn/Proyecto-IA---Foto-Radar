"""Filtros OpenCV reutilizables para limpiar placas y caracteres."""

import cv2
import numpy as np

from src.recognition.image_utils import asegurar_grayscale


def _quitar_lineas_finas_marco(binaria: np.ndarray) -> np.ndarray:
    # FIX-TOP: elimina SOLO lineas finas tipo marco metalico o guion fisico
    # (componentes muy alargados y delgados). Los trazos superiores/inferiores de
    # un caracter (E, F, T, Z, 5, 7) forman parte del propio componente del
    # caracter, por lo que NO se borran.
    if binaria is None or binaria.size == 0:
        return binaria
    alto, ancho = binaria.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if num_labels <= 1:
        return binaria
    salida = binaria.copy()
    grosor_max_h = max(2, int(round(alto * 0.06)))
    grosor_max_v = max(2, int(round(ancho * 0.06)))
    for label in range(1, num_labels):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        es_linea_h = w >= ancho * 0.80 and h <= grosor_max_h
        es_linea_v = h >= alto * 0.80 and w <= grosor_max_v
        if es_linea_h or es_linea_v:
            salida[labels == label] = 0
    if cv2.countNonZero(salida) == 0:
        return binaria
    return salida

def _denoise_componentes_placa(binaria: np.ndarray) -> np.ndarray:
    # FIX-RUIDO: eliminar motas/speckle de la binarizacion en placas borrosas o
    # diagonales, conservando los componentes con altura tipica de caracter. Esto
    # evita que aparezcan "zonas que se toman y no deberian" y que un caracter
    # agarre basura vecina.
    if binaria is None or binaria.size == 0:
        return binaria
    alto, ancho = binaria.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if num_labels <= 2:
        return binaria
    alturas = [int(stats[label, cv2.CC_STAT_HEIGHT]) for label in range(1, num_labels)]
    alturas_altas = [h for h in alturas if h >= alto * 0.25]
    altura_ref = float(np.median(alturas_altas)) if alturas_altas else float(np.median(alturas))
    area_min = max(12, int(alto * ancho * 0.0008))
    altura_min = max(4, int(altura_ref * 0.35))
    salida = np.zeros_like(binaria)
    conservados = 0
    for label in range(1, num_labels):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        # mota pequena (poca area y baja altura) -> ruido
        if area < area_min and h < altura_min:
            continue
        # fragmento muy bajo y angosto -> ruido
        if h < altura_min and w < ancho * 0.02:
            continue
        salida[labels == label] = 255
        conservados += 1
    if conservados == 0 or cv2.countNonZero(salida) < cv2.countNonZero(binaria) * 0.15:
        return binaria
    return salida

def _quitar_marco_rectangular(binaria: np.ndarray) -> np.ndarray:
    # FIX-MARCO: eliminar el borde/marco rectangular de la placa. Un marco es un
    # componente que cubre casi todo el bbox (ancho y alto) pero con baja densidad
    # de relleno (rectangulo hueco) y que toca los bordes de la imagen. Los
    # caracteres son bloques solidos (alta densidad), por lo que no se eliminan.
    if binaria is None or binaria.size == 0:
        return binaria
    alto, ancho = binaria.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if num_labels <= 1:
        return binaria
    salida = binaria.copy()
    elimino = False
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cubre_bbox = w >= ancho * 0.80 and h >= alto * 0.55
        densidad = area / max(w * h, 1)
        toca_borde = x <= 2 or y <= 2 or (x + w) >= ancho - 2 or (y + h) >= alto - 2
        if cubre_bbox and densidad < 0.35 and toca_borde:
            salida[labels == label] = 0
            elimino = True
    if not elimino or cv2.countNonZero(salida) < cv2.countNonZero(binaria) * 0.10:
        return binaria
    return salida


def _asegurar_caracteres_blancos(imagen) -> np.ndarray:
    gris = asegurar_grayscale(imagen)
    if len(np.unique(gris)) > 2:
        _, gris = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    invertida = cv2.bitwise_not(gris)
    return invertida if _score_orientacion_caracteres(invertida) > _score_orientacion_caracteres(gris) else gris

def _score_orientacion_caracteres(binaria) -> float:
    alto, ancho = binaria.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    score = 0.0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(4, ancho * alto * 0.001):
            continue
        aspect = w / max(h, 1)
        if h >= alto * 0.22 and ancho * 0.01 <= w <= ancho * 0.22 and 0.10 <= aspect <= 1.25:
            score += 2.0
        elif h >= alto * 0.12 and area >= ancho * alto * 0.002:
            score += 0.5
    return score


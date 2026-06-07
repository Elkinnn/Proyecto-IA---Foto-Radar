"""Rectificacion geometrica de recortes de placa para el lector CNN."""

import cv2
import numpy as np

from src.recognition.image_utils import asegurar_bgr, asegurar_grayscale
from src.recognition.preprocessing_filters import (
    _asegurar_caracteres_blancos,
    _denoise_componentes_placa,
)


def _ordenar_puntos_cuadrilatero(puntos: np.ndarray) -> np.ndarray:
    puntos = puntos.reshape(4, 2).astype("float32")
    suma = puntos.sum(axis=1)
    diff = np.diff(puntos, axis=1).ravel()
    ordenados = np.zeros((4, 2), dtype="float32")
    ordenados[0] = puntos[np.argmin(suma)]
    ordenados[2] = puntos[np.argmax(suma)]
    ordenados[1] = puntos[np.argmin(diff)]
    ordenados[3] = puntos[np.argmax(diff)]
    return ordenados

def _warp_perspectiva_placa(imagen_bgr: np.ndarray, puntos: np.ndarray) -> np.ndarray | None:
    rect = _ordenar_puntos_cuadrilatero(puntos)
    tl, tr, br, bl = rect
    ancho_sup = np.linalg.norm(tr - tl)
    ancho_inf = np.linalg.norm(br - bl)
    alto_der = np.linalg.norm(br - tr)
    alto_izq = np.linalg.norm(bl - tl)
    ancho = int(max(ancho_sup, ancho_inf))
    alto = int(max(alto_der, alto_izq))
    if ancho < 40 or alto < 14:
        return None
    if ancho / max(alto, 1) < 1.4:
        return None
    destino = np.array([[0, 0], [ancho - 1, 0], [ancho - 1, alto - 1], [0, alto - 1]], dtype="float32")
    matriz = cv2.getPerspectiveTransform(rect, destino)
    return cv2.warpPerspective(imagen_bgr, matriz, (ancho, alto), borderMode=cv2.BORDER_REPLICATE)

def _candidato_perspectiva_4_puntos(imagen_bgr: np.ndarray) -> np.ndarray | None:
    gris = asegurar_grayscale(imagen_bgr)
    gris = cv2.bilateralFilter(gris, 5, 45, 45)
    bordes = cv2.Canny(gris, 60, 180)
    bordes = cv2.dilate(bordes, np.ones((2, 2), np.uint8), iterations=1)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alto, ancho = gris.shape[:2]
    area_total = alto * ancho
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contorno)
        if area < area_total * 0.18:
            continue
        perimetro = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.025 * perimetro, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / max(h, 1)
        if 1.5 <= aspect <= 7.5:
            warp = _warp_perspectiva_placa(imagen_bgr, approx)
            if warp is not None:
                return warp
    return None

def _rotar_imagen_sin_cortar(imagen_bgr: np.ndarray, angulo: float) -> np.ndarray:
    alto, ancho = imagen_bgr.shape[:2]
    centro = (ancho / 2, alto / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos = abs(matriz[0, 0])
    sin = abs(matriz[0, 1])
    nuevo_ancho = int((alto * sin) + (ancho * cos))
    nuevo_alto = int((alto * cos) + (ancho * sin))
    matriz[0, 2] += (nuevo_ancho / 2) - centro[0]
    matriz[1, 2] += (nuevo_alto / 2) - centro[1]
    return cv2.warpAffine(imagen_bgr, matriz, (nuevo_ancho, nuevo_alto), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _estimar_angulo_deskew_proyeccion(binaria: np.ndarray, rango: float = 18.0, paso: float = 1.0) -> float:
    # FIX-DESKEW: estima el angulo maximizando la varianza del perfil de proyeccion
    # horizontal. Cuando la fila de caracteres queda recta, las filas con tinta
    # forman picos marcados (alta varianza). Es mucho mas estable que minAreaRect
    # ante ruido en placas borrosas o diagonales.
    alto, ancho = binaria.shape[:2]
    if alto < 8 or ancho < 8:
        return 0.0
    centro = (ancho / 2.0, alto / 2.0)
    mejor_ang = 0.0
    mejor_score = -1.0
    ang = -rango
    while ang <= rango + 1e-6:
        matriz = cv2.getRotationMatrix2D(centro, ang, 1.0)
        rotada = cv2.warpAffine(
            binaria,
            matriz,
            (ancho, alto),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        proyeccion = np.sum(rotada > 0, axis=1).astype(np.float32)
        score = float(np.var(proyeccion))
        if score > mejor_score:
            mejor_score = score
            mejor_ang = ang
        ang += paso
    return float(mejor_ang)

def _mascara_caracteres_para_deskew(binaria: np.ndarray) -> np.ndarray:
    # FIX-DESKEW: conservar solo componentes con forma de caracter para estimar el
    # angulo. Asi el marco y el fondo del recorte no desvian el deskew en tomas
    # diagonales.
    alto, ancho = binaria.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    mascara = np.zeros_like(binaria)
    area_min = max(8, int(alto * ancho * 0.001))
    conservados = 0
    for label in range(1, num_labels):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < area_min:
            continue
        if h < alto * 0.18 or h > alto * 0.95:
            continue
        if w > ancho * 0.30:
            continue
        aspect = w / max(h, 1)
        if aspect > 2.0:
            continue
        mascara[labels == label] = 255
        conservados += 1
    if conservados >= 3:
        return mascara
    return binaria

def _estimar_angulo_skew_robusto(imagen_bgr: np.ndarray) -> float | None:
    # FIX-DESKEW: estima el angulo de inclinacion maximizando la varianza del
    # perfil de proyeccion horizontal SOBRE LA MASCARA DE CARACTERES (ignora marco
    # y fondo). Es el metodo mas fiable para una fila de caracteres. Se hace en dos
    # pasadas: la 2a re-mide el residual tras enderezar y corrige la subestimacion.
    gris = asegurar_grayscale(imagen_bgr)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binaria = _denoise_componentes_placa(_asegurar_caracteres_blancos(binaria))
    mascara = _mascara_caracteres_para_deskew(binaria)
    if int(cv2.countNonZero(mascara)) < 60:
        return None

    # Pasada 1: barrido amplio + refinamiento fino.
    a1 = _estimar_angulo_deskew_proyeccion(mascara, rango=18.0, paso=0.5)
    a1 = _estimar_angulo_deskew_proyeccion_local(mascara, a1)

    # Pasada 2: enderezar con a1 y medir el residual que aun queda.
    mascara_2 = _rotar_imagen_sin_cortar(mascara, a1)
    mascara_2 = (mascara_2 > 127).astype(np.uint8) * 255
    a2 = _estimar_angulo_deskew_proyeccion(mascara_2, rango=8.0, paso=0.5)
    a2 = _estimar_angulo_deskew_proyeccion_local(mascara_2, a2)

    total = float(a1 + a2)
    return total if abs(total) >= 1.5 else None

def _candidato_rotacion_angulo(imagen_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    gris = asegurar_grayscale(imagen_bgr)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # FIX-DESKEW: limpiar ruido antes de estimar el angulo.
    binaria = _denoise_componentes_placa(_asegurar_caracteres_blancos(binaria))
    puntos = cv2.findNonZero(binaria)
    if puntos is None or len(puntos) < 25:
        return None, 0.0

    # FIX-DESKEW: estimar el angulo solo sobre la masa de caracteres, ignorando
    # el marco y el fondo del recorte (clave para tomas diagonales).
    mascara_chars = _mascara_caracteres_para_deskew(binaria)

    # Estimacion robusta por perfil de proyeccion, refinada con paso fino.
    angulo = _estimar_angulo_deskew_proyeccion(mascara_chars, rango=18.0, paso=1.0)
    angulo = _estimar_angulo_deskew_proyeccion_local(mascara_chars, angulo)

    # Validacion cruzada con minAreaRect: si coinciden, se promedia para afinar.
    rect = cv2.minAreaRect(puntos)
    (w, h) = rect[1]
    if w > 0 and h > 0:
        ang_rect = rect[-1]
        if w < h:
            ang_rect += 90
        if abs(ang_rect) <= 25 and abs(ang_rect - angulo) <= 6.0:
            angulo = (angulo + float(ang_rect)) / 2.0

    if abs(angulo) > 25:
        return None, float(angulo)
    if abs(angulo) < 0.8:
        return None, float(angulo)
    return _rotar_imagen_sin_cortar(imagen_bgr, angulo), float(angulo)

def _estimar_angulo_deskew_proyeccion_local(binaria: np.ndarray, angulo_base: float) -> float:
    # Refina el angulo en una ventana estrecha con paso fino.
    inicio = angulo_base - 1.0
    fin = angulo_base + 1.0
    alto, ancho = binaria.shape[:2]
    if alto < 8 or ancho < 8:
        return float(angulo_base)
    centro = (ancho / 2.0, alto / 2.0)
    mejor_ang = float(angulo_base)
    mejor_score = -1.0
    ang = inicio
    while ang <= fin + 1e-6:
        matriz = cv2.getRotationMatrix2D(centro, ang, 1.0)
        rotada = cv2.warpAffine(
            binaria,
            matriz,
            (ancho, alto),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        proyeccion = np.sum(rotada > 0, axis=1).astype(np.float32)
        score = float(np.var(proyeccion))
        if score > mejor_score:
            mejor_score = score
            mejor_ang = ang
        ang += 0.25
    return float(mejor_ang)

def _mejorar_contraste_placa(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = asegurar_grayscale(imagen_bgr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mejorada = clahe.apply(gris)
    return asegurar_bgr(mejorada)


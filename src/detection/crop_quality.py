"""Evaluacion de calidad para seleccionar el mejor recorte de placa."""

from __future__ import annotations

import cv2

from src.recognition.image_utils import asegurar_grayscale


DEFAULT_PLATE_CROP_SELECTION = {
    "enabled": True,
    "buffer_frames": 25,
    "min_aspect_ratio": 1.5,
    "max_aspect_ratio": 6.5,
    "min_area_relative": 0.0003,
    "max_area_relative": 0.08,
    "border_margin_px": 5,
    "min_sharpness": 30.0,
    "cooldown_frames": 30,
    "min_score_improvement": 0.03,
}


def config_recorte_placa(config: dict | None) -> dict:
    salida = DEFAULT_PLATE_CROP_SELECTION.copy()
    if config:
        salida.update({k: v for k, v in config.items() if v is not None})
    salida["buffer_frames"] = max(int(salida.get("buffer_frames", 15)), 1)
    return salida


def calcular_puntaje_recorte_placa(frame, bbox: list[int], recorte, conf_yolo: float, config: dict | None = None) -> dict:
    cfg = config_recorte_placa(config)
    alto_frame, ancho_frame = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    ancho_bbox = max(x2 - x1, 1)
    alto_bbox = max(y2 - y1, 1)
    area_relativa = (ancho_bbox * alto_bbox) / max(ancho_frame * alto_frame, 1)
    aspect_ratio = ancho_bbox / alto_bbox
    cerca_borde = (
        x1 <= int(cfg["border_margin_px"])
        or y1 <= int(cfg["border_margin_px"])
        or x2 >= ancho_frame - int(cfg["border_margin_px"])
        or y2 >= alto_frame - int(cfg["border_margin_px"])
    )

    if recorte is None or recorte.size == 0:
        nitidez = 0.0
        contraste = 0.0
        alto_recorte = 0
        ancho_recorte = 0
    else:
        gris = asegurar_grayscale(recorte)
        nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
        contraste = float(gris.std())
        alto_recorte, ancho_recorte = recorte.shape[:2]

    centro_aspect = (float(cfg["min_aspect_ratio"]) + float(cfg["max_aspect_ratio"])) / 2
    rango_aspect = max((float(cfg["max_aspect_ratio"]) - float(cfg["min_aspect_ratio"])) / 2, 0.1)
    score_aspect = max(0.0, 1.0 - abs(aspect_ratio - centro_aspect) / rango_aspect)

    min_area = float(cfg["min_area_relative"])
    max_area = float(cfg["max_area_relative"])
    centro_area = (min_area + max_area) / 2
    rango_area = max((max_area - min_area) / 2, 0.0001)
    score_area = max(0.0, 1.0 - abs(area_relativa - centro_area) / rango_area)

    score_nitidez = min(nitidez / max(float(cfg["min_sharpness"]) * 4, 1.0), 1.0)
    score_contraste = min(contraste / 80.0, 1.0)
    score_tamano = min((ancho_recorte * alto_recorte) / (120 * 35), 1.0)
    penalizacion = 0.0
    if cerca_borde:
        penalizacion += 0.25
    if aspect_ratio < float(cfg["min_aspect_ratio"]) or aspect_ratio > float(cfg["max_aspect_ratio"]):
        penalizacion += 0.35
    if area_relativa < min_area or area_relativa > max_area:
        penalizacion += 0.30
    if nitidez < float(cfg["min_sharpness"]):
        penalizacion += 0.25
    if ancho_recorte < 50 or alto_recorte < 18:
        penalizacion += 0.25

    puntaje = (
        0.25 * float(conf_yolo)
        + 0.40 * score_nitidez
        + 0.15 * score_aspect
        + 0.10 * score_area
        + 0.05 * score_tamano
        + 0.05 * score_contraste
        - penalizacion
    )
    return {
        "puntaje_total": round(max(puntaje, 0.0), 4),
        "area_relativa": round(area_relativa, 6),
        "aspect_ratio": round(aspect_ratio, 4),
        "nitidez": round(nitidez, 4),
        "contraste": round(contraste, 4),
        "cerca_borde": bool(cerca_borde),
        "ancho_recorte": int(ancho_recorte),
        "alto_recorte": int(alto_recorte),
    }


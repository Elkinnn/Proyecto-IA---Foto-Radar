"""Utilidades robustas de imagen para OpenCV y Streamlit."""

from __future__ import annotations

import cv2
import numpy as np


def _imread_seguro(ruta, flags=cv2.IMREAD_COLOR):
    """Lee una imagen sin propagar errores por archivos vacios o bloqueados."""
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if datos is None or datos.size == 0:
        return None
    try:
        return cv2.imdecode(datos, flags)
    except cv2.error:
        return None
    except Exception:
        return None


def _convertir_uint8(imagen: np.ndarray) -> np.ndarray:
    if imagen.dtype == np.uint8:
        return imagen
    if np.issubdtype(imagen.dtype, np.floating):
        maximo = float(np.max(imagen)) if imagen.size else 0.0
        imagen = imagen * 255.0 if maximo <= 1.0 else imagen
    return np.clip(imagen, 0, 255).astype(np.uint8)


def describir_imagen(img) -> dict:
    if img is None:
        return {"shape": None, "dtype": None, "canales": None, "min": None, "max": None}
    try:
        imagen = np.asarray(img)
    except Exception:
        return {"shape": None, "dtype": str(type(img)), "canales": None, "min": None, "max": None}
    canales = 1 if imagen.ndim == 2 else int(imagen.shape[2]) if imagen.ndim == 3 else None
    return {
        "shape": list(imagen.shape),
        "dtype": str(imagen.dtype),
        "canales": canales,
        "min": float(np.min(imagen)) if imagen.size else None,
        "max": float(np.max(imagen)) if imagen.size else None,
    }


def asegurar_grayscale(img) -> np.ndarray:
    if img is None:
        raise ValueError("No se puede convertir a grayscale una imagen None.")
    es_pil = img.__class__.__module__.startswith("PIL.")
    try:
        imagen = _convertir_uint8(np.asarray(img))
    except Exception as exc:
        raise ValueError(f"No se pudo convertir la imagen a un arreglo numpy: {exc}") from exc
    if imagen.size == 0:
        raise ValueError("No se puede convertir a grayscale una imagen vacia.")
    if imagen.ndim == 2:
        return imagen.copy()
    if imagen.ndim == 3:
        canales = imagen.shape[2]
        if canales == 1:
            return imagen[:, :, 0].copy()
        if canales == 3:
            codigo = cv2.COLOR_RGB2GRAY if es_pil else cv2.COLOR_BGR2GRAY
            return cv2.cvtColor(imagen, codigo)
        if canales == 4:
            codigo = cv2.COLOR_RGBA2GRAY if es_pil else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(imagen, codigo)
    raise ValueError(f"Formato de imagen no soportado para grayscale: shape={imagen.shape}.")


def asegurar_bgr(img) -> np.ndarray:
    if img is None:
        raise ValueError("No se puede convertir a BGR una imagen None.")
    es_pil = img.__class__.__module__.startswith("PIL.")
    try:
        imagen = _convertir_uint8(np.asarray(img))
    except Exception as exc:
        raise ValueError(f"No se pudo convertir la imagen a un arreglo numpy: {exc}") from exc
    if imagen.size == 0:
        raise ValueError("No se puede convertir a BGR una imagen vacia.")
    if imagen.ndim == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    if imagen.ndim == 3:
        canales = imagen.shape[2]
        if canales == 1:
            return cv2.cvtColor(imagen[:, :, 0], cv2.COLOR_GRAY2BGR)
        if canales == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR) if es_pil else imagen.copy()
        if canales == 4:
            codigo = cv2.COLOR_RGBA2BGR if es_pil else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(imagen, codigo)
    raise ValueError(f"Formato de imagen no soportado para BGR: shape={imagen.shape}.")


def asegurar_rgb(img) -> np.ndarray:
    return cv2.cvtColor(asegurar_bgr(img), cv2.COLOR_BGR2RGB)


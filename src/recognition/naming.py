"""Nombres seguros y huellas para artefactos del lector CNN."""

from datetime import datetime
from pathlib import Path
import hashlib
import re


def _nombre_seguro(ruta_imagen: str, nombre_base: str) -> str:
    stem = Path(ruta_imagen).stem if ruta_imagen else nombre_base
    texto = re.sub(r"[^A-Za-z0-9_-]+", "_", stem or nombre_base)
    return texto[:80]

def _huella_archivo_imagen(ruta_imagen: str, longitud: int = 12) -> str:
    """Identifica el contenido, no solo el nombre reutilizable del evento."""
    digest = hashlib.sha256()
    try:
        with open(ruta_imagen, "rb") as archivo:
            for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
                digest.update(bloque)
        return digest.hexdigest()[:longitud]
    except OSError:
        respaldo = f"{ruta_imagen}|{datetime.now().isoformat(timespec='microseconds')}"
        return hashlib.sha256(respaldo.encode("utf-8")).hexdigest()[:longitud]

def _nombre_lectura_unica(ruta_imagen: str, nombre_base: str = "placa") -> str:
    nombre = _nombre_seguro(ruta_imagen, nombre_base)
    huella = _huella_archivo_imagen(ruta_imagen)
    return f"{nombre[:67]}_{huella}"


"""Contexto de ejecucion del monitoreo y snapshots inmutables para CNN."""

from datetime import datetime
from pathlib import Path
import hashlib

import streamlit as st


OCR_ENTRADAS_INMUTABLES_DIR = (
    Path("reports") / "evidencias" / "reconocimiento_caracteres" / "entradas_monitoreo"
)


def nuevo_id_ejecucion_monitoreo() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def id_ejecucion_monitoreo_actual() -> str:
    return str(st.session_state.get("monitoreo_run_id") or "sin_ejecucion")


def crear_snapshot_ocr_inmutable(
    ruta_recorte: str,
    *,
    evento_id: int | None = None,
    frame_actual: int | None = None,
) -> str:
    """Copia el recorte antes del hilo CNN para que nunca cambie bajo sus pies."""
    origen = Path(str(ruta_recorte))
    try:
        datos = origen.read_bytes()
    except OSError:
        return str(origen)
    if not datos:
        return str(origen)

    huella = hashlib.sha256(datos).hexdigest()[:16]
    run_id = "".join(c for c in id_ejecucion_monitoreo_actual() if c.isalnum() or c in "_-")
    evento = max(int(evento_id or 0), 0)
    frame = max(int(frame_actual or 0), 0)
    sufijo = origen.suffix.lower() if origen.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else ".jpg"
    carpeta = OCR_ENTRADAS_INMUTABLES_DIR / run_id
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"evento_{evento:04d}_frame_{frame:06d}_{huella}{sufijo}"
    if not destino.exists():
        # El nombre ya es inmutable por contenido. Escribir directamente evita
        # fallos de os.replace en Windows/OneDrive cuando el antivirus inspecciona
        # el archivo temporal justo al crearlo.
        destino.write_bytes(datos)
    return str(destino)

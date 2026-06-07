"""Compatibilidad para el lector CNN de caracteres de placa.

La implementacion real vive en `src.recognition.reader_pipeline`, organizada
dentro del paquete de reconocimiento. Este archivo se conserva para que `app.py`
y los scripts existentes sigan importando `src.plate_reader` sin cambios.
"""

from src.recognition import reader_pipeline as _reader_pipeline

globals().update(
    {
        nombre: getattr(_reader_pipeline, nombre)
        for nombre in dir(_reader_pipeline)
        if not nombre.startswith("__")
    }
)

__all__ = [nombre for nombre in globals() if not nombre.startswith("_")]


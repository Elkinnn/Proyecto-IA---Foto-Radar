"""Carga cacheada de la CNN propia de caracteres."""

from __future__ import annotations

from pathlib import Path
import json
import threading


MODELO_CARACTERES_PATH = Path("models") / "character_reader" / "character_cnn.keras"
MODELO_CARACTERES_ROBUSTO_PATH = Path("models") / "character_reader" / "character_cnn_robusto.keras"
MODELO_CARACTERES_FINETUNED_PATH = Path("models") / "character_reader" / "character_cnn_finetuned.keras"
CLASS_NAMES_PATH = Path("models") / "character_reader" / "class_names.json"
_MODELO_CARACTERES_LOCK = threading.RLock()
_MODELO_CARACTERES_CACHE = None
_MODELO_CARACTERES_CACHE_KEY = None


def resolver_ruta_modelo_caracteres() -> Path | None:
    """Prioridad: fine-tune con reales > robusto > modelo base."""
    for ruta in (MODELO_CARACTERES_FINETUNED_PATH, MODELO_CARACTERES_ROBUSTO_PATH, MODELO_CARACTERES_PATH):
        if ruta.exists():
            return ruta
    return None


def cargar_modelo_caracteres():
    global _MODELO_CARACTERES_CACHE, _MODELO_CARACTERES_CACHE_KEY

    MODELO_CARACTERES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ruta_modelo = resolver_ruta_modelo_caracteres()
    if ruta_modelo is None:
        return None, "No existe un modelo propio de caracteres. Entrene primero el clasificador de caracteres."
    if not CLASS_NAMES_PATH.exists():
        return None, "No existe class_names.json para interpretar las salidas del modelo de caracteres."

    cache_key = (
        str(ruta_modelo.resolve()),
        int(ruta_modelo.stat().st_mtime_ns),
        int(CLASS_NAMES_PATH.stat().st_mtime_ns),
    )
    with _MODELO_CARACTERES_LOCK:
        if _MODELO_CARACTERES_CACHE is not None and _MODELO_CARACTERES_CACHE_KEY == cache_key:
            return _MODELO_CARACTERES_CACHE, f"Modelo de caracteres reutilizado: {ruta_modelo.name}."
        try:
            import tensorflow as tf

            modelo = tf.keras.models.load_model(ruta_modelo)
        except ImportError:
            try:
                import keras

                modelo = keras.models.load_model(ruta_modelo)
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

        _MODELO_CARACTERES_CACHE = (modelo, class_names)
        _MODELO_CARACTERES_CACHE_KEY = cache_key
        return _MODELO_CARACTERES_CACHE, f"Modelo de caracteres cargado: {ruta_modelo.name}."


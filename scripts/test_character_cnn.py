from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "character_reader" / "character_cnn.keras"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "character_reader" / "class_names.json"
TEST_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador_split" / "test"
SEED = 42
MAX_EJEMPLOS = 20


def importar_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow no esta instalado en este entorno. No se puede probar la CNN.") from exc
    return tf


def cargar_imagen(ruta: Path) -> np.ndarray | None:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return None
    imagen = cv2.resize(imagen, (32, 32), interpolation=cv2.INTER_AREA)
    imagen = imagen.astype("float32") / 255.0
    return imagen.reshape(1, 32, 32, 1)


def listar_imagenes_test() -> list[tuple[Path, str]]:
    ejemplos = []
    for clase_dir in sorted(TEST_DIR.iterdir()) if TEST_DIR.exists() else []:
        if not clase_dir.is_dir():
            continue
        for ruta in sorted(clase_dir.glob("*.png")):
            ejemplos.append((ruta, clase_dir.name))
    rng = random.Random(SEED)
    rng.shuffle(ejemplos)
    return ejemplos[:MAX_EJEMPLOS]


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"No existe modelo entrenado: {MODEL_PATH.relative_to(ROOT_DIR)}")
        sys.exit(1)
    if not CLASS_NAMES_PATH.exists():
        print(f"No existe class_names.json: {CLASS_NAMES_PATH.relative_to(ROOT_DIR)}")
        sys.exit(1)

    try:
        tf = importar_tensorflow()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    modelo = tf.keras.models.load_model(MODEL_PATH)
    ejemplos = listar_imagenes_test()
    if not ejemplos:
        print("No hay imagenes en el split de test.")
        sys.exit(1)

    print("Prueba CNN caracteres OCR")
    print("------------------------")
    for ruta, etiqueta_real in ejemplos:
        imagen = cargar_imagen(ruta)
        if imagen is None:
            print(f"No se pudo cargar: {ruta}")
            continue
        pred = modelo.predict(imagen, verbose=0)[0]
        idx = int(np.argmax(pred))
        etiqueta_pred = class_names[idx]
        confianza = float(pred[idx])
        print(f"Ruta: {ruta.relative_to(ROOT_DIR)}")
        print(f"Real: {etiqueta_real} | Prediccion: {etiqueta_pred} | Confianza: {confianza:.4f}")
        print("-" * 72)


if __name__ == "__main__":
    main()

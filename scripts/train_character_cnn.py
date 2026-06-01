from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador_split"
TRAIN_DIR = SPLIT_DIR / "train"
VALID_DIR = SPLIT_DIR / "valid"
TEST_DIR = SPLIT_DIR / "test"
MODEL_DIR = ROOT_DIR / "models" / "character_reader"
MODEL_PATH = MODEL_DIR / "character_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
REPORT_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "character_cnn"
HISTORY_CSV = REPORT_DIR / "historial_entrenamiento.csv"
METRICS_JSON = REPORT_DIR / "metricas_test.json"
CONFUSION_PNG = REPORT_DIR / "matriz_confusion.png"
TRAIN_REPORT = REPORT_DIR / "reporte_entrenamiento.txt"
CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
IMG_SIZE = (32, 32)


def importar_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no esta instalado en este entorno. Instale TensorFlow compatible con Python 3.11 antes de entrenar."
        ) from exc
    return tf


def contar_por_clase() -> dict:
    conteos = {}
    for clase in CLASES:
        carpeta = TRAIN_DIR / clase
        conteos[clase] = len(list(carpeta.glob("*.png"))) if carpeta.exists() else 0
    return conteos


def advertencias_desbalance(conteos: dict) -> list[str]:
    valores = [valor for valor in conteos.values() if valor > 0]
    if not valores:
        return ["No hay imagenes de entrenamiento."]
    promedio = sum(valores) / len(valores)
    advertencias = []
    for clase in ["O", "Z"]:
        cantidad = conteos.get(clase, 0)
        if cantidad < promedio * 0.60:
            advertencias.append(f"Advertencia: clase {clase} tiene menos muestras que el promedio ({cantidad} vs {promedio:.1f}).")
    return advertencias


def crear_datasets(tf):
    kwargs = {
        "image_size": IMG_SIZE,
        "color_mode": "grayscale",
        "label_mode": "int",
        "batch_size": BATCH_SIZE,
        "class_names": CLASES,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, shuffle=True, seed=SEED, **kwargs)
    valid_ds = tf.keras.utils.image_dataset_from_directory(VALID_DIR, shuffle=False, **kwargs)
    test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, shuffle=False, **kwargs)

    normalizador = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalizador(x), y)).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x, y: (normalizador(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalizador(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def crear_modelo(tf):
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(len(CLASES), activation="softmax"),
        ]
    )
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def guardar_historial(historial) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    claves = list(historial.history.keys())
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["epoch", *claves])
        for idx in range(len(historial.history[claves[0]])):
            writer.writerow([idx + 1, *[historial.history[clave][idx] for clave in claves]])


def predecir_test(modelo, test_ds) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for imagenes, etiquetas in test_ds:
        predicciones = modelo.predict(imagenes, verbose=0)
        y_true.extend(etiquetas.numpy().tolist())
        y_pred.extend(np.argmax(predicciones, axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matriz = np.zeros((len(CLASES), len(CLASES)), dtype=int)
    for real, pred in zip(y_true, y_pred):
        matriz[int(real), int(pred)] += 1
    return matriz


def guardar_matriz_confusion(matriz: np.ndarray) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, rotation=90)
    ax.set_yticklabels(CLASES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion OCR caracteres")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=160)
    plt.close(fig)


def reporte_por_clase(matriz: np.ndarray) -> list[dict]:
    reporte = []
    for idx, clase in enumerate(CLASES):
        tp = matriz[idx, idx]
        total_real = matriz[idx, :].sum()
        total_pred = matriz[:, idx].sum()
        recall = float(tp / total_real) if total_real else 0.0
        precision = float(tp / total_pred) if total_pred else 0.0
        reporte.append(
            {
                "clase": clase,
                "precision": precision,
                "recall": recall,
                "soporte": int(total_real),
            }
        )
    return reporte


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    try:
        tf = importar_tensorflow()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    tf.keras.utils.set_random_seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    conteos = contar_por_clase()
    advertencias = advertencias_desbalance(conteos)
    for advertencia in advertencias:
        print(advertencia)

    train_ds, valid_ds, test_ds = crear_datasets(tf)
    modelo = crear_modelo(tf)
    historial = modelo.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, batch_size=BATCH_SIZE)

    guardar_historial(historial)
    loss_test, accuracy_test = modelo.evaluate(test_ds, verbose=0)
    y_true, y_pred = predecir_test(modelo, test_ds)
    matriz = matriz_confusion(y_true, y_pred)
    guardar_matriz_confusion(matriz)
    reporte_clases = reporte_por_clase(matriz)

    modelo.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(CLASES, ensure_ascii=False, indent=2), encoding="utf-8")
    METRICS_JSON.write_text(
        json.dumps(
            {
                "loss_test": float(loss_test),
                "accuracy_test": float(accuracy_test),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "reporte_por_clase": reporte_clases,
                "advertencias": advertencias,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lineas = [
        "Entrenamiento CNN caracteres OCR",
        "--------------------------------",
        "Modelo entrenado desde cero, sin pesos preentrenados.",
        f"Modelo: {MODEL_PATH.relative_to(ROOT_DIR)}",
        f"Class names: {CLASS_NAMES_PATH.relative_to(ROOT_DIR)}",
        f"Accuracy test: {accuracy_test:.4f}",
        f"Loss test: {loss_test:.4f}",
        "",
        "Advertencias:",
        *(f"- {item}" for item in advertencias),
        "",
        "Reporte por clase:",
        *(f"- {item['clase']}: precision={item['precision']:.4f} recall={item['recall']:.4f} soporte={item['soporte']}" for item in reporte_clases),
    ]
    TRAIN_REPORT.write_text("\n".join(lineas), encoding="utf-8")

    print(f"Modelo guardado en: {MODEL_PATH.relative_to(ROOT_DIR)}")
    print(f"Accuracy test: {accuracy_test:.4f}")
    print(f"Loss test: {loss_test:.4f}")
    print(f"Reportes en: {REPORT_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

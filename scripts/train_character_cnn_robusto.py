"""Entrenamiento robusto de la CNN de caracteres.

Objetivo: cerrar la brecha de dominio entre el dataset (fuente gruesa/estilizada,
posible fuga de datos) y los caracteres reales segmentados en produccion (mas
finos, desplazados, borrosos, con ruido). Para ello se aplica augmentation fuerte
en el grafo (solo en entrenamiento):

  - Rotacion / zoom / traslacion (placas inclinadas y segmentacion imperfecta).
  - Jitter de GROSOR de trazo via dilatacion/erosion morfologica (maxpool/minpool)
    para que el modelo vea trazos finos y gruesos del mismo caracter.
  - Desenfoque gaussiano leve y ruido (capturas de video).

El modelo se guarda en un archivo SEPARADO (character_cnn_robusto.keras) para no
pisar el modelo en uso hasta validarlo.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador_split"
TRAIN_DIR = SPLIT_DIR / "train"
VALID_DIR = SPLIT_DIR / "valid"
TEST_DIR = SPLIT_DIR / "test"
MODEL_DIR = ROOT_DIR / "models" / "character_reader"
MODEL_PATH = MODEL_DIR / "character_cnn_robusto.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
IMG_SIZE = (32, 32)


def importar_tf():
    import tensorflow as tf
    return tf


def _augmentar_lote(tf, geom):
    # Augmentation aplicado SOLO en el pipeline de entrenamiento (no queda en el
    # modelo, por lo que la inferencia es limpia). x: (B,32,32,1) en [0,1].
    def fn(x, y):
        x = geom(x, training=True)  # rotacion/zoom/traslacion
        # Jitter de grosor de trazo: dilatar (engrosar) o erosionar (adelgazar).
        r = tf.random.uniform([])
        x = tf.cond(
            r < 0.33,
            lambda: tf.nn.max_pool2d(x, ksize=3, strides=1, padding="SAME"),
            lambda: tf.cond(
                r > 0.66,
                lambda: -tf.nn.max_pool2d(-x, ksize=3, strides=1, padding="SAME"),
                lambda: x,
            ),
        )
        # Desenfoque gaussiano leve aleatorio (capturas de video).
        def difuminar():
            k = tf.constant([1.0, 2.0, 1.0]) / 4.0
            kh = tf.reshape(k, [1, 3, 1, 1])
            kv = tf.reshape(k, [3, 1, 1, 1])
            xb = tf.nn.conv2d(x, kh, strides=1, padding="SAME")
            return tf.nn.conv2d(xb, kv, strides=1, padding="SAME")
        x = tf.cond(tf.random.uniform([]) < 0.5, difuminar, lambda: x)
        # Ruido gaussiano leve.
        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.06)
        return tf.clip_by_value(x, 0.0, 1.0), y

    return fn


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
    norm = tf.keras.layers.Rescaling(1.0 / 255)

    geom = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03, fill_mode="constant"),
            tf.keras.layers.RandomZoom(0.12, fill_mode="constant"),
            tf.keras.layers.RandomTranslation(0.10, 0.10, fill_mode="constant"),
        ],
        name="geom_aug",
    )
    aug_fn = _augmentar_lote(tf, geom)

    train_ds = (
        train_ds.map(lambda x, y: (norm(x), y))
        .map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = valid_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def crear_modelo(tf):
    layers = tf.keras.layers
    modelo = tf.keras.Sequential(
        [
            layers.Input(shape=(32, 32, 1)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(len(CLASES), activation="softmax"),
        ]
    )
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def main() -> None:
    tf = importar_tf()
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, valid_ds, test_ds = crear_datasets(tf)
    modelo = crear_modelo(tf)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    modelo.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=callbacks)

    loss, acc = modelo.evaluate(test_ds, verbose=0)
    print(f"ACC test (con leakage posible): {acc:.4f}  loss {loss:.4f}")

    modelo.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(CLASES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo robusto guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()

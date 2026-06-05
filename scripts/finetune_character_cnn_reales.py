"""Fine-tune de la CNN con caracteres reales cosechados + dataset original."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "datasets" / "caracteres_ecuador_split"
REALES_DIR = ROOT / "datasets" / "caracteres_reales" / "train"
MODEL_DIR = ROOT / "models" / "character_reader"
BASE_MODEL = MODEL_DIR / "character_cnn_robusto.keras"
FALLBACK_MODEL = MODEL_DIR / "character_cnn.keras"
OUT_MODEL = MODEL_DIR / "character_cnn_finetuned.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SEED = 42
EPOCHS = 15
BATCH_SIZE = 64
IMG_SIZE = (32, 32)


def importar_tf():
    import tensorflow as tf
    return tf


def _augmentar_lote(tf, geom):
    def fn(x, y):
        x = geom(x, training=True)
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
        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.04)
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
    train_syn = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "train", shuffle=True, seed=SEED, **kwargs
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "valid", shuffle=False, **kwargs
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "test", shuffle=False, **kwargs
    )
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    geom = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.02, fill_mode="constant"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="constant"),
            tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
        ]
    )
    aug_fn = _augmentar_lote(tf, geom)

    train_syn = (
        train_syn.map(lambda x, y: (norm(x), y))
        .map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    )

    # Repetir reales 3x para darles mas peso en el entrenamiento
    if REALES_DIR.exists() and any(p.is_dir() for p in REALES_DIR.iterdir()):
        subdirs = sorted(p.name for p in REALES_DIR.iterdir() if p.is_dir())
        local_to_global = tf.constant([CLASES.index(c) for c in subdirs], dtype=tf.int32)
        train_real = tf.keras.utils.image_dataset_from_directory(
            REALES_DIR,
            shuffle=True,
            seed=SEED,
            image_size=IMG_SIZE,
            color_mode="grayscale",
            label_mode="int",
            batch_size=BATCH_SIZE,
        )
        train_real = train_real.map(
            lambda x, y: (norm(x), tf.gather(local_to_global, y)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_real = train_real.repeat(3)
        train_ds = train_syn.concatenate(train_real).shuffle(5000, seed=SEED)
    else:
        train_ds = train_syn

    valid_ds = valid_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def main() -> None:
    tf = importar_tf()
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    n_reales = sum(1 for _ in REALES_DIR.rglob("*.jpg")) if REALES_DIR.exists() else 0
    if n_reales < 20:
        print(f"AVISO: solo {n_reales} caracteres reales. Recomendado >= 50 tras revision.")

    base = BASE_MODEL if BASE_MODEL.exists() else FALLBACK_MODEL
    if not base.exists():
        print("No hay modelo base. Entrene train_character_cnn_robusto.py primero.")
        sys.exit(1)

    print(f"Base: {base.name}  |  Reales en train/: {n_reales}")
    modelo = tf.keras.models.load_model(base)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds, valid_ds, test_ds = crear_datasets(tf)
    modelo.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ],
    )

    loss, acc = modelo.evaluate(test_ds, verbose=0)
    print(f"ACC test split: {acc:.4f}  loss {loss:.4f}")

    modelo.save(OUT_MODEL)
    CLASS_NAMES_PATH.write_text(json.dumps(CLASES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo fine-tuneado: {OUT_MODEL}")
    print("El lector lo usara automaticamente (finetuned > robusto > base).")


if __name__ == "__main__":
    main()

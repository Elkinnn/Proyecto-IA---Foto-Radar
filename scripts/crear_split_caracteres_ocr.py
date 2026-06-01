from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import shutil
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador"
SPLIT_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador_split"
SUMMARY_CSV = SPLIT_DIR / "split_summary.csv"
CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}
SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea split train/valid/test para OCR de caracteres.")
    parser.add_argument("--reset", action="store_true", help="Borra y recrea datasets/caracteres_ecuador_split.")
    return parser.parse_args()


def preparar_destino(reset: bool) -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No existe dataset fuente: {DATASET_DIR}")

    if SPLIT_DIR.exists() and reset:
        destino = SPLIT_DIR.resolve()
        datasets_dir = (ROOT_DIR / "datasets").resolve()
        if datasets_dir not in destino.parents:
            raise RuntimeError(f"Ruta insegura para borrar: {destino}")
        shutil.rmtree(SPLIT_DIR)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        for clase in CLASES:
            (SPLIT_DIR / split / clase).mkdir(parents=True, exist_ok=True)


def listar_imagenes(clase: str) -> list[Path]:
    clase_dir = DATASET_DIR / clase
    if not clase_dir.exists():
        return []
    return sorted(
        ruta
        for ruta in clase_dir.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
    )


def copiar_split(clase: str, imagenes: list[Path]) -> tuple[int, int, int]:
    rng = random.Random(SEED)
    imagenes = list(imagenes)
    rng.shuffle(imagenes)

    total = len(imagenes)
    n_train = int(total * TRAIN_RATIO)
    n_valid = int(total * VALID_RATIO)
    n_test = total - n_train - n_valid

    grupos = {
        "train": imagenes[:n_train],
        "valid": imagenes[n_train : n_train + n_valid],
        "test": imagenes[n_train + n_valid :],
    }

    for split, rutas in grupos.items():
        destino_clase = SPLIT_DIR / split / clase
        for ruta in rutas:
            shutil.copy2(ruta, destino_clase / ruta.name)

    return len(grupos["train"]), len(grupos["valid"]), n_test


def crear_split() -> list[dict]:
    resumen = []
    for clase in CLASES:
        imagenes = listar_imagenes(clase)
        train, valid, test = copiar_split(clase, imagenes)
        resumen.append(
            {
                "clase": clase,
                "total": len(imagenes),
                "train": train,
                "valid": valid,
                "test": test,
            }
        )
        print(f"{clase}: total={len(imagenes)} train={train} valid={valid} test={test}")
    return resumen


def guardar_resumen(resumen: list[dict]) -> None:
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=["clase", "total", "train", "valid", "test"])
        writer.writeheader()
        writer.writerows(resumen)


def main() -> None:
    args = parse_args()
    try:
        preparar_destino(args.reset)
        resumen = crear_split()
        guardar_resumen(resumen)
    except Exception as exc:
        print(f"Error creando split OCR: {exc}")
        sys.exit(1)

    print("-" * 72)
    print(f"Split creado en: {SPLIT_DIR.relative_to(ROOT_DIR)}")
    print(f"Resumen: {SUMMARY_CSV.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

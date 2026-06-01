from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DESTINO_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador"
DESTINO_O_DIR = DESTINO_DIR / "O"
LABELS_CSV = DESTINO_DIR / "labels.csv"
TAMANO_FINAL = 32
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FUENTES = {
    "train": Path(r"D:\Documentos\Universidad\Septimo Semestre\INTELIGENCIA ARTIFICIAL\sistema_placas_ia\datasets\caracteres_cnn\train\O"),
    "valid": Path(r"D:\Documentos\Universidad\Septimo Semestre\INTELIGENCIA ARTIFICIAL\sistema_placas_ia\datasets\caracteres_cnn\val\O"),
    "test": Path(r"D:\Documentos\Universidad\Septimo Semestre\INTELIGENCIA ARTIFICIAL\sistema_placas_ia\datasets\caracteres_cnn\test\O"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa unicamente la clase O desde un proyecto anterior.")
    parser.add_argument(
        "--reset-o-importada",
        action="store_true",
        help="Borra solo archivos O_importada_* en datasets/caracteres_ecuador/O antes de importar.",
    )
    return parser.parse_args()


def normalizar_imagen(ruta: Path) -> tuple[np.ndarray | None, tuple[int, int]]:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return None, (0, 0)

    alto_original, ancho_original = imagen.shape[:2]
    imagen = cv2.GaussianBlur(imagen, (3, 3), 0)
    _, binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blancos = cv2.countNonZero(binaria)
    total = binaria.shape[0] * binaria.shape[1]
    if blancos > total * 0.5:
        binaria = cv2.bitwise_not(binaria)

    coords = cv2.findNonZero(binaria)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binaria = binaria[y : y + h, x : x + w]

    alto, ancho = binaria.shape[:2]
    escala = min((TAMANO_FINAL - 6) / max(ancho, 1), (TAMANO_FINAL - 6) / max(alto, 1))
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    redimensionada = cv2.resize(binaria, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((TAMANO_FINAL, TAMANO_FINAL), dtype=np.uint8)
    x0 = (TAMANO_FINAL - nuevo_ancho) // 2
    y0 = (TAMANO_FINAL - nuevo_alto) // 2
    lienzo[y0 : y0 + nuevo_alto, x0 : x0 + nuevo_ancho] = redimensionada
    return lienzo, (ancho_original, alto_original)


def listar_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        return []
    return sorted(ruta for ruta in carpeta.rglob("*") if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES)


def leer_labels() -> list[dict]:
    if not LABELS_CSV.exists():
        return []
    with open(LABELS_CSV, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def escribir_labels(filas: list[dict]) -> None:
    DESTINO_DIR.mkdir(parents=True, exist_ok=True)
    campos = [
        "ruta_imagen",
        "etiqueta",
        "origen",
        "ruta_original",
        "ancho_original",
        "alto_original",
        "ancho_final",
        "alto_final",
        "fecha_procesamiento",
    ]
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)


def reset_importadas(filas: list[dict]) -> list[dict]:
    DESTINO_O_DIR.mkdir(parents=True, exist_ok=True)
    for ruta in DESTINO_O_DIR.glob("O_importada_*"):
        if ruta.is_file():
            ruta.unlink()

    return [
        fila
        for fila in filas
        if not (
            fila.get("etiqueta") == "O"
            and fila.get("origen") == "importado_proyecto_anterior"
            and Path(fila.get("ruta_imagen", "")).name.startswith("O_importada_")
        )
    ]


def siguiente_indice_existente(split: str) -> int:
    existentes = sorted(DESTINO_O_DIR.glob(f"O_importada_{split}_*.png"))
    maximo = 0
    for ruta in existentes:
        try:
            maximo = max(maximo, int(ruta.stem.split("_")[-1]))
        except ValueError:
            continue
    return maximo + 1


def importar(reset_o_importada: bool) -> dict:
    DESTINO_O_DIR.mkdir(parents=True, exist_ok=True)
    filas = leer_labels()
    if reset_o_importada:
        filas = reset_importadas(filas)

    originales_importados = {
        fila.get("ruta_original")
        for fila in filas
        if fila.get("etiqueta") == "O" and fila.get("origen") == "importado_proyecto_anterior"
    }

    fecha = datetime.now().isoformat(timespec="seconds")
    resumen = {"train": 0, "valid": 0, "test": 0}

    for split, carpeta in FUENTES.items():
        if not carpeta.exists():
            print(f"Advertencia: no existe carpeta fuente {split}: {carpeta}")
            continue

        indice = siguiente_indice_existente(split)
        for ruta_original in listar_imagenes(carpeta):
            ruta_original_str = str(ruta_original)
            if ruta_original_str in originales_importados:
                continue

            imagen, (ancho_original, alto_original) = normalizar_imagen(ruta_original)
            if imagen is None:
                print(f"Advertencia: no se pudo cargar {ruta_original}")
                continue

            nombre = f"O_importada_{split}_{indice:06d}.png"
            ruta_destino = DESTINO_O_DIR / nombre
            while ruta_destino.exists():
                indice += 1
                nombre = f"O_importada_{split}_{indice:06d}.png"
                ruta_destino = DESTINO_O_DIR / nombre

            cv2.imwrite(str(ruta_destino), imagen)
            filas.append(
                {
                    "ruta_imagen": str(ruta_destino.relative_to(ROOT_DIR)),
                    "etiqueta": "O",
                    "origen": "importado_proyecto_anterior",
                    "ruta_original": ruta_original_str,
                    "ancho_original": ancho_original,
                    "alto_original": alto_original,
                    "ancho_final": TAMANO_FINAL,
                    "alto_final": TAMANO_FINAL,
                    "fecha_procesamiento": fecha,
                }
            )
            originales_importados.add(ruta_original_str)
            resumen[split] += 1
            indice += 1

    escribir_labels(filas)
    return resumen


def main() -> None:
    args = parse_args()
    try:
        resumen = importar(args.reset_o_importada)
    except Exception as exc:
        print(f"Error importando clase O: {exc}")
        sys.exit(1)

    total = sum(resumen.values())
    print("Clase O importada")
    print("-----------------")
    print(f"Train: {resumen['train']} imagenes")
    print(f"Valid: {resumen['valid']} imagenes")
    print(f"Test: {resumen['test']} imagenes")
    print(f"Total importadas: {total}")
    print(f"Destino: {DESTINO_O_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

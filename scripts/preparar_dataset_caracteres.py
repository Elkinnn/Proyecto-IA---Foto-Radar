from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
FUENTE_DIR = ROOT_DIR / "datasets" / "fuente_caracteres_externa"
DESTINO_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador"
LABELS_CSV = DESTINO_DIR / "labels.csv"
CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TAMANO_FINAL = 32
MAX_POR_CLASE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara dataset OCR de caracteres de placas.")
    parser.add_argument("--reset", action="store_true", help="Limpia datasets/caracteres_ecuador antes de preparar.")
    parser.add_argument(
        "--max-por-clase",
        type=int,
        default=MAX_POR_CLASE,
        help="Limita la cantidad de imagenes procesadas por clase.",
    )
    return parser.parse_args()


def validar_rutas(reset: bool) -> None:
    if not FUENTE_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset fuente: {FUENTE_DIR}")

    if DESTINO_DIR.exists() and any(DESTINO_DIR.iterdir()) and not reset:
        print(f"Advertencia: {DESTINO_DIR} ya contiene archivos.")
        print("No se borrara nada. Se agregaran/actualizaran archivos con nombres normalizados.")
        print("Use --reset si desea limpiar el destino antes de preparar.")

    if reset and DESTINO_DIR.exists():
        destino_resuelto = DESTINO_DIR.resolve()
        datasets_resuelto = (ROOT_DIR / "datasets").resolve()
        if datasets_resuelto not in destino_resuelto.parents:
            raise RuntimeError(f"Ruta de destino insegura para reset: {destino_resuelto}")
        shutil.rmtree(DESTINO_DIR)

    DESTINO_DIR.mkdir(parents=True, exist_ok=True)
    for clase in CLASES:
        (DESTINO_DIR / clase).mkdir(parents=True, exist_ok=True)


def listar_imagenes_clase(clase: str) -> list[Path]:
    clase_dir = FUENTE_DIR / clase
    if not clase_dir.exists():
        return []
    return sorted(
        ruta
        for ruta in clase_dir.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
    )


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


def preparar_dataset(max_por_clase: int | None) -> int:
    filas = []
    total = 0
    fecha = datetime.now().isoformat(timespec="seconds")

    for clase in CLASES:
        imagenes = listar_imagenes_clase(clase)
        if max_por_clase is not None:
            imagenes = imagenes[: max(max_por_clase, 0)]

        contador = 0
        for ruta_original in imagenes:
            normalizada, (ancho_original, alto_original) = normalizar_imagen(ruta_original)
            if normalizada is None:
                print(f"No se pudo cargar: {ruta_original}")
                continue

            contador += 1
            total += 1
            nombre = f"{clase}_{contador:06d}.png"
            ruta_destino = DESTINO_DIR / clase / nombre
            cv2.imwrite(str(ruta_destino), normalizada)
            filas.append(
                {
                    "ruta_imagen": str(ruta_destino.relative_to(ROOT_DIR)),
                    "etiqueta": clase,
                    "origen": "fuente_caracteres_externa",
                    "ruta_original": str(ruta_original.relative_to(ROOT_DIR)),
                    "ancho_original": ancho_original,
                    "alto_original": alto_original,
                    "ancho_final": TAMANO_FINAL,
                    "alto_final": TAMANO_FINAL,
                    "fecha_procesamiento": fecha,
                }
            )

        print(f"{clase}: {contador} imagenes procesadas")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as archivo:
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
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    return total


def main() -> None:
    args = parse_args()
    try:
        validar_rutas(args.reset)
        total = preparar_dataset(args.max_por_clase)
    except Exception as exc:
        print(f"Error preparando dataset de caracteres: {exc}")
        sys.exit(1)

    print("-" * 72)
    print(f"Dataset preparado en: {DESTINO_DIR.relative_to(ROOT_DIR)}")
    print(f"Total imagenes procesadas: {total}")
    print(f"Labels: {LABELS_CSV.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

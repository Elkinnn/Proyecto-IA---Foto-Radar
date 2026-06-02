from __future__ import annotations

from pathlib import Path
import csv
import random
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador"
REALES_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "caracteres_segmentados"
CSV_DIAGNOSTICO = ROOT_DIR / "datasets" / "diagnostico_caracteres_reales.csv"
SALIDA_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "comparacion_dataset_reales"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}
CLASES_INTERES = ["P", "B", "F", "6", "H", "M", "4", "2", "8", "O", "0", "I", "1", "S", "5", "Z"]
PARES_INTERES = [("P", "B"), ("F", "6"), ("H", "M"), ("4", "B"), ("4", "2"), ("B", "8"), ("O", "0")]
SEED = 42


def listar_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        return []
    return sorted(ruta for ruta in carpeta.rglob("*") if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES)


def seleccionar(rutas: list[Path], cantidad: int) -> list[Path]:
    rng = random.Random(SEED)
    copia = list(rutas)
    rng.shuffle(copia)
    return copia[:cantidad]


def reales_por_clase() -> dict[str, list[Path]]:
    agrupadas = {clase: [] for clase in CLASES_INTERES}
    if CSV_DIAGNOSTICO.exists():
        with open(CSV_DIAGNOSTICO, newline="", encoding="utf-8") as archivo:
            for fila in csv.DictReader(archivo):
                etiqueta = (fila.get("etiqueta_esperada") or "").strip().upper()
                ruta_txt = fila.get("ruta_caracter") or ""
                ruta = ROOT_DIR / ruta_txt
                if etiqueta in agrupadas and ruta.exists():
                    agrupadas[etiqueta].append(ruta)
        return agrupadas

    todas = listar_imagenes(REALES_DIR)
    for clase in agrupadas:
        agrupadas[clase] = todas
    return agrupadas


def preparar_miniatura(ruta: Path, tamano: int = 64) -> np.ndarray:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return np.zeros((tamano, tamano, 3), dtype=np.uint8)
    imagen = cv2.resize(imagen, (tamano, tamano), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)


def poner_titulo(lienzo: np.ndarray, texto: str, x: int, y: int) -> None:
    cv2.putText(lienzo, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def crear_grilla(clases: list[str], nombre_salida: str, reales: dict[str, list[Path]]) -> Path:
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    celda = 74
    columnas = 8
    filas_por_clase = 2
    alto_header = 34
    ancho = columnas * celda
    alto = alto_header + len(clases) * filas_por_clase * celda
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    poner_titulo(lienzo, f"Comparacion dataset vs reales: {' / '.join(clases)}", 8, 23)

    y = alto_header
    for clase in clases:
        dataset = seleccionar(listar_imagenes(DATASET_DIR / clase), columnas - 1)
        reales_clase = seleccionar(reales.get(clase, []), columnas - 1)

        poner_titulo(lienzo, f"{clase} dataset", 4, y + 22)
        for idx, ruta in enumerate(dataset, start=1):
            x = idx * celda
            lienzo[y : y + 64, x : x + 64] = preparar_miniatura(ruta)

        y += celda
        poner_titulo(lienzo, "reales", 4, y + 22)
        for idx, ruta in enumerate(reales_clase, start=1):
            x = idx * celda
            lienzo[y : y + 64, x : x + 64] = preparar_miniatura(ruta)
        y += celda

    ruta_salida = SALIDA_DIR / nombre_salida
    cv2.imwrite(str(ruta_salida), lienzo)
    return ruta_salida


def main() -> None:
    if not DATASET_DIR.exists():
        print(f"No existe dataset base: {DATASET_DIR.relative_to(ROOT_DIR)}")
        sys.exit(1)

    generadas = []
    reales = reales_por_clase()
    for clase in CLASES_INTERES:
        generadas.append(crear_grilla([clase], f"comparacion_{clase}.png", reales))
    for izquierda, derecha in PARES_INTERES:
        generadas.append(crear_grilla([izquierda, derecha], f"comparacion_{izquierda}_{derecha}.png", reales))

    print("Comparacion dataset vs reales")
    print("-----------------------------")
    print(f"Imagenes generadas: {len(generadas)}")
    print(f"Salida: {SALIDA_DIR.relative_to(ROOT_DIR)}")
    if not CSV_DIAGNOSTICO.exists() and (not REALES_DIR.exists() or not listar_imagenes(REALES_DIR)):
        print("Advertencia: no se encontraron caracteres reales segmentados; las filas reales quedaran vacias/negras.")
    elif not CSV_DIAGNOSTICO.exists():
        print("Advertencia: no existe CSV etiquetado; las filas reales son ejemplos segmentados generales, no por clase.")


if __name__ == "__main__":
    main()

"""Genera un montaje indexado de una carpeta de revision para inspeccion visual.

Uso: python scripts/montaje_revision.py 7
Crea reports/_bench/revision_montaje_<clase>.png con cada caracter numerado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = ROOT / "datasets" / "caracteres_reales" / "revision"
OUT_DIR = ROOT / "reports" / "_bench"


def main() -> None:
    clase = sys.argv[1] if len(sys.argv) > 1 else "7"
    carpeta = REVISION_DIR / clase
    if not carpeta.exists():
        print(f"No existe {carpeta}")
        return
    archivos = sorted(carpeta.glob("*.jpg"))
    if not archivos:
        print(f"Carpeta vacia: {carpeta}")
        return

    celda = 64
    pad = 18
    cols = 10
    filas = (len(archivos) + cols - 1) // cols
    W = cols * (celda + pad) + pad
    H = filas * (celda + pad + 14) + pad
    lienzo = np.full((H, W, 3), 40, np.uint8)

    indice = []
    for i, ruta in enumerate(archivos):
        r, c = divmod(i, cols)
        x = pad + c * (celda + pad)
        y = pad + r * (celda + pad + 14)
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (celda, celda), interpolation=cv2.INTER_NEAREST)
        lienzo[y : y + celda, x : x + celda] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(lienzo, str(i), (x, y + celda + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        indice.append(f"{i}: {ruta.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"revision_montaje_{clase}.png"
    cv2.imwrite(str(out), lienzo)
    (OUT_DIR / f"revision_indice_{clase}.txt").write_text("\n".join(indice), encoding="utf-8")
    print(f"Montaje: {out}  ({len(archivos)} caracteres)")


if __name__ == "__main__":
    main()

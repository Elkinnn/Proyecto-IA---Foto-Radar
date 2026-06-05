"""Aplica correcciones de alta confianza desde revision/ hacia train/.

Mapa explicito: por cada carpeta de revision, lista de (indice -> etiqueta correcta).
El indice corresponde al numero mostrado en reports/_bench/revision_montaje_<clase>.png
(orden alfabetico de los .jpg de esa carpeta).
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = ROOT / "datasets" / "caracteres_reales" / "revision"
TRAIN_DIR = ROOT / "datasets" / "caracteres_reales" / "train"

# carpeta_origen -> { etiqueta_correcta: [indices] }
CORRECCIONES = {
    "7": {
        "7": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 21, 22, 23, 24, 25, 31, 32, 33, 34],
    },
    "9": {
        "9": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    },
    "1": {
        "7": [16, 30, 31, 32, 40, 48],
        "1": [8, 9, 36, 43, 44, 46],
    },
}


def main() -> None:
    movidos = {}
    for carpeta, mapa in CORRECCIONES.items():
        origen = REVISION_DIR / carpeta
        if not origen.exists():
            print(f"(omitido) no existe {origen}")
            continue
        archivos = sorted(origen.glob("*.jpg"))
        for etiqueta, indices in mapa.items():
            dest_dir = TRAIN_DIR / etiqueta
            dest_dir.mkdir(parents=True, exist_ok=True)
            for idx in indices:
                if idx < 0 or idx >= len(archivos):
                    print(f"  indice fuera de rango {carpeta}#{idx}")
                    continue
                src = archivos[idx]
                if not src.exists():
                    continue
                dest = dest_dir / src.name
                if dest.exists():
                    dest = dest_dir / f"{src.stem}_rev{src.suffix}"
                shutil.move(str(src), str(dest))
                movidos[etiqueta] = movidos.get(etiqueta, 0) + 1

    print("Movidos a train/ por etiqueta:", movidos)
    n_train = sum(1 for _ in TRAIN_DIR.rglob("*.jpg"))
    n_rev = sum(1 for _ in REVISION_DIR.rglob("*.jpg"))
    print(f"Total train/: {n_train}  |  revision/: {n_rev}")


if __name__ == "__main__":
    main()

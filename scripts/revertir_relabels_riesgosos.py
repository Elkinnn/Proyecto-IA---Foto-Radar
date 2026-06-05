"""Revierte los relabels arriesgados (carpeta 1 -> 7) de vuelta a revision/1.

Identifica en train/7 los archivos cuyo campo de prediccion original era '1'
(nombre: <placa>_posNN_1_<conf>.jpg), es decir, los que se movieron desde
revision/1, y los regresa a revision/1.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_7 = ROOT / "datasets" / "caracteres_reales" / "train" / "7"
REVISION_1 = ROOT / "datasets" / "caracteres_reales" / "revision" / "1"

# nombre tipo: mejor_recorte_001527_pos05_1_0.50.jpg  -> pred field = "1"
PAT = re.compile(r"_pos\d+_([0-9A-Z])_[\d.]+(?:_rev)?\.jpg$", re.I)


def main() -> None:
    REVISION_1.mkdir(parents=True, exist_ok=True)
    revertidos = 0
    for img in sorted(TRAIN_7.glob("*.jpg")):
        m = PAT.search(img.name)
        if m and m.group(1) == "1":
            dest = REVISION_1 / img.name
            if dest.exists():
                dest = REVISION_1 / f"{img.stem}_back{img.suffix}"
            shutil.move(str(img), str(dest))
            revertidos += 1
    print(f"Revertidos train/7 -> revision/1: {revertidos}")
    n7 = sum(1 for _ in TRAIN_7.glob("*.jpg"))
    print(f"train/7 ahora: {n7}")


if __name__ == "__main__":
    main()

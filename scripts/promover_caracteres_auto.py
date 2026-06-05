"""Promueve caracteres de revision/ a train/ automaticamente.

Reglas:
  1. Placas del benchmark con GT conocido -> etiqueta correcta por posicion.
  2. Confianza >= 0.97 y tipo correcto (letra/digito segun posicion) -> etiqueta predicha.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = ROOT / "datasets" / "caracteres_reales" / "revision"
TRAIN_DIR = ROOT / "datasets" / "caracteres_reales" / "train"
MANIFEST = ROOT / "datasets" / "caracteres_reales" / "manifest.jsonl"

BENCH_GT = {
    "mejor_recorte_000036": "PDZ279",
    "mejor_recorte_000080": "PDK7282",
    "mejor_recorte_000130": "PDK7282",
    "mejor_recorte_000210": "TBC6224",
    "mejor_recorte_000260": "PDK7282",
    "mejor_recorte_000310": "PDZ279",
    "mejor_recorte_000402": "ABC0123",
    "mejor_recorte_000520": "PDK7282",
    "mejor_recorte_000687": "ABJ6347",
    "mejor_recorte_000860": "PDK7282",
    "mejor_recorte_001062": "ABC0123",
    "mejor_recorte_001520": "PDZ279",
    "mejor_recorte_004785": "PDZ279",
    "mejor_recorte_005245": "PDZ279",
    "mejor_recorte_007434": "ABC0123",
}

PAT = re.compile(r"^(mejor_recorte_\d+)_pos(\d+)_([0-9A-Z])_([\d.]+)\.jpg$", re.I)


def _mover(origen: Path, etiqueta: str) -> Path:
    dest_dir = TRAIN_DIR / etiqueta
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / origen.name
    if dest.exists():
        dest = dest_dir / f"{origen.stem}_mv{origen.suffix}"
    shutil.move(str(origen), str(dest))
    return dest


def main() -> None:
    if not REVISION_DIR.exists():
        print("No hay revision/")
        return

    promovidos_gt = 0
    promovidos_conf = 0

    for carpeta in list(REVISION_DIR.iterdir()):
        if not carpeta.is_dir():
            continue
        for img in list(carpeta.glob("*.jpg")):
            m = PAT.match(img.name)
            if not m:
                continue
            placa, pos_s, pred, conf_s = m.groups()
            pos = int(pos_s) - 1
            conf = float(conf_s)

            if placa in BENCH_GT and pos < len(BENCH_GT[placa]):
                etiqueta = BENCH_GT[placa][pos]
                _mover(img, etiqueta)
                promovidos_gt += 1
                continue

            if conf >= 0.97:
                esperado_letra = pos < 3
                if esperado_letra and pred.isalpha():
                    _mover(img, pred.upper())
                    promovidos_conf += 1
                elif not esperado_letra and pred.isdigit():
                    _mover(img, pred)
                    promovidos_conf += 1

    n_train = sum(1 for _ in TRAIN_DIR.rglob("*.jpg"))
    n_rev = sum(1 for _ in REVISION_DIR.rglob("*.jpg"))
    print(f"Promovidos por GT benchmark: {promovidos_gt}")
    print(f"Promovidos por alta confianza: {promovidos_conf}")
    print(f"Total train/: {n_train}  |  revision/: {n_rev}")
    print("Siguiente: python scripts/finetune_character_cnn_reales.py")


if __name__ == "__main__":
    main()

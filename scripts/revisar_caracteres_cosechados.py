"""Revision rapida de caracteres cosechados (teclado).

Uso:
  python scripts/revisar_caracteres_cosechados.py

Controles:
  0-9, A-Z  -> mover a train/{caracter}/
  Enter     -> confirmar etiqueta sugerida (nombre de carpeta actual)
  s / Espacio -> saltar
  b         -> volver al anterior
  q         -> salir
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = ROOT / "datasets" / "caracteres_reales" / "revision"
TRAIN_DIR = ROOT / "datasets" / "caracteres_reales" / "train"
CLASES = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _listar_pendientes() -> list[Path]:
    imgs = []
    if not REVISION_DIR.exists():
        return imgs
    for carpeta in sorted(REVISION_DIR.iterdir()):
        if carpeta.is_dir():
            imgs.extend(sorted(carpeta.glob("*.jpg")))
    return imgs


def _mover_a_train(origen: Path, etiqueta: str) -> Path:
    dest_dir = TRAIN_DIR / etiqueta
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / origen.name
    if dest.exists():
        dest = dest_dir / f"{origen.stem}_dup{origen.suffix}"
    shutil.move(str(origen), str(dest))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clase", type=str, default="", help="Solo revisar una carpeta (ej. 7, J)")
    args = parser.parse_args()

    pendientes = _listar_pendientes()
    if args.clase:
        pendientes = [p for p in pendientes if p.parent.name == args.clase.upper()]

    if not pendientes:
        print("No hay caracteres pendientes en revision/.")
        print("Ejecuta primero: python scripts/cosechar_caracteres_reales.py")
        return

    print(f"Revisar {len(pendientes)} caracteres. q=salir, Enter=confirmar carpeta, 0-9/A-Z=etiqueta")
    historial: list[tuple[Path, Path]] = []
    idx = 0

    while idx < len(pendientes):
        ruta = pendientes[idx]
        etiqueta_sug = ruta.parent.name
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            show = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("revision (Enter=ok, 0-9/A-Z=etiqueta, s=saltar, b=atras, q=salir)", show)

        print(f"[{idx+1}/{len(pendientes)}] {ruta.name}  sugerido={etiqueta_sug}")
        tecla = cv2.waitKey(0) & 0xFF

        if tecla in (ord("q"), 27):
            break
        if tecla in (ord("s"), ord(" ")):
            idx += 1
            continue
        if tecla == ord("b") and historial:
            prev_orig, prev_dest = historial.pop()
            if prev_dest.exists():
                prev_dest.rename(prev_orig)
            idx = max(0, idx - 1)
            continue

        if tecla in (13, 10):  # Enter
            nueva = etiqueta_sug if etiqueta_sug in CLASES else "?"
        elif 48 <= tecla <= 57:
            nueva = chr(tecla)
        elif 65 <= tecla <= 90:
            nueva = chr(tecla)
        elif 97 <= tecla <= 122:
            nueva = chr(tecla).upper()
        else:
            continue

        if nueva not in CLASES:
            print(f"  Etiqueta invalida: {nueva}")
            continue

        dest = _mover_a_train(ruta, nueva)
        historial.append((ruta, dest))
        print(f"  -> train/{nueva}/{dest.name}")
        idx += 1

    cv2.destroyAllWindows()
    restantes = len(_listar_pendientes())
    print(f"\nListo. Pendientes en revision: {restantes}")
    if restantes == 0:
        print("Siguiente: python scripts/finetune_character_cnn_reales.py")


if __name__ == "__main__":
    main()

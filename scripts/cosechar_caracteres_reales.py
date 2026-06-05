"""Cosecha caracteres reales desde recortes de placa para fine-tune de la CNN.

Flujo:
  1. Preprocesa cada recorte y segmenta con camino limpio.
  2. Predice cada caracter y normaliza a 32x32 (igual que inferencia).
  3. Auto-etiqueta en train/ cuando la lectura es confiable.
  4. Deja el resto en revision/ para correccion manual.

Criterio auto-etiqueta (conservador):
  - 6 o 7 caracteres segmentados.
  - Formato ecuatoriano valido tras postproceso.
  - Confianza minima >= 0.80 y promedio >= 0.88.
  - Cada caracter cumple tipo esperado (letra en pos 0-2, digito en 3+).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.plate_reader as pr  # noqa: E402

DATASET_DIR = ROOT / "datasets" / "caracteres_reales"
TRAIN_DIR = DATASET_DIR / "train"
REVISION_DIR = DATASET_DIR / "revision"
MANIFEST_PATH = DATASET_DIR / "manifest.jsonl"
RECORTES_DIR = ROOT / "reports" / "evidencias" / "mejores_recortes_placa"

# Ground-truth manual del benchmark (si el recorte esta en la lista).
CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

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


def _tipo_posicion(idx: int) -> str:
    return "letra" if idx < 3 else "numero"


def _cumple_tipo(caracter: str, idx: int) -> bool:
    if _tipo_posicion(idx) == "letra":
        return caracter.isalpha()
    return caracter.isdigit()


def _guardar_normalizado(imagen, destino: Path) -> bool:
    norm = pr.normalizar_imagen_caracter_para_cnn(imagen, destino.stem)
    if norm.get("array") is None:
        return False
    arr = (norm["array"][0, :, :, 0] * 255).astype(np.uint8)
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), arr)
    return True


def _procesar_recorte(ruta: Path, modelo, class_names: list, contadores: dict) -> list[dict]:
    nombre = ruta.stem
    prep = pr.preprocesar_placa(str(ruta), nombre)
    if not prep.get("ruta_imagen_procesada"):
        return []

    rutas_dbg = prep.get("rutas_debug") or {}
    ruta_gris = rutas_dbg.get("contraste") or rutas_dbg.get("gris_normalizado") or rutas_dbg.get("gris")
    seg = pr.segmentar_caracteres_camino_limpio(prep["ruta_imagen_procesada"], ruta_gris, nombre)
    chars = seg.get("caracteres", [])
    if not chars:
        return []

    recon = pr.reconstruir_placa_desde_caracteres(chars, modelo, class_names)
    preds = recon.get("predicciones_caracteres", [])
    post = pr.postprocesar_por_formato_ecuador(recon.get("texto_detectado", ""), preds)
    texto_pp = post.get("texto_postprocesado", "")
    formato_ok = bool((post.get("formato") or {}).get("valido"))

    confs = [float(p.get("confianza", 0.0)) for p in preds]
    conf_min = min(confs) if confs else 0.0
    conf_avg = sum(confs) / len(confs) if confs else 0.0

    gt_placa = BENCH_GT.get(nombre)
    usar_gt = gt_placa is not None and len(gt_placa) == len(preds)

    auto_ok = (
        len(preds) in (6, 7)
        and formato_ok
        and conf_min >= 0.80
        and conf_avg >= 0.88
        and all(_cumple_tipo(c, i) for i, c in enumerate(texto_pp))
    )

    registros = []
    for i, (ch, pred) in enumerate(zip(chars, preds)):
        crop = cv2.imread(ch.get("ruta_caracter", ""), cv2.IMREAD_GRAYSCALE)
        if crop is None:
            continue

        pred_char = pred.get("caracter_predicho", "")
        conf = float(pred.get("confianza", 0.0))

        if usar_gt:
            etiqueta = gt_placa[i]
            destino_tipo = "train_gt"
            dest_dir = TRAIN_DIR / etiqueta
        elif auto_ok:
            etiqueta = texto_pp[i] if i < len(texto_pp) else pred_char
            destino_tipo = "train_auto"
            dest_dir = TRAIN_DIR / etiqueta
        else:
            etiqueta = pred_char or "?"
            destino_tipo = "revision"
            dest_dir = REVISION_DIR / (etiqueta if etiqueta in CLASES else "?")

        fname = f"{nombre}_pos{i+1:02d}_{pred_char}_{conf:.2f}.jpg"
        dest = dest_dir / fname
        if _guardar_normalizado(crop, dest):
            contadores[destino_tipo] = contadores.get(destino_tipo, 0) + 1
            registros.append(
                {
                    "placa": nombre,
                    "posicion": i + 1,
                    "etiqueta": etiqueta,
                    "prediccion": pred_char,
                    "confianza": conf,
                    "destino": destino_tipo,
                    "ruta": str(dest.relative_to(ROOT)),
                    "texto_placa": texto_pp,
                    "gt_placa": gt_placa,
                }
            )
    return registros


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosechar caracteres reales desde recortes de placa.")
    parser.add_argument("--dir", type=str, default=str(RECORTES_DIR), help="Carpeta con mejor_recorte_*.jpg")
    parser.add_argument("--limite", type=int, default=0, help="Maximo de placas (0 = todas)")
    args = parser.parse_args()

    carga, msg = pr.cargar_modelo_caracteres()
    if carga is None:
        print(f"Error: {msg}")
        sys.exit(1)
    modelo, class_names = carga

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    REVISION_DIR.mkdir(parents=True, exist_ok=True)

    dir_recortes = Path(args.dir)
    placas = sorted(dir_recortes.glob("mejor_recorte_*.jpg"))
    if args.limite > 0:
        placas = placas[: args.limite]

    contadores: dict[str, int] = {}
    todos: list[dict] = []

    print(f"Cosechando {len(placas)} placas desde {dir_recortes} ...")
    for idx, ruta in enumerate(placas, start=1):
        regs = _procesar_recorte(ruta, modelo, class_names, contadores)
        todos.extend(regs)
        if idx % 25 == 0 or idx == len(placas):
            print(f"  {idx}/{len(placas)} placas procesadas, {len(todos)} caracteres")

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for reg in todos:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    resumen = {
        "placas_procesadas": len(placas),
        "caracteres_totales": len(todos),
        "por_destino": contadores,
        "train_por_clase": {
            c.name: len(list((TRAIN_DIR / c.name).glob("*.jpg")))
            for c in TRAIN_DIR.iterdir()
            if c.is_dir()
        },
        "revision_por_clase": {
            c.name: len(list((REVISION_DIR / c.name).glob("*.jpg")))
            for c in REVISION_DIR.iterdir()
            if c.is_dir()
        },
    }
    (DATASET_DIR / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RESUMEN COSECHA ===")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\nManifest: {MANIFEST_PATH}")
    print("Siguiente paso: python scripts/revisar_caracteres_cosechados.py")


if __name__ == "__main__":
    main()

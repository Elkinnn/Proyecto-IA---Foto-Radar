from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "datasets" / "placas_yolo_roboflow"
DST_DIR = ROOT_DIR / "datasets" / "placas_yolo_roboflow_limpio"
REPORT_DIR = ROOT_DIR / "reports" / "evidencias" / "yolo_roboflow_limpieza"
REPORT_TXT = REPORT_DIR / "limpieza_dataset_yolo_roboflow.txt"
REPORT_CSV = REPORT_DIR / "limpieza_dataset_yolo_roboflow.csv"
SPLITS = ["train", "valid", "test"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def copiar_dataset() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"No existe dataset fuente: {SRC_DIR}")
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)


def ajustar_data_yaml() -> None:
    data_path = DST_DIR / "data.yaml"
    if not data_path.exists():
        return
    with open(data_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["path"] = "datasets/placas_yolo_roboflow_limpio"
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    with open(data_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def class_ids_from_yaml() -> set[int]:
    data_path = DST_DIR / "data.yaml"
    if not data_path.exists():
        return {0}
    with open(data_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return set(range(len(names)))
    if isinstance(names, dict):
        return {int(k) for k in names}
    return {0}


def label_dirs() -> list[Path]:
    dirs = []
    for split in SPLITS:
        base = DST_DIR / split
        labels = base / "labels" if (base / "labels").exists() else base
        if labels.exists():
            dirs.append(labels)
    return dirs


def validar_linea(line: str, valid_ids: set[int]) -> tuple[bool, str]:
    parts = line.strip().split()
    if len(parts) != 5:
        return False, "cantidad_valores_distinta_de_5"
    try:
        class_id = int(float(parts[0]))
        x, y, w, h = [float(v) for v in parts[1:]]
    except ValueError:
        return False, "valores_no_numericos"
    if class_id not in valid_ids:
        return False, f"clase_invalida_{class_id}"
    if x < 0 or x > 1:
        return False, "x_center_fuera_0_1"
    if y < 0 or y > 1:
        return False, "y_center_fuera_0_1"
    if w <= 0 or w > 1:
        return False, "width_invalido"
    if h <= 0 or h > 1:
        return False, "height_invalido"
    return True, ""


def limpiar_labels() -> dict:
    valid_ids = class_ids_from_yaml()
    report_rows = []
    labels_revisados = 0
    lineas_revisadas = 0
    lineas_validas = 0
    lineas_invalidas = 0
    labels_corregidos = set()
    labels_vacios_final = 0

    for labels_dir in label_dirs():
        for label_path in sorted(labels_dir.rglob("*.txt")):
            labels_revisados += 1
            originales = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            validas = []
            for num_linea, line in enumerate(originales, start=1):
                if not line.strip():
                    lineas_revisadas += 1
                    lineas_invalidas += 1
                    labels_corregidos.add(label_path)
                    report_rows.append(row(label_path, num_linea, line, "linea_vacia"))
                    continue
                lineas_revisadas += 1
                ok, motivo = validar_linea(line, valid_ids)
                if ok:
                    lineas_validas += 1
                    validas.append(line.strip())
                else:
                    lineas_invalidas += 1
                    labels_corregidos.add(label_path)
                    report_rows.append(row(label_path, num_linea, line, motivo))

            label_path.write_text("\n".join(validas) + ("\n" if validas else ""), encoding="utf-8")
            if not validas:
                labels_vacios_final += 1

    return {
        "rows": report_rows,
        "labels_revisados": labels_revisados,
        "lineas_revisadas": lineas_revisadas,
        "lineas_validas": lineas_validas,
        "lineas_invalidas": lineas_invalidas,
        "labels_corregidos": len(labels_corregidos),
        "labels_vacios_final": labels_vacios_final,
    }


def row(label_path: Path, num_linea: int, line: str, motivo: str) -> dict:
    return {
        "label_corregido": rel(label_path),
        "linea_numero": num_linea,
        "linea_original_eliminada": line,
        "motivo_eliminacion": motivo,
    }


def write_reports(stats: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["label_corregido", "linea_numero", "linea_original_eliminada", "motivo_eliminacion"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats["rows"])

    lines = [
        "Limpieza dataset YOLO Roboflow",
        "-------------------------------",
        f"Fecha: {datetime.now().isoformat(timespec='seconds')}",
        f"Fuente: {rel(SRC_DIR)}",
        f"Destino limpio: {rel(DST_DIR)}",
        "",
        f"Total labels revisados: {stats['labels_revisados']}",
        f"Total lineas revisadas: {stats['lineas_revisadas']}",
        f"Lineas validas: {stats['lineas_validas']}",
        f"Lineas invalidas eliminadas: {stats['lineas_invalidas']}",
        f"Labels corregidos: {stats['labels_corregidos']}",
        f"Labels que quedaron vacios: {stats['labels_vacios_final']}",
        "",
        f"Reporte CSV: {rel(REPORT_CSV)}",
    ]
    if stats["rows"][:20]:
        lines.extend(["", "Primeras lineas eliminadas:"])
        lines.extend(
            f"- {r['label_corregido']} linea {r['linea_numero']}: {r['motivo_eliminacion']} | {r['linea_original_eliminada']}"
            for r in stats["rows"][:20]
        )
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    try:
        copiar_dataset()
        ajustar_data_yaml()
        stats = limpiar_labels()
        write_reports(stats)
    except Exception as exc:
        print(f"Error limpiando dataset YOLO Roboflow: {exc}")
        sys.exit(1)

    print(REPORT_TXT.read_text(encoding="utf-8"))
    print("")
    print(f"Dataset limpio creado en: {rel(DST_DIR)}")


if __name__ == "__main__":
    main()

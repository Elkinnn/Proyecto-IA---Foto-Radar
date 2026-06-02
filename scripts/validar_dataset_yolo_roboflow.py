from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "placas_yolo_roboflow"
REPORT_DIR = ROOT_DIR / "reports" / "evidencias" / "yolo_roboflow"
REPORT_TXT = REPORT_DIR / "validacion_dataset_yolo_roboflow.txt"
REPORT_CSV = REPORT_DIR / "validacion_dataset_yolo_roboflow.csv"
SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASS_HINTS = {"placa", "plate", "license_plate", "license-plate", "licenseplate", "matricula"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida un dataset YOLO de Roboflow para placas.")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_DIR),
        help="Ruta del dataset a validar. Por defecto: datasets/placas_yolo_roboflow",
    )
    return parser.parse_args()


def split_dirs(dataset_dir: Path, split: str) -> tuple[Path, Path]:
    base = dataset_dir / split
    images = base / "images" if (base / "images").exists() else base
    labels = base / "labels" if (base / "labels").exists() else base
    return images, labels


def load_yaml(dataset_dir: Path) -> dict:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return {}
    with open(data_yaml, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def names_from_yaml(data: dict) -> dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)


def list_labels(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.txt") if p.is_file())


def stem_map(paths: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def validate_label(path: Path, valid_class_ids: set[int]) -> tuple[int, list[str], set[int]]:
    errors = []
    classes = set()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        errors.append("label_vacio")
        return 0, errors, classes
    annotations = 0
    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"linea_{line_no}_no_tiene_5_valores")
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"linea_{line_no}_valores_no_numericos")
            continue
        classes.add(class_id)
        if class_id not in valid_class_ids:
            errors.append(f"linea_{line_no}_class_id_invalido_{class_id}")
        x, y, w, h = coords
        if any(v < 0 or v > 1 for v in coords):
            errors.append(f"linea_{line_no}_coord_fuera_0_1")
        if w <= 0 or h <= 0:
            errors.append(f"linea_{line_no}_ancho_alto_invalido")
        annotations += 1
    return annotations, errors, classes


def class_name_ok(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return normalized in VALID_CLASS_HINTS or "plate" in normalized or "placa" in normalized


def validar(dataset_dir: Path = DATASET_DIR) -> dict:
    data = load_yaml(dataset_dir)
    names = names_from_yaml(data)
    valid_ids = set(names.keys())
    rows = []
    errors = []
    warnings = []
    total_annotations = 0
    classes_present = set()

    if not dataset_dir.exists():
        errors.append(f"No existe dataset: {dataset_dir}")
    if not (dataset_dir / "data.yaml").exists():
        errors.append("No existe data.yaml")
    if len(names) > 1:
        warnings.append(f"Dataset con mas de una clase: {names}")
    for class_id, name in names.items():
        if not class_name_ok(name):
            warnings.append(f"Clase {class_id} con nombre no esperado para placa: {name}")

    for split in SPLITS:
        base = dataset_dir / split
        images_dir, labels_dir = split_dirs(dataset_dir, split)
        images = list_images(images_dir)
        labels = list_labels(labels_dir)
        image_map = stem_map(images)
        label_map = stem_map(labels)
        images_without_label = sorted(set(image_map) - set(label_map))
        labels_without_image = sorted(set(label_map) - set(image_map))
        split_annotations = 0
        split_errors = []
        empty_labels = []

        if not base.exists():
            errors.append(f"No existe split {split}")
        if not images_dir.exists():
            errors.append(f"No existe directorio de imagenes para {split}: {images_dir}")
        if not labels_dir.exists():
            errors.append(f"No existe directorio de labels para {split}: {labels_dir}")

        for label in labels:
            annotations, label_errors, label_classes = validate_label(label, valid_ids)
            split_annotations += annotations
            classes_present.update(label_classes)
            if "label_vacio" in label_errors:
                empty_labels.append(label)
            for err in label_errors:
                if err == "label_vacio":
                    continue
                split_errors.append(f"{label.relative_to(ROOT_DIR)}:{err}")

        total_annotations += split_annotations
        rows.append(
            {
                "split": split,
                "images_dir": _rel(images_dir) if images_dir.exists() else str(images_dir),
                "labels_dir": _rel(labels_dir) if labels_dir.exists() else str(labels_dir),
                "imagenes": len(images),
                "labels": len(labels),
                "imagenes_sin_label": len(images_without_label),
                "labels_sin_imagen": len(labels_without_image),
                "labels_vacios": len(empty_labels),
                "anotaciones": split_annotations,
                "errores_labels": len(split_errors),
                "detalle_imagenes_sin_label": ";".join(images_without_label[:50]),
                "detalle_labels_sin_imagen": ";".join(labels_without_image[:50]),
                "detalle_errores": ";".join(split_errors[:50]),
            }
        )
        if images_without_label:
            warnings.append(f"{split}: {len(images_without_label)} imagenes sin label")
        if labels_without_image:
            warnings.append(f"{split}: {len(labels_without_image)} labels sin imagen")
        if empty_labels:
            warnings.append(f"{split}: {len(empty_labels)} labels vacios tratados como posibles imagenes negativas/background")
        if split_errors:
            errors.append(f"{split}: {len(split_errors)} errores de formato YOLO")

    valid = not errors
    return {
        "rows": rows,
        "errors": errors,
        "warnings": warnings,
        "names": names,
        "classes_present": sorted(classes_present),
        "total_annotations": total_annotations,
        "valid": valid,
        "dataset_dir": dataset_dir,
    }


def write_reports(result: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()) if result["rows"] else ["split"])
        writer.writeheader()
        writer.writerows(result["rows"])

    lines = [
        "Validacion dataset YOLO Roboflow",
        "--------------------------------",
        f"Ruta: {_rel(result['dataset_dir'])}",
        f"data.yaml existe: {'si' if (result['dataset_dir'] / 'data.yaml').exists() else 'no'}",
        f"Clases definidas: {result['names']}",
        f"Clases presentes en labels: {result['classes_present']}",
        f"Total anotaciones: {result['total_annotations']}",
        f"Dataset valido: {'si' if result['valid'] else 'no'}",
        "",
        "Resumen por split:",
    ]
    for row in result["rows"]:
        lines.append(
            f"- {row['split']}: imagenes={row['imagenes']} labels={row['labels']} "
            f"sin_label={row['imagenes_sin_label']} labels_sin_imagen={row['labels_sin_imagen']} "
            f"labels_vacios={row['labels_vacios']} anotaciones={row['anotaciones']} errores={row['errores_labels']}"
        )
    if result["warnings"]:
        lines.extend(["", "Advertencias:"])
        lines.extend(f"- {w}" for w in result["warnings"])
    if result["errors"]:
        lines.extend(["", "Errores:"])
        lines.extend(f"- {e}" for e in result["errors"])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT_DIR / dataset_dir
    result = validar(dataset_dir)
    write_reports(result)
    print(REPORT_TXT.read_text(encoding="utf-8"))
    print("")
    print(f"Reporte TXT: {REPORT_TXT.relative_to(ROOT_DIR)}")
    print(f"Reporte CSV: {REPORT_CSV.relative_to(ROOT_DIR)}")
    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

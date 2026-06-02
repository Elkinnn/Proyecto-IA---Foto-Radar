from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import torch
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
ARQUITECTURA_YOLO = "yolov8n.yaml"
DEFAULT_DATA = ROOT_DIR / "datasets" / "placas_yolo_roboflow" / "data.yaml"
PROJECT_DIR = ROOT_DIR / "runs" / "plate_detector_roboflow"
BEST_MODEL_DEST = ROOT_DIR / "models" / "plate_detector" / "placas_roboflow.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO desde cero con dataset Roboflow de placas.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Ruta a data.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="entrenamiento_placas_roboflow")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraccion del dataset a usar durante entrenamiento.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, etc.")
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path
    if not data_path.exists():
        print(f"No existe data.yaml: {data_path}")
        sys.exit(1)

    device = resolve_device(args.device)
    print(f"Arquitectura: {ARQUITECTURA_YOLO}")
    print(f"Dataset: {data_path}")
    print(f"Device: {device}")
    print("Entrenamiento desde cero: pretrained=False")

    model = YOLO(ARQUITECTURA_YOLO)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        workers=0,
        device=device,
        pretrained=False,
        amp=False,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists() and getattr(model.trainer, "best", None):
        best_pt = Path(model.trainer.best)
    if not best_pt.exists():
        raise FileNotFoundError(f"No se encontro best.pt en {save_dir}")

    BEST_MODEL_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, BEST_MODEL_DEST)
    print(f"Resultados guardados en: {save_dir}")
    print(f"Mejor modelo copiado a: {BEST_MODEL_DEST.relative_to(ROOT_DIR)}")
    print("Modelo actual placas_ecuador.pt no fue sobrescrito.")


if __name__ == "__main__":
    main()

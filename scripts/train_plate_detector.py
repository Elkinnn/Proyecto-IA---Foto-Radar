import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


ARQUITECTURA_YOLO = "yolov8n.yaml"
DATASET_YAML = "datasets/placas_ecuador/data.yaml"
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
WORKERS = 0
DEVICE = 0
PROJECT_DIR = Path("runs/plate_detector").resolve()
RUN_NAME = "entrenamiento_desde_cero"
BEST_MODEL_DEST = Path("models/plate_detector/placas_ecuador.pt")


def main() -> None:
    if not Path(DATASET_YAML).exists():
        raise FileNotFoundError(f"No existe {DATASET_YAML}. Ejecute primero scripts/prepare_plate_dataset.py")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Se encontro GPU NVIDIA en el sistema, pero PyTorch no tiene CUDA disponible en este entorno. "
            "Instale una version de PyTorch con soporte CUDA dentro del venv antes de entrenar en GPU."
        )

    device = DEVICE
    print(f"Dispositivo de entrenamiento: cuda:{device} - {torch.cuda.get_device_name(device)}")

    model = YOLO(ARQUITECTURA_YOLO)
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        pretrained=False,
        amp=False,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists() and getattr(model.trainer, "best", None):
        best_pt = Path(model.trainer.best)

    print(f"Resultados guardados en: {save_dir}")
    if not best_pt.exists():
        raise FileNotFoundError(f"No se encontro el modelo entrenado en {best_pt}")

    BEST_MODEL_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, BEST_MODEL_DEST)
    print(f"Modelo copiado a: {BEST_MODEL_DEST}")


if __name__ == "__main__":
    main()

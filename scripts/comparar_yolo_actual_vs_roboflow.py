from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import sys

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ACTUAL = ROOT_DIR / "models" / "plate_detector" / "placas_ecuador.pt"
MODEL_ROBOFLOW = ROOT_DIR / "models" / "plate_detector" / "placas_roboflow.pt"
OUT_DIR = ROOT_DIR / "reports" / "evidencias" / "yolo_comparacion_roboflow"
BBOX_ACTUAL_DIR = OUT_DIR / "bbox_actual"
BBOX_ROBOFLOW_DIR = OUT_DIR / "bbox_roboflow"
CROP_ACTUAL_DIR = OUT_DIR / "recortes_actual"
CROP_ROBOFLOW_DIR = OUT_DIR / "recortes_roboflow"
CSV_PATH = OUT_DIR / "comparacion_yolo_actual_vs_roboflow.csv"
SUMMARY_PATH = OUT_DIR / "resumen_comparacion_yolo_actual_vs_roboflow.txt"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
SOURCES = [
    ROOT_DIR / "datasets" / "placas_ecuador" / "images",
    ROOT_DIR / "datasets" / "placas_yolo_roboflow" / "test",
    ROOT_DIR / "reports" / "evidencias" / "monitoreo_video",
    ROOT_DIR / "data" / "videos_prueba",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara YOLO actual vs YOLO Roboflow.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--video-step", type=int, default=10)
    parser.add_argument("--source", default="", help="Imagen, carpeta o video especifico para comparar.")
    return parser.parse_args()


def load_model(path: Path):
    if not path.exists():
        return None
    from ultralytics import YOLO

    return YOLO(str(path))


def prepare_dirs() -> None:
    for path in [OUT_DIR, BBOX_ACTUAL_DIR, BBOX_ROBOFLOW_DIR, CROP_ACTUAL_DIR, CROP_ROBOFLOW_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def list_items(limit: int, video_step: int, source: str = "") -> list[dict]:
    if source:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = ROOT_DIR / source_path
        return list_items_from_source(source_path, limit, video_step)

    items = []
    frames_dir = OUT_DIR / "frames_video"
    for source in SOURCES:
        if not source.exists():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in IMG_EXT:
                items.append({"ruta": path, "tipo": "imagen", "frame": ""})
            elif path.suffix.lower() in VIDEO_EXT:
                items.extend(extract_video_frames(path, frames_dir, limit - len(items), video_step))
            if len(items) >= limit:
                return items[:limit]
    return items[:limit]


def list_items_from_source(source: Path, limit: int, video_step: int) -> list[dict]:
    frames_dir = OUT_DIR / "frames_video"
    if source.is_file() and source.suffix.lower() in IMG_EXT:
        return [{"ruta": source, "tipo": "imagen", "frame": ""}]
    if source.is_file() and source.suffix.lower() in VIDEO_EXT:
        return extract_video_frames(source, frames_dir, limit, video_step)
    if source.is_dir():
        items = [
            {"ruta": path, "tipo": "imagen", "frame": ""}
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        return items[:limit]
    return []


def extract_video_frames(video: Path, frames_dir: Path, remaining: int, video_step: int) -> list[dict]:
    frames = []
    if remaining <= 0:
        return frames
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return frames
    frames_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while len(frames) < remaining:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(video_step, 1) == 0:
            out = frames_dir / f"{video.stem}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            frames.append({"ruta": out, "tipo": "video_frame", "video_origen": video, "frame": idx})
        idx += 1
    cap.release()
    return frames


def infer(model, image, conf: float, out_bbox: Path, out_crop: Path, base_name: str) -> dict:
    if model is None:
        return {
            "available": False,
            "detectada": False,
            "confianza": "",
            "tiempo_ms": "",
            "bbox": "",
            "recorte": "",
            "posible_fp": False,
            "area_relativa": "",
            "aspect_ratio": "",
        }
    start = time.perf_counter()
    results = model.predict(source=image, conf=conf, verbose=False)
    elapsed = (time.perf_counter() - start) * 1000
    boxes = results[0].boxes if results else []
    overlay = image.copy()
    best = None
    for box in boxes:
        confidence = float(box.conf[0])
        if best is None or confidence > best["confianza"]:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            best = {"confianza": confidence, "bbox": [x1, y1, x2, y2]}
    crop_rel = ""
    area_rel = ""
    aspect = ""
    posible_fp = False
    if best:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = best["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        area_rel = (bw * bh) / max(w * h, 1)
        aspect = bw / max(bh, 1)
        posible_fp = confidence_is_suspicious(best["confianza"], area_rel, aspect)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(overlay, f"{best['confianza']:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        crop_path = out_crop / f"{base_name}_crop.jpg"
        cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
        crop_rel = str(crop_path.relative_to(ROOT_DIR))
    bbox_path = out_bbox / f"{base_name}_bbox.jpg"
    cv2.imwrite(str(bbox_path), overlay)
    return {
        "available": True,
        "detectada": best is not None,
        "confianza": best["confianza"] if best else "",
        "bbox": best["bbox"] if best else "",
        "area_relativa": area_rel,
        "aspect_ratio": aspect,
        "tiempo_ms": elapsed,
        "bbox_img": str(bbox_path.relative_to(ROOT_DIR)),
        "recorte": crop_rel,
        "detecciones": len(boxes),
        "posible_fp": posible_fp,
    }


def confidence_is_suspicious(confidence: float, area_rel: float, aspect: float) -> bool:
    return confidence < 0.35 or area_rel < 0.001 or area_rel > 0.30 or aspect < 1.5 or aspect > 7.0


def compare(args: argparse.Namespace) -> list[dict]:
    prepare_dirs()
    model_actual = load_model(MODEL_ACTUAL)
    model_roboflow = load_model(MODEL_ROBOFLOW)
    rows = []
    for idx, item in enumerate(list_items(args.limit, args.video_step, args.source), start=1):
        img = cv2.imread(str(item["ruta"]))
        if img is None:
            continue
        base = f"{idx:05d}_{item['ruta'].stem}"
        actual = infer(model_actual, img, args.conf, BBOX_ACTUAL_DIR, CROP_ACTUAL_DIR, f"{base}_actual")
        robo = infer(model_roboflow, img, args.conf, BBOX_ROBOFLOW_DIR, CROP_ROBOFLOW_DIR, f"{base}_roboflow")
        fuente = rel(item["ruta"])
        frame_index = item.get("frame", "")
        rows.append(row_model(idx, fuente, frame_index, "actual", actual))
        rows.append(row_model(idx, fuente, frame_index, "roboflow", robo))
    return rows


def row_model(indice: int, fuente: str, frame_index, modelo: str, result: dict) -> dict:
    return {
        "indice": indice,
        "fuente": fuente,
        "frame_index": frame_index,
        "modelo": modelo,
        "modelo_disponible": result["available"],
        "detecto": result["detectada"],
        "confidence": result["confianza"],
        "bbox": result["bbox"],
        "area_relativa": result.get("area_relativa", ""),
        "aspect_ratio": result.get("aspect_ratio", ""),
        "tiempo_ms": result["tiempo_ms"],
        "posible_falso_positivo": result["posible_fp"],
        "bbox_img": result.get("bbox_img", ""),
        "recorte": result.get("recorte", ""),
    }


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def save(rows: list[dict], conf: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["indice"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_PATH.write_text(summary(rows, conf), encoding="utf-8")


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summary(rows: list[dict], conf: float) -> str:
    pairs = group_pairs(rows)
    total = len(pairs)
    actual_rows = [r for r in rows if r["modelo"] == "actual"]
    robo_rows = [r for r in rows if r["modelo"] == "roboflow"]
    actual_detect = [r for r in actual_rows if r["detecto"]]
    robo_detect = [r for r in robo_rows if r["detecto"]]
    actual_conf = [float(r["confidence"]) for r in actual_rows if r["confidence"] != ""]
    robo_conf = [float(r["confidence"]) for r in robo_rows if r["confidence"] != ""]
    actual_time = [float(r["tiempo_ms"]) for r in actual_rows if r["tiempo_ms"] != ""]
    robo_time = [float(r["tiempo_ms"]) for r in robo_rows if r["tiempo_ms"] != ""]
    actual_fails_robo_detects = []
    robo_fails_actual_detects = []
    for key, pair in pairs.items():
        actual = pair.get("actual", {})
        robo = pair.get("roboflow", {})
        fuente = key[0]
        if not actual.get("detecto") and robo.get("detecto"):
            actual_fails_robo_detects.append(fuente)
        if actual.get("detecto") and not robo.get("detecto"):
            robo_fails_actual_detects.append(fuente)
    actual_rate = len(actual_detect) / total * 100 if total else 0
    robo_rate = len(robo_detect) / total * 100 if total else 0
    recommendation = "Usar Roboflow como candidato principal." if robo_rate > actual_rate and avg(robo_conf) >= avg(actual_conf) * 0.9 else "Mantener modelo actual hasta revisar evidencia visual."
    return "\n".join(
        [
            "Comparacion YOLO actual vs Roboflow",
            "-----------------------------------",
            f"Fecha: {datetime.now().isoformat(timespec='seconds')}",
            f"Conf threshold: {conf}",
            f"Modelo actual: {rel(MODEL_ACTUAL)} existe={'si' if MODEL_ACTUAL.exists() else 'no'}",
            f"Modelo Roboflow: {rel(MODEL_ROBOFLOW)} existe={'si' if MODEL_ROBOFLOW.exists() else 'no'}",
            "",
            f"Total evaluado: {total}",
            f"Deteccion modelo actual: {actual_rate:.2f}%",
            f"Deteccion modelo Roboflow: {robo_rate:.2f}%",
            f"Confianza promedio actual: {avg(actual_conf):.4f}",
            f"Confianza promedio Roboflow: {avg(robo_conf):.4f}",
            f"Tiempo promedio actual: {avg(actual_time):.2f} ms",
            f"Tiempo promedio Roboflow: {avg(robo_time):.2f} ms",
            f"Falsos positivos probables actual: {sum(1 for r in actual_rows if r['posible_falso_positivo'])}",
            f"Falsos positivos probables Roboflow: {sum(1 for r in robo_rows if r['posible_falso_positivo'])}",
            "",
            "Actual falla y Roboflow detecta:",
            *(f"- {r}" for r in actual_fails_robo_detects[:50]),
            "",
            "Roboflow falla y actual detecta:",
            *(f"- {r}" for r in robo_fails_actual_detects[:50]),
            "",
            f"Recomendacion: {recommendation}",
        ]
    )


def group_pairs(rows: list[dict]) -> dict:
    pairs = {}
    for row in rows:
        key = (row["fuente"], row["frame_index"])
        pairs.setdefault(key, {})[row["modelo"]] = row
    return pairs


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print("Ultralytics no esta instalado en este entorno.")
        sys.exit(1)
    rows = compare(args)
    save(rows, args.conf)
    print(SUMMARY_PATH.read_text(encoding="utf-8"))
    print("")
    print(f"CSV: {CSV_PATH.relative_to(ROOT_DIR)}")
    print(f"Resumen: {SUMMARY_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

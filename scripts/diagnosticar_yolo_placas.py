from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_detector import DEFAULT_MODEL_PATH  # noqa: E402


MODEL_PATH = ROOT_DIR / DEFAULT_MODEL_PATH
SALIDA_DIR = ROOT_DIR / "reports" / "evidencias" / "yolo_diagnostico"
ORIGINALES_DIR = SALIDA_DIR / "originales"
BBOX_DIR = SALIDA_DIR / "bbox"
RECORTES_DIR = SALIDA_DIR / "recortes"
CSV_PATH = SALIDA_DIR / "diagnostico_yolo_placas.csv"
JSON_PATH = SALIDA_DIR / "diagnostico_yolo_placas.json"
RESUMEN_PATH = SALIDA_DIR / "resumen_diagnostico_yolo.txt"
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
FUENTES = [
    ROOT_DIR / "datasets" / "placas_ecuador" / "images",
    ROOT_DIR / "reports" / "evidencias" / "monitoreo_video",
    ROOT_DIR / "reports" / "evidencias" / "eventos_placa",
]
CONF_BAJA = 0.35
AREA_MIN_FP = 0.001
AREA_MAX_FP = 0.30
ASPECT_MIN_FP = 1.5
ASPECT_MAX_FP = 7.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostica el detector YOLO de placas ecuatorianas.")
    parser.add_argument("--limit", type=int, default=200, help="Maximo de imagenes/frames a evaluar.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold para YOLO.")
    parser.add_argument("--video-step", type=int, default=30, help="Tomar un frame cada N frames en videos.")
    return parser.parse_args()


def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe modelo YOLO: {MODEL_PATH.relative_to(ROOT_DIR)}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics no esta instalado en este entorno.") from exc
    return YOLO(str(MODEL_PATH))


def listar_fuentes(limit: int, video_step: int) -> list[dict]:
    items = []
    for base in FUENTES:
        if not base.exists():
            continue
        for ruta in sorted(base.rglob("*")):
            if not ruta.is_file():
                continue
            sufijo = ruta.suffix.lower()
            if sufijo in EXT_IMAGEN:
                items.append({"tipo": "imagen", "ruta": ruta, "frame": None})
            elif sufijo in EXT_VIDEO:
                items.extend(extraer_frames_video(ruta, limit - len(items), video_step))
            if len(items) >= limit:
                return items[:limit]
    return items[:limit]


def extraer_frames_video(ruta_video: Path, restante: int, video_step: int) -> list[dict]:
    if restante <= 0:
        return []
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        return []
    items = []
    frame_idx = 0
    while len(items) < restante:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(video_step, 1) == 0:
            TMP_VIDEO_FRAMES_DIR = SALIDA_DIR / "frames_video"
            TMP_VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            ruta_frame = TMP_VIDEO_FRAMES_DIR / f"{ruta_video.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(ruta_frame), frame)
            items.append({"tipo": "video_frame", "ruta": ruta_frame, "video_origen": ruta_video, "frame": frame_idx})
        frame_idx += 1
    cap.release()
    return items


def preparar_salida() -> None:
    for carpeta in [SALIDA_DIR, ORIGINALES_DIR, BBOX_DIR, RECORTES_DIR]:
        carpeta.mkdir(parents=True, exist_ok=True)


def diagnosticar_item(modelo, item: dict, conf: float, indice: int) -> list[dict]:
    ruta = item["ruta"]
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        return [
            fila_base(item, indice)
            | {
                "detectada": False,
                "mensaje": "No se pudo leer imagen/frame.",
            }
        ]

    alto, ancho = imagen.shape[:2]
    nombre_base = f"{indice:05d}_{ruta.stem}"
    ruta_original = ORIGINALES_DIR / f"{nombre_base}.jpg"
    ruta_bbox = BBOX_DIR / f"{nombre_base}_bbox.jpg"
    cv2.imwrite(str(ruta_original), imagen)

    resultados = modelo.predict(source=imagen, conf=conf, verbose=False)
    cajas = resultados[0].boxes if resultados else []
    overlay = imagen.copy()
    filas = []

    if not cajas:
        cv2.imwrite(str(ruta_bbox), overlay)
        filas.append(
            fila_base(item, indice)
            | {
                "detectada": False,
                "detecciones_en_imagen": 0,
                "ruta_original_guardada": str(ruta_original.relative_to(ROOT_DIR)),
                "ruta_bbox_guardada": str(ruta_bbox.relative_to(ROOT_DIR)),
                "mensaje": "Sin deteccion YOLO.",
            }
        )
        return filas

    for det_idx, caja in enumerate(cajas, start=1):
        x1, y1, x2, y2 = [int(valor) for valor in caja.xyxy[0].tolist()]
        confianza = float(caja.conf[0])
        clase_id = int(caja.cls[0]) if caja.cls is not None else 0
        x1 = max(0, min(x1, ancho - 1))
        x2 = max(0, min(x2, ancho - 1))
        y1 = max(0, min(y1, alto - 1))
        y2 = max(0, min(y2, alto - 1))
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        area_rel = (bbox_w * bbox_h) / max(ancho * alto, 1)
        aspect = bbox_w / max(bbox_h, 1)
        posible_fp, motivo_fp = evaluar_posible_falso_positivo(confianza, area_rel, aspect)

        crop_rel = ""
        if bbox_w > 0 and bbox_h > 0:
            crop = imagen[y1:y2, x1:x2]
            ruta_crop = RECORTES_DIR / f"{nombre_base}_det_{det_idx:02d}.jpg"
            cv2.imwrite(str(ruta_crop), crop)
            crop_rel = str(ruta_crop.relative_to(ROOT_DIR))

        color = (0, 180, 0) if not posible_fp else (0, 180, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"placa {confianza:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        filas.append(
            fila_base(item, indice)
            | {
                "detectada": True,
                "detecciones_en_imagen": len(cajas),
                "deteccion_idx": det_idx,
                "confianza": confianza,
                "clase_id": clase_id,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_ancho": bbox_w,
                "bbox_alto": bbox_h,
                "area_relativa": area_rel,
                "aspect_ratio": aspect,
                "detecciones_multiples": len(cajas) > 1,
                "posible_falso_positivo": posible_fp,
                "motivo_posible_fp": motivo_fp,
                "ruta_original_guardada": str(ruta_original.relative_to(ROOT_DIR)),
                "ruta_recorte_placa": crop_rel,
                "mensaje": "Deteccion YOLO registrada.",
            }
        )

    cv2.imwrite(str(ruta_bbox), overlay)
    for fila in filas:
        fila["ruta_bbox_guardada"] = str(ruta_bbox.relative_to(ROOT_DIR))
    return filas


def evaluar_posible_falso_positivo(confianza: float, area_rel: float, aspect: float) -> tuple[bool, str]:
    motivos = []
    if confianza < CONF_BAJA:
        motivos.append("baja_confianza")
    if area_rel < AREA_MIN_FP:
        motivos.append("area_muy_pequena")
    if area_rel > AREA_MAX_FP:
        motivos.append("area_muy_grande")
    if aspect < ASPECT_MIN_FP or aspect > ASPECT_MAX_FP:
        motivos.append("aspect_ratio_atipico")
    return bool(motivos), ";".join(motivos)


def fila_base(item: dict, indice: int) -> dict:
    ruta = item["ruta"]
    try:
        ruta_rel = str(ruta.relative_to(ROOT_DIR))
    except ValueError:
        ruta_rel = str(ruta)
    video_origen = item.get("video_origen")
    return {
        "indice": indice,
        "tipo": item.get("tipo"),
        "ruta_fuente": ruta_rel,
        "video_origen": str(video_origen.relative_to(ROOT_DIR)) if video_origen else "",
        "frame_video": item.get("frame") if item.get("frame") is not None else "",
        "detectada": False,
        "detecciones_en_imagen": 0,
        "deteccion_idx": "",
        "confianza": "",
        "clase_id": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "bbox_ancho": "",
        "bbox_alto": "",
        "area_relativa": "",
        "aspect_ratio": "",
        "detecciones_multiples": False,
        "posible_falso_positivo": False,
        "motivo_posible_fp": "",
        "ruta_original_guardada": "",
        "ruta_bbox_guardada": "",
        "ruta_recorte_placa": "",
        "mensaje": "",
    }


def guardar_reportes(filas: list[dict], modelo, conf: float) -> None:
    campos = list(filas[0].keys()) if filas else list(fila_base({"ruta": Path(""), "tipo": ""}, 0).keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    payload = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "modelo": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "modelo_existe": MODEL_PATH.exists(),
        "clases_modelo": {str(k): v for k, v in getattr(modelo, "names", {}).items()},
        "confidence_threshold": conf,
        "resultados": filas,
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    RESUMEN_PATH.write_text(generar_resumen(filas, modelo, conf), encoding="utf-8")


def generar_resumen(filas: list[dict], modelo, conf: float) -> str:
    indices = {fila["indice"] for fila in filas}
    con_det = {fila["indice"] for fila in filas if fila.get("detectada")}
    sin_det = sorted(indices - con_det)
    confs = [float(fila["confianza"]) for fila in filas if fila.get("confianza") not in {"", None}]
    multiples = sorted({fila["indice"] for fila in filas if fila.get("detecciones_multiples")})
    bajas = [fila for fila in filas if fila.get("confianza") not in {"", None, ""} and float(fila["confianza"]) < CONF_BAJA]
    fps = [fila for fila in filas if fila.get("posible_falso_positivo")]
    total = len(indices)
    tasa = len(con_det) / total * 100 if total else 0.0

    por_indice = {}
    for fila in filas:
        por_indice.setdefault(fila["indice"], fila.get("ruta_fuente", ""))

    lineas = [
        "Diagnostico YOLO placas",
        "-----------------------",
        f"Modelo: {MODEL_PATH.relative_to(ROOT_DIR)}",
        f"Modelo existe: {'si' if MODEL_PATH.exists() else 'no'}",
        f"Clases modelo: {getattr(modelo, 'names', {})}",
        f"Confidence threshold usado: {conf}",
        "",
        f"Total de imagenes/frames evaluados: {total}",
        f"Imagenes con deteccion: {len(con_det)}",
        f"Imagenes sin deteccion: {len(sin_det)}",
        f"Detecciones multiples: {len(multiples)}",
        f"Confianza promedio: {(sum(confs) / len(confs)):.4f}" if confs else "Confianza promedio: no disponible",
        f"Confianza minima: {min(confs):.4f}" if confs else "Confianza minima: no disponible",
        f"Confianza maxima: {max(confs):.4f}" if confs else "Confianza maxima: no disponible",
        f"Porcentaje de deteccion: {tasa:.2f}%",
        "",
        "Imagenes donde no detecto:",
        *(f"- {por_indice.get(idx, idx)}" for idx in sin_det[:50]),
        "",
        "Imagenes con baja confianza:",
        *(f"- {fila.get('ruta_fuente')} conf={float(fila.get('confianza')):.3f}" for fila in bajas[:50]),
        "",
        "Posibles falsos positivos:",
        *(f"- {fila.get('ruta_fuente')} conf={fila.get('confianza')} motivo={fila.get('motivo_posible_fp')}" for fila in fps[:50]),
    ]
    return "\n".join(lineas)


def main() -> None:
    args = parse_args()
    preparar_salida()
    try:
        modelo = cargar_modelo()
    except Exception as exc:
        print(f"Error cargando modelo YOLO: {exc}")
        sys.exit(1)

    items = listar_fuentes(args.limit, args.video_step)
    filas = []
    for indice, item in enumerate(items, start=1):
        filas.extend(diagnosticar_item(modelo, item, args.conf, indice))

    guardar_reportes(filas, modelo, args.conf)
    print("Diagnostico YOLO placas")
    print("-----------------------")
    print(f"Evaluados: {len({fila['indice'] for fila in filas})}")
    print(f"Reporte CSV: {CSV_PATH.relative_to(ROOT_DIR)}")
    print(f"Reporte JSON: {JSON_PATH.relative_to(ROOT_DIR)}")
    print(f"Resumen: {RESUMEN_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

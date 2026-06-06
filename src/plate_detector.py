from pathlib import Path
import time

import cv2

from src.plate_reader import _imread_seguro


DEFAULT_MODEL_PATH = "models/plate_detector/placas_ecuador.pt"
MARGEN_BORDE_MIN_PX = 5
AREA_MAX_RELATIVA = 0.25


class PlateDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.estado = "modelo_no_encontrado"
        self._cargar_modelo_si_existe()

    def _cargar_modelo_si_existe(self) -> None:
        if not self.model_path.exists():
            self.estado = "Modelo de placa no encontrado. Entrene primero el detector."
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.estado = "modelo_cargado"
        except Exception as exc:
            self.estado = f"error_cargando_modelo: {exc}"
            self.model = None

    def detectar_en_frame(self, frame, conf_min: float = 0.25, imgsz: int | None = None) -> dict:
        inicio = time.perf_counter()
        if self.model is None:
            return {
                "detectada": False,
                "detecciones": [],
                "detecciones_brutas": 0,
                "debug_detecciones": [],
                "frame_procesado": frame,
                "tiempo_yolo_ms": 0.0,
                "resolucion_inferencia": f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "0x0",
                "mensaje": "Modelo de placa no encontrado. Entrene primero el detector.",
            }

        frame_inferencia, escala_x, escala_y = _preparar_frame_inferencia(frame, imgsz)
        resultados = self.model.predict(source=frame_inferencia, conf=conf_min, verbose=False)
        cajas = resultados[0].boxes if resultados else []
        frame_procesado = frame.copy()
        detecciones = []
        debug_detecciones = []

        for caja in cajas:
            xi1, yi1, xi2, yi2 = caja.xyxy[0].tolist()
            x1 = int(xi1 / escala_x)
            y1 = int(yi1 / escala_y)
            x2 = int(xi2 / escala_x)
            y2 = int(yi2 / escala_y)
            confianza = float(caja.conf[0])
            if confianza < conf_min:
                debug_detecciones.append(_crear_debug(confianza, [x1, y1, x2, y2], False, "baja confianza"))
                continue

            alto, ancho = frame.shape[:2]
            x1 = max(0, min(x1, ancho - 1))
            x2 = max(0, min(x2, ancho - 1))
            y1 = max(0, min(y1, alto - 1))
            y2 = max(0, min(y2, alto - 1))

            if x2 <= x1 or y2 <= y1:
                debug_detecciones.append(_crear_debug(confianza, [x1, y1, x2, y2], False, "bbox invalida"))
                continue

            ancho_bbox = x2 - x1
            alto_bbox = y2 - y1
            area_relativa = (ancho_bbox * alto_bbox) / max(alto * ancho, 1)
            toca_borde = (
                x1 <= MARGEN_BORDE_MIN_PX
                or y1 <= MARGEN_BORDE_MIN_PX
                or x2 >= ancho - MARGEN_BORDE_MIN_PX
                or y2 >= alto - MARGEN_BORDE_MIN_PX
            )
            if toca_borde or area_relativa > AREA_MAX_RELATIVA:
                debug_detecciones.append(_crear_debug(confianza, [x1, y1, x2, y2], False, "placa demasiado cerca o recortada"))
                continue

            recorte = frame[y1:y2, x1:x2].copy()
            etiqueta = f"placa {confianza:.2f}"
            cv2.rectangle(frame_procesado, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(
                frame_procesado,
                etiqueta,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 0),
                2,
            )

            detecciones.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confianza": confianza,
                    "recorte_placa": recorte,
                    "area_relativa": area_relativa,
                }
            )
            debug_detecciones.append(_crear_debug(confianza, [x1, y1, x2, y2], True, ""))

        return {
            "detectada": bool(detecciones),
            "detecciones": detecciones,
            "detecciones_brutas": len(cajas),
            "debug_detecciones": debug_detecciones,
            "frame_procesado": frame_procesado,
            "tiempo_yolo_ms": round((time.perf_counter() - inicio) * 1000, 3),
            "resolucion_inferencia": f"{frame_inferencia.shape[1]}x{frame_inferencia.shape[0]}",
            "mensaje": "Placa detectada." if detecciones else "No se detecto placa.",
        }

    def detectar(self, ruta_archivo: str) -> dict:
        ruta = Path(ruta_archivo)
        if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": "La deteccion por archivo solo soporta imagenes.",
            }

        imagen = _imread_seguro(str(ruta))
        if imagen is None:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": "No se pudo leer la imagen de entrada.",
            }

        resultado = self.detectar_en_frame(imagen)
        if not resultado["detectada"]:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": resultado["mensaje"],
            }

        mejor = max(resultado["detecciones"], key=lambda item: item["confianza"])
        return {
            "detectada": True,
            "bbox": mejor["bbox"],
            "confianza": mejor["confianza"],
            "mensaje": "Placa detectada por modelo entrenado localmente.",
        }


def dibujar_deteccion(ruta_archivo: str, deteccion: dict, output_dir: str) -> str | None:
    ruta = Path(ruta_archivo)
    if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return None

    imagen = _imread_seguro(str(ruta))
    if imagen is None:
        return None

    bbox = deteccion.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        confianza = float(deteccion.get("confianza", 0.0))
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(imagen, f"placa {confianza:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    salida = Path(output_dir) / f"{ruta.stem}_procesado{ruta.suffix}"
    cv2.imwrite(str(salida), imagen)
    return str(salida)


def _crear_debug(confianza: float, bbox: list[int], aceptada: bool, motivo_rechazo: str) -> dict:
    return {
        "confianza": confianza,
        "bbox": bbox,
        "aceptada": aceptada,
        "motivo_rechazo": motivo_rechazo,
    }


def _preparar_frame_inferencia(frame, imgsz: int | None):
    if not imgsz or imgsz <= 0:
        return frame, 1.0, 1.0
    alto, ancho = frame.shape[:2]
    lado_mayor = max(ancho, alto)
    if lado_mayor <= imgsz:
        return frame, 1.0, 1.0
    escala = float(imgsz) / float(lado_mayor)
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    return redimensionado, nuevo_ancho / ancho, nuevo_alto / alto

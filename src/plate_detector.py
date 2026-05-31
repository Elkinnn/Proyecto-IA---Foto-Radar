from pathlib import Path

import cv2


DEFAULT_MODEL_PATH = "models/plate_detector/placas_ecuador.pt"


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

    def detectar_en_frame(self, frame, conf: float = 0.25) -> dict:
        if self.model is None:
            return {
                "detectada": False,
                "detecciones": [],
                "frame_procesado": frame,
                "mensaje": "Modelo de placa no encontrado. Entrene primero el detector.",
            }

        resultados = self.model.predict(source=frame, conf=conf, verbose=False)
        cajas = resultados[0].boxes if resultados else []
        frame_procesado = frame.copy()
        detecciones = []

        for caja in cajas:
            x1, y1, x2, y2 = [int(valor) for valor in caja.xyxy[0].tolist()]
            confianza = float(caja.conf[0])
            alto, ancho = frame.shape[:2]
            x1 = max(0, min(x1, ancho - 1))
            x2 = max(0, min(x2, ancho - 1))
            y1 = max(0, min(y1, alto - 1))
            y2 = max(0, min(y2, alto - 1))

            if x2 <= x1 or y2 <= y1:
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
                }
            )

        return {
            "detectada": bool(detecciones),
            "detecciones": detecciones,
            "frame_procesado": frame_procesado,
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

        imagen = cv2.imread(str(ruta))
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

    imagen = cv2.imread(str(ruta))
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

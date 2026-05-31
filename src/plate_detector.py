from pathlib import Path

import cv2


class PlateDetector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.estado = "modelo_no_entrenado"
        self._cargar_modelo_si_existe()

    def _cargar_modelo_si_existe(self) -> None:
        if not self.model_path.exists():
            return

        # Punto de integracion futuro. No se descargan pesos ni se usan modelos preentrenados.
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.estado = "modelo_cargado"
        except Exception as exc:
            self.estado = f"error_cargando_modelo: {exc}"
            self.model = None

    def detectar(self, ruta_archivo: str) -> dict:
        ruta = Path(ruta_archivo)
        if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": "La deteccion inicial para video queda preparada para integracion posterior.",
            }

        imagen = cv2.imread(str(ruta))
        if imagen is None:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": "No se pudo leer la imagen de entrada.",
            }

        if self.model is None:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": "Detector de placa aun no entrenado. No se usaron pesos preentrenados.",
            }

        resultados = self.model.predict(source=str(ruta), verbose=False)
        cajas = resultados[0].boxes if resultados else []
        if not cajas:
            return {
                "detectada": False,
                "bbox": None,
                "confianza": 0.0,
                "mensaje": "No se detecto placa.",
            }

        caja = max(cajas, key=lambda box: float(box.conf[0]))
        x1, y1, x2, y2 = [int(valor) for valor in caja.xyxy[0].tolist()]
        return {
            "detectada": True,
            "bbox": [x1, y1, x2, y2],
            "confianza": float(caja.conf[0]),
            "mensaje": "Placa detectada por modelo entrenado localmente.",
        }


def dibujar_deteccion(ruta_archivo: str, deteccion: dict, output_dir: str) -> str | None:
    ruta = Path(ruta_archivo)
    if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        return None

    imagen = cv2.imread(str(ruta))
    if imagen is None:
        return None

    bbox = deteccion.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(imagen, "Placa", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    salida = Path(output_dir) / f"{ruta.stem}_procesado{ruta.suffix}"
    cv2.imwrite(str(salida), imagen)
    return str(salida)

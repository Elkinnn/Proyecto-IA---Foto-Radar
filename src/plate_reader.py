from src.utils import normalizar_placa


class PlateReader:
    def leer(self, deteccion: dict, modo_ocr: str, placa_manual: str | None = None) -> dict:
        if modo_ocr == "manual_controlado":
            placa = normalizar_placa(placa_manual)
            return {
                "texto": placa,
                "modo": modo_ocr,
                "confianza": 1.0 if placa else 0.0,
                "mensaje": "OCR manual/controlado usado temporalmente.",
            }

        if modo_ocr == "automatico":
            return {
                "texto": "",
                "modo": modo_ocr,
                "confianza": 0.0,
                "mensaje": "OCR automatico pendiente de integrar con modelo entrenado desde cero.",
            }

        raise ValueError(f"Modo de OCR no soportado: {modo_ocr}")

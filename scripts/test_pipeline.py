from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import procesar_entrada
from src.utils import cargar_config


def crear_imagen_prueba(ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imagen = np.full((480, 720, 3), 235, dtype=np.uint8)
    cv2.rectangle(imagen, (260, 210), (460, 270), (30, 30, 30), 2)
    cv2.putText(imagen, "PBC1234", (285, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.imwrite(str(ruta), imagen)


def main() -> None:
    config = cargar_config(str(ROOT / "config.yaml"))
    ruta_imagen = ROOT / "data" / "input" / "prueba_pipeline.jpg"
    crear_imagen_prueba(ruta_imagen)

    resultado = procesar_entrada(
        str(ruta_imagen),
        config,
        modo_ocr="manual_controlado",
        placa_manual=config["ocr"]["manual_test_plate"],
        tiempo_segundos=1.2,
    )

    print("Pipeline ejecutado correctamente")
    print(f"Placa: {resultado['texto_placa']}")
    print(f"Velocidad: {resultado['velocidad_kmh']:.2f} km/h")
    print(f"Estado: {resultado['clasificacion_difusa']['estado']}")
    print(f"Sancion: {resultado['sancion_generada']}")
    print(f"Vehiculo: {resultado['vehiculo']}")
    print(f"Reporte: {resultado['ruta_reporte']}")


if __name__ == "__main__":
    main()

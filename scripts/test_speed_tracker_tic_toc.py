from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.speed_estimator import SpeedTracker
import src.pipeline as pipeline


def _bbox_centrada_en(y: float) -> list[int]:
    return [100, int(y - 20), 200, int(y + 20)]


def probar_video_por_fps() -> None:
    tracker = SpeedTracker(450, 650, 10.0, 30.0)
    tracker.actualizar(_bbox_centrada_en(400), 0)
    tracker.actualizar(_bbox_centrada_en(500), 15)
    resultado = tracker.actualizar(_bbox_centrada_en(700), 45)

    assert resultado["fuente_tiempo"] == "frames_fps"
    assert abs(resultado["tiempo_entre_lineas"] - 1.0) < 1e-9
    assert abs(resultado["velocidad_kmh"] - 36.0) < 1e-9


def probar_camara_con_reloj_real() -> None:
    tracker = SpeedTracker(450, 650, 10.0, 30.0)
    tracker.actualizar(_bbox_centrada_en(400), 0, timestamp_segundos=100.0)
    tracker.actualizar(_bbox_centrada_en(500), 15, timestamp_segundos=100.4)
    resultado = tracker.actualizar(_bbox_centrada_en(700), 45, timestamp_segundos=102.4)

    # Cruce L1: t=0.2 s. Cruce L2: t=1.9 s. Delta real: 1.7 s.
    assert resultado["fuente_tiempo"] == "reloj_monotonico"
    assert abs(resultado["tiempo_entre_lineas"] - 1.7) < 1e-9
    assert abs(resultado["velocidad_kmh"] - (10.0 / 1.7 * 3.6)) < 1e-9


def probar_vehiculos_independientes() -> None:
    primero = SpeedTracker(450, 650, 5.0, 30.0)
    segundo = SpeedTracker(450, 650, 5.0, 30.0)

    for tracker, origen in ((primero, 10.0), (segundo, 20.0)):
        tracker.actualizar(_bbox_centrada_en(400), 1, timestamp_segundos=origen)
        tracker.actualizar(_bbox_centrada_en(500), 2, timestamp_segundos=origen + 0.5)
        tracker.actualizar(_bbox_centrada_en(700), 3, timestamp_segundos=origen + 1.5)

    assert primero.velocidad_kmh == segundo.velocidad_kmh
    assert primero.tiempo_cruce_linea_1 != segundo.tiempo_cruce_linea_1 or primero.timestamp_origen != segundo.timestamp_origen


def probar_pipeline_usa_bbox_real_para_cruces() -> None:
    estado = pipeline._crear_estado_persistencia()
    estado["_tiempo_real"] = True
    frame = np.zeros((1000, 1200, 3), dtype=np.uint8)
    recorte = np.zeros((60, 180, 3), dtype=np.uint8)

    originales = (
        pipeline._registrar_candidato_recorte,
        pipeline._actualizar_mejor_evento_placa,
        pipeline._intentar_disparar_ocr_snapshot_evento,
    )
    pipeline._registrar_candidato_recorte = lambda *args, **kwargs: None
    pipeline._actualizar_mejor_evento_placa = lambda *args, **kwargs: None
    pipeline._intentar_disparar_ocr_snapshot_evento = lambda *args, **kwargs: None
    try:
        velocidad = None
        for numero_frame, centro_y, timestamp in (
            (1, 400, 100.0),
            (2, 500, 100.4),
            (3, 700, 102.4),
        ):
            detecciones = [
                {
                    "bbox": [100, centro_y - 20, 280, centro_y + 20],
                    "confianza": 0.90,
                    "recorte_placa": recorte.copy(),
                }
            ]
            _, _, datos = pipeline._actualizar_tracks_multiobjeto(
                estado,
                detecciones,
                frame.copy(),
                frame.copy(),
                numero_frame,
                30.0,
                1,
                3,
                10.0,
                0.45,
                0.65,
                None,
                True,
                timestamp,
            )
            velocidad = datos["velocidad"]
    finally:
        (
            pipeline._registrar_candidato_recorte,
            pipeline._actualizar_mejor_evento_placa,
            pipeline._intentar_disparar_ocr_snapshot_evento,
        ) = originales

    assert velocidad is not None
    assert velocidad["estado"] == "velocidad_calculada"
    assert abs(velocidad["tiempo_entre_lineas"] - 1.7) < 1e-9


def probar_cruce_de_ambas_lineas_entre_detecciones() -> None:
    tracker = SpeedTracker(450, 650, 10.0, 30.0)
    tracker.actualizar(_bbox_centrada_en(400), 1, timestamp_segundos=50.0)
    resultado = tracker.actualizar(_bbox_centrada_en(700), 2, timestamp_segundos=51.5)

    assert resultado["estado"] == "velocidad_calculada"
    assert abs(resultado["tiempo_cruce_linea_1"] - 0.25) < 1e-9
    assert abs(resultado["tiempo_cruce_linea_2"] - 1.25) < 1e-9
    assert abs(resultado["velocidad_kmh"] - 36.0) < 1e-9


def main() -> None:
    probar_video_por_fps()
    probar_camara_con_reloj_real()
    probar_vehiculos_independientes()
    probar_pipeline_usa_bbox_real_para_cruces()
    probar_cruce_de_ambas_lineas_entre_detecciones()
    print("SpeedTracker tic-toc")
    print("--------------------")
    print("Video grabado: usa frames/FPS")
    print("Camara en vivo: usa reloj monotónico real")
    print("Seguimiento multiobjeto: trackers independientes")
    print("Pipeline: cruces calculados con bbox real, no bbox visual suavizada")
    print("Vehiculo rapido: permite TIC y TOC entre dos detecciones")
    print("Resultado: OK")


if __name__ == "__main__":
    main()

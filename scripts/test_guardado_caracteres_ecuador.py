from __future__ import annotations

from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generar_caracteres_desde_placas_ecuador import guardar_caracteres  # noqa: E402


TMP_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "test_guardado_caracteres_ecuador"
PLACA = "GPA1527"
ESPERADAS = list(PLACA)


def crear_caracter_fake(indice: int, etiqueta: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    imagen = np.zeros((48, 32), dtype=np.uint8)
    cv2.putText(imagen, etiqueta, (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
    ruta = TMP_DIR / f"segmento_{indice:02d}_{etiqueta}.png"
    cv2.imwrite(str(ruta), imagen)
    return ruta


def main() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    caracteres = []
    for indice, etiqueta in enumerate(ESPERADAS, start=1):
        ruta = crear_caracter_fake(indice, etiqueta)
        caracteres.append({"ruta_caracter": str(ruta), "bbox": [0, 0, 32, 48]})

    ruta_original = ROOT_DIR / "datasets" / "placas_ecuador" / "GPA1527_test.jpg"
    filas_labels = []
    existentes = set()
    cantidad, rutas = guardar_caracteres(
        ruta_original,
        PLACA,
        caracteres,
        filas_labels,
        existentes,
        dry_run=True,
        indices_guardar=list(range(len(caracteres))),
    )

    assert cantidad == len(ESPERADAS), f"Se esperaban {len(ESPERADAS)} rutas, se obtuvieron {cantidad}"
    for etiqueta, ruta_txt in zip(ESPERADAS, rutas):
        ruta = Path(ruta_txt)
        assert ruta.parts[-2] == etiqueta, f"Etiqueta incorrecta: {ruta_txt}; se esperaba carpeta {etiqueta}"
        assert ruta.name.startswith(f"{etiqueta}_ecuador_"), f"Nombre incorrecto para {etiqueta}: {ruta.name}"
        assert ruta.suffix == ".png", f"Extension incorrecta: {ruta.name}"

    cantidad_repetida, rutas_repetidas = guardar_caracteres(
        ruta_original,
        PLACA,
        caracteres,
        filas_labels,
        set(rutas),
        dry_run=True,
        indices_guardar=list(range(len(caracteres))),
    )
    assert cantidad_repetida == 0, "No se deben generar duplicados para la misma imagen, placa e indice."
    assert rutas_repetidas == [], "Las rutas repetidas debieron bloquearse."

    try:
        guardar_caracteres(
            ruta_original,
            PLACA,
            caracteres,
            filas_labels,
            set(),
            dry_run=True,
            indices_guardar=list(range(len(caracteres) - 1)),
        )
    except ValueError as exc:
        assert "cantidad segmentada no coincide" in str(exc)
    else:
        raise AssertionError("Debio bloquearse el guardado si faltan caracteres seleccionados.")

    print("Prueba guardado caracteres Ecuador")
    print("----------------------------------")
    print(f"Placa: {PLACA}")
    print(f"Rutas verificadas: {cantidad}")
    print("Asignacion esperada: G -> P -> A -> 1 -> 5 -> 2 -> 7")
    print("Resultado: OK")


if __name__ == "__main__":
    main()

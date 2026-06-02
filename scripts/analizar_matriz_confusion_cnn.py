from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_reader import cargar_modelo_caracteres, normalizar_caracter_para_cnn  # noqa: E402


TEST_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador_split" / "test"
REPORTE_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "character_cnn"
REPORTE_PATH = REPORTE_DIR / "analisis_confusiones.txt"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}
PARES_OBJETIVO = {("O", "0"), ("B", "8"), ("4", "2"), ("P", "B"), ("F", "6"), ("H", "M")}
BATCH_SIZE = 128


def listar_test() -> list[tuple[Path, str]]:
    ejemplos = []
    if not TEST_DIR.exists():
        return ejemplos
    for clase_dir in sorted(TEST_DIR.iterdir()):
        if not clase_dir.is_dir():
            continue
        for ruta in sorted(clase_dir.rglob("*")):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES:
                ejemplos.append((ruta, clase_dir.name))
    return ejemplos


def recomendacion(real: str, pred: str) -> str:
    par = {real, pred}
    if par == {"O", "0"}:
        return "Revisar postprocesamiento por posicion y agregar ejemplos de O/0 con tipografia de placas."
    if par == {"B", "8"}:
        return "Reforzar B/8 con caracteres reales y augmentacion de blur/contraste."
    if par == {"4", "2"}:
        return "Revisar segmentacion de diagonales y reforzar numeros 4/2 reales."
    if par == {"P", "B"}:
        return "Comparar si P queda cerrada por binarizacion; reforzar P real y ajustar normalizacion."
    if par == {"F", "6"}:
        return "Verificar que F no se deforme al centrar; reforzar F real con bajo contraste."
    if par == {"H", "M"}:
        return "Revisar anchura/aspecto del recorte; H/M suelen confundirse si el resize distorsiona."
    return "Inspeccionar ejemplos fallidos y comparar con dataset base antes de reentrenar."


def analizar() -> dict:
    carga, mensaje = cargar_modelo_caracteres()
    if carga is None:
        raise RuntimeError(mensaje)
    modelo, class_names = carga
    ejemplos = listar_test()
    if not ejemplos:
        raise RuntimeError(f"No hay imagenes de test en {TEST_DIR.relative_to(ROOT_DIR)}")

    confusiones = Counter()
    total = 0
    errores = 0
    batch_arrays = []
    batch_labels = []
    for ruta, etiqueta_real in ejemplos:
        normalizada = normalizar_caracter_para_cnn(str(ruta))
        if normalizada.get("array") is None:
            continue
        batch_arrays.append(normalizada["array"][0])
        batch_labels.append(etiqueta_real)

        if len(batch_arrays) >= BATCH_SIZE:
            total, errores = _evaluar_batch(modelo, class_names, batch_arrays, batch_labels, confusiones, total, errores)
            batch_arrays = []
            batch_labels = []

    if batch_arrays:
        total, errores = _evaluar_batch(modelo, class_names, batch_arrays, batch_labels, confusiones, total, errores)

    REPORTE_DIR.mkdir(parents=True, exist_ok=True)
    lineas = [
        "Analisis de confusiones CNN OCR",
        "--------------------------------",
        f"Total test evaluado: {total}",
        f"Errores: {errores}",
        f"Tasa de error: {(errores / total * 100) if total else 0:.2f}%",
        "",
        "Pares mas confundidos:",
    ]

    for (real, pred), cantidad in confusiones.most_common(30):
        porcentaje = cantidad / max(errores, 1) * 100
        objetivo = " [par observado en placas reales]" if (real, pred) in PARES_OBJETIVO or (pred, real) in PARES_OBJETIVO else ""
        lineas.append(
            f"- {real} -> {pred}: {cantidad} errores ({porcentaje:.2f}% de errores){objetivo}"
        )
        lineas.append(f"  Recomendacion: {recomendacion(real, pred)}")

    if not confusiones:
        lineas.append("- Sin confusiones en el split de test.")

    lineas.extend(
        [
            "",
            "Lectura practica:",
            "- Si el test casi no muestra errores, pero los recortes reales fallan, el problema es diferencia de dominio, segmentacion o normalizacion.",
            "- Si aparecen pares P/B, F/6, H/M o 4/B, conviene reforzar esas clases con ejemplos ecuatorianos reales.",
        ]
    )
    REPORTE_PATH.write_text("\n".join(lineas), encoding="utf-8")
    return {"total": total, "errores": errores, "confusiones": len(confusiones)}


def _evaluar_batch(modelo, class_names: list, arrays: list, labels: list, confusiones: Counter, total: int, errores: int) -> tuple[int, int]:
    entradas = np.asarray(arrays, dtype="float32")
    predicciones = modelo.predict(entradas, verbose=0)
    indices = np.argmax(predicciones, axis=1)
    for etiqueta_real, indice in zip(labels, indices):
        total += 1
        etiqueta_pred = str(class_names[int(indice)])
        if etiqueta_pred != etiqueta_real:
            errores += 1
            confusiones[(etiqueta_real, etiqueta_pred)] += 1
    return total, errores


def main() -> None:
    try:
        resultado = analizar()
    except Exception as exc:
        print(f"Error analizando matriz de confusion CNN: {exc}")
        sys.exit(1)

    print("Analisis de confusiones CNN OCR")
    print("--------------------------------")
    print(f"Total test evaluado: {resultado['total']}")
    print(f"Errores: {resultado['errores']}")
    print(f"Reporte: {REPORTE_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

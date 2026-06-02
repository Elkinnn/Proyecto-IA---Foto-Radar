from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_reader import cargar_modelo_caracteres, predecir_caracter  # noqa: E402


CSV_DIAGNOSTICO = ROOT_DIR / "datasets" / "diagnostico_caracteres_reales.csv"
CARACTERES_SEGMENTADOS_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "caracteres_segmentados"
REPORTE_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "diagnostico_cnn_caracteres"
REPORTE_CSV = REPORTE_DIR / "reporte_diagnostico_cnn.csv"
RESUMEN_TXT = REPORTE_DIR / "resumen_diagnostico_cnn.txt"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}


def leer_csv_diagnostico() -> tuple[list[dict], bool]:
    if not CSV_DIAGNOSTICO.exists():
        return [], False

    filas = []
    with open(CSV_DIAGNOSTICO, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            ruta = fila.get("ruta_caracter", "")
            etiqueta = (fila.get("etiqueta_esperada") or "").strip().upper()
            if ruta:
                filas.append({"ruta_caracter": ROOT_DIR / ruta, "etiqueta_esperada": etiqueta})
    return filas, True


def buscar_caracteres_recientes() -> list[dict]:
    if not CARACTERES_SEGMENTADOS_DIR.exists():
        return []
    rutas = sorted(
        (
            ruta
            for ruta in CARACTERES_SEGMENTADOS_DIR.rglob("*")
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
        ),
        key=lambda ruta: ruta.stat().st_mtime,
        reverse=True,
    )
    return [{"ruta_caracter": ruta, "etiqueta_esperada": ""} for ruta in rutas[:200]]


def diagnosticar() -> dict:
    carga, mensaje = cargar_modelo_caracteres()
    if carga is None:
        raise RuntimeError(mensaje)
    modelo, class_names = carga

    entradas, tiene_etiquetas = leer_csv_diagnostico()
    if not entradas:
        entradas = buscar_caracteres_recientes()

    REPORTE_DIR.mkdir(parents=True, exist_ok=True)
    filas_reporte = []
    total_con_etiqueta = 0
    aciertos = 0
    errores = 0

    for entrada in entradas:
        ruta = Path(entrada["ruta_caracter"])
        etiqueta = entrada.get("etiqueta_esperada", "")
        observacion = ""
        if not ruta.is_absolute():
            ruta = ROOT_DIR / ruta

        if not ruta.exists():
            filas_reporte.append(
                _fila(ruta, etiqueta, "", 0.0, [], "", "no", "ruta no existe")
            )
            continue

        pred = predecir_caracter(str(ruta), modelo, class_names)
        prediccion = pred.get("caracter_predicho", "")
        confianza = float(pred.get("confianza", 0.0))
        acierto = ""
        if etiqueta:
            total_con_etiqueta += 1
            acierto = "si" if prediccion == etiqueta else "no"
            if acierto == "si":
                aciertos += 1
            else:
                errores += 1
                observacion = f"esperado {etiqueta}, predicho {prediccion}"
        elif not tiene_etiquetas:
            observacion = "Sin CSV de etiquetas; no se puede calcular acierto."

        filas_reporte.append(
            _fila(
                ruta,
                etiqueta,
                prediccion,
                confianza,
                pred.get("top3_predicciones", []),
                pred.get("ruta_debug_normalizada", ""),
                acierto,
                observacion,
            )
        )

    with open(REPORTE_CSV, "w", newline="", encoding="utf-8") as archivo:
        campos = [
            "ruta_caracter",
            "etiqueta_esperada",
            "prediccion",
            "confianza",
            "top3",
            "acierto",
            "ruta_debug_normalizada",
            "observacion",
        ]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas_reporte)

    accuracy = aciertos / total_con_etiqueta if total_con_etiqueta else None
    lineas = [
        "Diagnostico CNN sobre caracteres reales",
        "---------------------------------------",
        f"Entradas evaluadas: {len(filas_reporte)}",
        f"CSV de etiquetas: {'si' if tiene_etiquetas else 'no'}",
        f"Caracteres con etiqueta: {total_con_etiqueta}",
        f"Aciertos: {aciertos}",
        f"Errores: {errores}",
        f"Accuracy etiquetado: {accuracy:.4f}" if accuracy is not None else "Accuracy etiquetado: no disponible",
        "",
        "Observacion:",
        "Si las imagenes debug normalizadas se ven deformadas, el problema esta en normalizacion.",
        "Si se ven bien pero la prediccion falla, probablemente hay diferencia de dominio o clases insuficientes.",
        "",
        f"Reporte CSV: {REPORTE_CSV.relative_to(ROOT_DIR)}",
    ]
    RESUMEN_TXT.write_text("\n".join(lineas), encoding="utf-8")
    return {
        "total": len(filas_reporte),
        "tiene_etiquetas": tiene_etiquetas,
        "aciertos": aciertos,
        "errores": errores,
        "accuracy": accuracy,
    }


def _fila(
    ruta: Path,
    etiqueta: str,
    prediccion: str,
    confianza: float,
    top3: list,
    ruta_debug: str,
    acierto: str,
    observacion: str,
) -> dict:
    try:
        ruta_rel = str(ruta.relative_to(ROOT_DIR))
    except ValueError:
        ruta_rel = str(ruta)
    return {
        "ruta_caracter": ruta_rel,
        "etiqueta_esperada": etiqueta,
        "prediccion": prediccion,
        "confianza": f"{confianza:.6f}",
        "top3": json.dumps(top3, ensure_ascii=False),
        "acierto": acierto,
        "ruta_debug_normalizada": ruta_debug,
        "observacion": observacion,
    }


def main() -> None:
    try:
        resultado = diagnosticar()
    except Exception as exc:
        print(f"Error diagnosticando CNN de caracteres reales: {exc}")
        sys.exit(1)

    print("Diagnostico CNN caracteres reales")
    print("--------------------------------")
    print(f"Entradas evaluadas: {resultado['total']}")
    print(f"CSV de etiquetas: {'si' if resultado['tiene_etiquetas'] else 'no'}")
    if resultado["accuracy"] is not None:
        print(f"Accuracy etiquetado: {resultado['accuracy']:.4f}")
    else:
        print("Accuracy etiquetado: no disponible")
    print(f"Reporte: {REPORTE_CSV.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

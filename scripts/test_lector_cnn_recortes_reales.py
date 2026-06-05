from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plate_reader import leer_placa_desde_recorte


FUENTES_RECORTE = [
    Path("reports/evidencias/placas_capturadas"),
    Path("reports/evidencias/mejores_recortes_placa"),
    Path("reports/evidencias/video_anotado/mejores_recortes"),
    Path("reports/evidencias/eventos_placa"),
]
SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_lector_cnn_recortes_reales")
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def buscar_recortes() -> list[Path]:
    rutas: list[Path] = []
    for carpeta in FUENTES_RECORTE:
        if not carpeta.exists():
            continue
        for ruta in carpeta.rglob("*"):
            nombre = ruta.name.lower()
            es_recorte = "recorte" in nombre or "placa" in nombre
            es_frame_completo = "mejor_frame_" in nombre or "frame_bbox" in nombre
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES and es_recorte and not es_frame_completo:
                rutas.append(ruta)
    return sorted(set(rutas))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba robustez del lector CNN sobre recortes reales.")
    parser.add_argument("--limit", type=int, default=20, help="Cantidad maxima de recortes a evaluar. 0 procesa todos.")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    recortes = buscar_recortes()
    if args.limit and args.limit > 0:
        recortes = recortes[: args.limit]

    filas = []
    resultados = []
    for ruta in recortes:
        resultado = leer_placa_desde_recorte(str(ruta), placa_esperada="")
        fila = {
            "archivo": str(ruta),
            "ok": bool(resultado.get("ok")),
            "estado_lectura": resultado.get("estado_lectura", ""),
            "motivo_sin_lectura": resultado.get("motivo_sin_lectura") or resultado.get("motivo", ""),
            "cantidad_caracteres_segmentados": resultado.get("cantidad_caracteres_segmentados", 0),
            "texto_crudo": resultado.get("texto_detectado_crudo") or resultado.get("texto_crudo", ""),
            "texto_corregido": resultado.get("texto_postprocesado") or resultado.get("texto_corregido_formato", ""),
            "confianza": resultado.get("confianza_promedio") or resultado.get("confianza_cnn_caracteres", 0.0),
            "formato_valido": bool(resultado.get("formato_valido")),
            "estado": resultado.get("estado", ""),
            "causa_probable": resultado.get("causa_probable", ""),
            "error": resultado.get("error", ""),
            "etapa_error": resultado.get("etapa_error", ""),
            "estrategia_segmentacion": (resultado.get("segmentacion") or {}).get("estrategia_segmentacion", ""),
            "puntaje_segmentacion": (resultado.get("segmentacion") or {}).get("puntaje_segmentacion", ""),
            "ruta_debug_segmentacion": resultado.get("ruta_debug_segmentacion", ""),
            "ruta_banda": resultado.get("ruta_banda", ""),
        }
        filas.append(fila)
        resultados.append(
            {
                **fila,
                "predicciones_caracteres": resultado.get("predicciones_caracteres", []),
                "diagnostico": resultado.get("diagnostico", {}),
                "debug_images": resultado.get("debug_images", {}),
            }
        )
        estado = "OK" if fila["ok"] else "ERROR"
        print(
            f"{estado} | {ruta.name} | chars={fila['cantidad_caracteres_segmentados']} | "
            f"crudo={fila['texto_crudo']} | corregido={fila['texto_corregido']} | "
            f"etapa={fila['etapa_error'] or '-'}"
        )

    csv_path = SALIDA / "reporte_lector_cnn_recortes_reales.csv"
    json_path = SALIDA / "reporte_lector_cnn_recortes_reales.json"
    campos = [
        "archivo",
            "ok",
            "estado_lectura",
            "motivo_sin_lectura",
            "cantidad_caracteres_segmentados",
        "texto_crudo",
        "texto_corregido",
        "confianza",
        "formato_valido",
        "estado",
        "causa_probable",
        "error",
        "etapa_error",
        "estrategia_segmentacion",
        "puntaje_segmentacion",
        "ruta_debug_segmentacion",
        "ruta_banda",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    json_path.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    errores = [fila for fila in filas if not fila["ok"]]
    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {len(filas)}")
    print(f"Errores lector CNN: {len(errores)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

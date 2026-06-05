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
]
SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_segmentacion_placas_reales")
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
    parser = argparse.ArgumentParser(description="Prueba segmentacion y lector CNN sobre recortes reales de placa.")
    parser.add_argument("--limit", type=int, default=0, help="Cantidad maxima de recortes a evaluar. 0 procesa todos.")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    recortes = buscar_recortes()
    if args.limit and args.limit > 0:
        recortes = recortes[: args.limit]
    filas = []
    resultados_json = []

    for ruta in recortes:
        resultado = leer_placa_desde_recorte(str(ruta), placa_esperada="")
        segmentacion = resultado.get("segmentacion") or {}
        predicciones = resultado.get("predicciones_caracteres") or []
        fila = {
            "archivo": str(ruta),
            "cantidad_caracteres_detectados": resultado.get("cantidad_caracteres_segmentados", 0),
            "texto_crudo": resultado.get("texto_detectado_crudo", ""),
            "texto_final": resultado.get("texto_postprocesado", ""),
            "confianza_promedio": resultado.get("confianza_promedio", 0.0),
            "formato_valido": (resultado.get("formato") or {}).get("valido", False),
            "causa_probable": resultado.get("causa_probable", ""),
            "causa_probable_segmentacion": segmentacion.get("causa_probable_segmentacion", ""),
            "ruta_banda": resultado.get("ruta_banda", ""),
            "ruta_debug_segmentacion": resultado.get("ruta_debug_segmentacion", ""),
            "caracteres_segmentados": " ".join([p.get("caracter_predicho", "") for p in predicciones]),
        }
        filas.append(fila)
        resultados_json.append(
            {
                **fila,
                "predicciones": predicciones,
                "diagnostico_segmentacion": segmentacion.get("diagnostico_segmentacion", {}),
                "caracteres": resultado.get("caracteres", []),
            }
        )
        print(
            f"{ruta.name}: chars={fila['cantidad_caracteres_detectados']} "
            f"crudo={fila['texto_crudo']} final={fila['texto_final']} "
            f"causa={fila['causa_probable']}"
        )

    csv_path = SALIDA / "reporte_segmentacion_placas_reales.csv"
    json_path = SALIDA / "reporte_segmentacion_placas_reales.json"
    campos = [
        "archivo",
        "cantidad_caracteres_detectados",
        "texto_crudo",
        "texto_final",
        "confianza_promedio",
        "formato_valido",
        "causa_probable",
        "causa_probable_segmentacion",
        "ruta_banda",
        "ruta_debug_segmentacion",
        "caracteres_segmentados",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    json_path.write_text(json.dumps(resultados_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {len(recortes)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

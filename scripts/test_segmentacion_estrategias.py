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


FUENTES = [
    Path("reports/evidencias/reconocimiento_caracteres/sin_lectura_debug"),
    Path("reports/evidencias/mejores_recortes_placa"),
    Path("reports/evidencias/placas_capturadas"),
]
SALIDA = Path("reports/evidencias/reconocimiento_caracteres/comparacion_segmentacion")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def buscar_recortes() -> list[Path]:
    rutas = []
    for carpeta in FUENTES:
        if not carpeta.exists():
            continue
        for ruta in carpeta.rglob("*"):
            nombre = ruta.name.lower()
            if not ruta.is_file() or ruta.suffix.lower() not in EXTS:
                continue
            if "debug" in nombre or "banda" in nombre or "preprocesada" in nombre or "caracter_" in nombre:
                continue
            if "recorte" in nombre or "placa" in nombre:
                rutas.append(ruta)
    return sorted(set(rutas))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara estrategias de segmentacion sobre recortes reales.")
    parser.add_argument("--limit", type=int, default=50, help="Cantidad maxima de recortes. 0 procesa todos.")
    args = parser.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)

    recortes = buscar_recortes()
    if args.limit and args.limit > 0:
        recortes = recortes[: args.limit]

    filas = []
    detalles = []
    for ruta in recortes:
        resultado = leer_placa_desde_recorte(str(ruta), metodo_segmentacion="v3_estrategias")
        segmentacion = resultado.get("segmentacion") or {}
        fila = {
            "archivo": str(ruta),
            "cantidad_caracteres": resultado.get("cantidad_caracteres_segmentados", 0),
            "rango_valido_6_7": 6 <= int(resultado.get("cantidad_caracteres_segmentados", 0) or 0) <= 7,
            "texto_crudo": resultado.get("texto_crudo", ""),
            "texto_corregido": resultado.get("texto_corregido_formato", ""),
            "estado_lectura": resultado.get("estado_lectura", ""),
            "estrategia": segmentacion.get("estrategia_segmentacion", ""),
            "puntaje": segmentacion.get("puntaje_segmentacion", ""),
            "ruta_debug": resultado.get("ruta_debug_segmentacion", ""),
            "detalle_estrategias": segmentacion.get("detalle_estrategias", ""),
        }
        filas.append(fila)
        detalles.append({"fila": fila, "segmentacion": segmentacion, "predicciones": resultado.get("predicciones_caracteres", [])})
        print(f"{ruta.name}: chars={fila['cantidad_caracteres']} estrategia={fila['estrategia']} texto={fila['texto_corregido']}")

    csv_path = SALIDA / "resumen_estrategias.csv"
    json_path = SALIDA / "detalle_estrategias.json"
    campos = [
        "archivo",
        "cantidad_caracteres",
        "rango_valido_6_7",
        "texto_crudo",
        "texto_corregido",
        "estado_lectura",
        "estrategia",
        "puntaje",
        "ruta_debug",
        "detalle_estrategias",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    json_path.write_text(json.dumps(detalles, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(filas)
    validos = sum(1 for fila in filas if fila["rango_valido_6_7"])
    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {total}")
    print(f"Con 6 o 7 caracteres: {validos}")
    print(f"Porcentaje 6/7: {(validos / total * 100 if total else 0):.2f}%")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

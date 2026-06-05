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
from src.utils import normalizar_placa


SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_segmentacion_slots_ecuador")
FUENTES = [
    Path("reports/evidencias/mejores_recortes_placa"),
    Path("reports/evidencias/reconocimiento_caracteres/sin_lectura_debug"),
    Path("reports/evidencias/placas_capturadas"),
    Path("reports/evidencias/eventos_placa"),
]


def _buscar_recortes(limit: int) -> list[Path]:
    extensiones = {".jpg", ".jpeg", ".png", ".bmp"}
    rutas: list[Path] = []
    for carpeta in FUENTES:
        if not carpeta.exists():
            continue
        for ruta in carpeta.rglob("*"):
            if ruta.is_file() and ruta.suffix.lower() in extensiones:
                rutas.append(ruta)
    rutas = sorted(rutas, key=lambda item: item.stat().st_mtime, reverse=True)
    return rutas[:limit]


def _comparar(esperado: str, detectado: str) -> str:
    esperado_norm = normalizar_placa(esperado)
    detectado_norm = normalizar_placa(detectado)
    if not esperado_norm:
        return "sin_esperado"
    return "OK" if esperado_norm == detectado_norm else "NO"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostica segmentacion slots Ecuador para el lector CNN.")
    parser.add_argument("--expected", default="", help="Placa esperada para comparar, ejemplo ABC0123.")
    parser.add_argument("--limit", type=int, default=10, help="Cantidad maxima de recortes recientes a evaluar.")
    parser.add_argument("--image", default="", help="Recorte especifico opcional.")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    recortes = [Path(args.image)] if args.image else _buscar_recortes(args.limit)
    filas = []
    if not recortes:
        print("No se encontraron recortes para evaluar.")
        return

    for ruta in recortes:
        if not ruta.exists():
            print(f"No existe: {ruta}")
            continue
        resultado = leer_placa_desde_recorte(str(ruta), placa_esperada=args.expected)
        segmentacion = resultado.get("segmentacion") or {}
        texto_crudo = resultado.get("texto_detectado_crudo") or resultado.get("texto_crudo") or ""
        texto_corr = resultado.get("texto_postprocesado") or resultado.get("texto_corregido_formato") or ""
        fila = {
            "ruta": str(ruta),
            "slots_generados": segmentacion.get("slots_utiles_generados", 0),
            "slots_dudosos": segmentacion.get("cantidad_slots_dudosos", 0),
            "texto_crudo": texto_crudo,
            "texto_corregido": texto_corr,
            "estrategia_ganadora": segmentacion.get("estrategia_segmentacion") or segmentacion.get("metodo"),
            "puntaje_segmentacion": segmentacion.get("puntaje_segmentacion"),
            "puntaje_slots": segmentacion.get("puntaje_slots_ecuador"),
            "slots_ecuador_aceptado": segmentacion.get("slots_ecuador_aceptado"),
            "motivo_rechazo_slots": segmentacion.get("motivo_rechazo_slots_ecuador"),
            "debug_slots": segmentacion.get("debug_slots_ecuador_path"),
            "comparacion": _comparar(args.expected, texto_corr),
        }
        filas.append(fila)
        print(
            f"{ruta.name}: slots={fila['slots_generados']} dudosos={fila['slots_dudosos']} "
            f"crudo={texto_crudo or '-'} corregido={texto_corr or '-'} "
            f"estrategia={fila['estrategia_ganadora']} puntaje_slots={fila['puntaje_slots']} "
            f"aceptado={fila['slots_ecuador_aceptado']} comparacion={fila['comparacion']}"
        )

    ruta_csv = SALIDA / "segmentacion_slots_ecuador.csv"
    ruta_json = SALIDA / "segmentacion_slots_ecuador.json"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    ruta_json.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {len(filas)}")
    print(f"CSV: {ruta_csv}")
    print(f"JSON: {ruta_json}")


if __name__ == "__main__":
    main()

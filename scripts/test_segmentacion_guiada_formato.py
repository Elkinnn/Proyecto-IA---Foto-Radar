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
    Path("reports/evidencias/placas_capturadas"),
    Path("reports/evidencias/mejores_recortes_placa"),
    Path("reports/evidencias/eventos_placa"),
    Path("reports/evidencias/reconocimiento_caracteres/sin_lectura_debug"),
]
SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_segmentacion_guiada_formato")
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


def comparar_posiciones(esperada: str, detectada: str) -> list[dict]:
    esperada = "".join(ch for ch in (esperada or "").upper() if ch.isalnum())
    detectada = "".join(ch for ch in (detectada or "").upper() if ch.isalnum())
    salida = []
    for idx in range(max(len(esperada), len(detectada))):
        exp = esperada[idx] if idx < len(esperada) else ""
        det = detectada[idx] if idx < len(detectada) else ""
        salida.append({"posicion": idx + 1, "esperado": exp, "detectado": det, "acierto": bool(exp and exp == det)})
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba segmentacion guiada por formato ecuatoriano.")
    parser.add_argument("--image", default="", help="Recorte especifico de placa.")
    parser.add_argument("--expected", default="ABC0123", help="Placa esperada solo para diagnostico.")
    parser.add_argument("--limit", type=int, default=10, help="Cantidad maxima si no se pasa --image.")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    rutas = [Path(args.image)] if args.image else buscar_recortes()
    if not args.image and args.limit and args.limit > 0:
        rutas = rutas[: args.limit]

    filas = []
    detalles = []
    for ruta in rutas:
        resultado = leer_placa_desde_recorte(str(ruta), placa_esperada=args.expected, metodo_segmentacion="v3_estrategias")
        segmentacion = resultado.get("segmentacion") or {}
        texto = resultado.get("texto_corregido_formato") or resultado.get("texto_postprocesado") or resultado.get("texto_crudo") or ""
        fila = {
            "archivo": str(ruta),
            "placa_esperada": args.expected,
            "texto_crudo": resultado.get("texto_crudo") or resultado.get("texto_detectado_crudo") or "",
            "texto_corregido": texto,
            "acierto": texto == "".join(ch for ch in args.expected.upper() if ch.isalnum()),
            "estado_lectura": resultado.get("estado_lectura", ""),
            "cantidad_caracteres": resultado.get("cantidad_caracteres_segmentados", 0),
            "estrategia_ganadora": segmentacion.get("estrategia_segmentacion", ""),
            "uso_segmentacion_guiada": bool(segmentacion.get("segmentacion_guiada_formato")),
            "guion_descartado": bool(segmentacion.get("guion_descartado")),
            "puntaje_segmentacion": segmentacion.get("puntaje_segmentacion", ""),
            "ruta_banda": resultado.get("ruta_banda", ""),
            "ruta_debug_segmentacion": resultado.get("ruta_debug_segmentacion", ""),
            "detalle_estrategias": segmentacion.get("detalle_estrategias", ""),
            "ruta_rectificada": (resultado.get("rectificacion") or {}).get("ruta_seleccionada", ""),
        }
        filas.append(fila)
        detalles.append(
            {
                "fila": fila,
                "comparacion_posiciones": comparar_posiciones(args.expected, texto),
                "predicciones_caracteres": resultado.get("predicciones_caracteres", []),
                "segmentacion": segmentacion,
            }
        )
        print(
            f"{ruta.name}: esperado={args.expected} obtenido={texto} "
            f"chars={fila['cantidad_caracteres']} estrategia={fila['estrategia_ganadora']} "
            f"guiada={fila['uso_segmentacion_guiada']}"
        )

    csv_path = SALIDA / "segmentacion_guiada_formato.csv"
    json_path = SALIDA / "segmentacion_guiada_formato.json"
    campos = list(filas[0].keys()) if filas else [
        "archivo",
        "placa_esperada",
        "texto_crudo",
        "texto_corregido",
        "acierto",
        "estado_lectura",
        "cantidad_caracteres",
        "estrategia_ganadora",
        "uso_segmentacion_guiada",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    json_path.write_text(json.dumps(detalles, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {len(filas)}")
    print(f"Uso segmentacion guiada: {sum(1 for fila in filas if fila['uso_segmentacion_guiada'])}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

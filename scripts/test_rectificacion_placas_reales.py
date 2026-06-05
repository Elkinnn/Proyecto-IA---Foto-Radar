from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plate_reader import evaluar_calidad_lectura_evento, consolidar_lecturas_evento_placa, guardar_debug_votacion_evento, leer_placa_desde_recorte


FUENTES = [
    Path("reports/evidencias/placas_capturadas"),
    Path("reports/evidencias/mejores_recortes_placa"),
    Path("reports/evidencias/reconocimiento_caracteres/sin_lectura_debug"),
]
SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_rectificacion_placas_reales")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def buscar_recortes() -> list[Path]:
    rutas: list[Path] = []
    for carpeta in FUENTES:
        if not carpeta.exists():
            continue
        for ruta in carpeta.rglob("*"):
            nombre = ruta.name.lower()
            if not ruta.is_file() or ruta.suffix.lower() not in EXTS:
                continue
            if "frame_bbox" in nombre or "mejor_frame" in nombre or "debug" in nombre:
                continue
            if "recorte" in nombre or "placa" in nombre:
                rutas.append(ruta)
    return sorted(set(rutas))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba rectificacion y lector CNN sobre recortes reales.")
    parser.add_argument("--limit", type=int, default=30, help="Cantidad maxima de recortes. Use 0 para todos.")
    parser.add_argument("--expected", default="", help="Placa esperada opcional para todos los recortes.")
    parser.add_argument("--consolidar-eventos", action="store_true", help="Agrupa recortes parecidos y aplica votacion temporal.")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    recortes = buscar_recortes()
    if args.limit and args.limit > 0:
        recortes = recortes[: args.limit]

    filas = []
    detalles = []
    for ruta in recortes:
        resultado = leer_placa_desde_recorte(str(ruta), placa_esperada=args.expected)
        rectificacion = resultado.get("rectificacion") or {}
        fila = {
            "archivo": str(ruta),
            "metodo_rectificacion": rectificacion.get("metodo_rectificacion") or resultado.get("metodo_rectificacion"),
            "puntaje_rectificacion": rectificacion.get("confianza_rectificacion") or resultado.get("puntaje_rectificacion"),
            "cantidad_caracteres_segmentados": resultado.get("cantidad_caracteres_segmentados", 0),
            "texto_crudo": resultado.get("texto_detectado_crudo") or resultado.get("texto_crudo", ""),
            "texto_corregido": resultado.get("texto_postprocesado") or resultado.get("texto_corregido_formato", ""),
            "confianza_cnn": resultado.get("confianza_promedio") or resultado.get("confianza_cnn_caracteres", 0.0),
            "estado_lectura": resultado.get("estado_lectura", ""),
            "estrategia_segmentacion": resultado.get("estrategia_segmentacion") or (resultado.get("segmentacion") or {}).get("estrategia_segmentacion"),
            "puntaje_segmentacion": resultado.get("puntaje_segmentacion") or (resultado.get("segmentacion") or {}).get("puntaje_segmentacion"),
            "ruta_rectificada": rectificacion.get("ruta_seleccionada", ""),
            "ruta_banda": resultado.get("ruta_banda", ""),
            "ruta_debug_segmentacion": resultado.get("ruta_debug_segmentacion", ""),
        }
        fila["grupo_evento"] = _grupo_evento_desde_ruta(ruta)
        filas.append(fila)
        detalles.append({"fila": fila, "resultado": resultado, "lectura_evento": _lectura_para_consolidar(fila, resultado)})
        print(
            f"{ruta.name}: rect={fila['metodo_rectificacion']} puntaje_rect={fila['puntaje_rectificacion']} "
            f"chars={fila['cantidad_caracteres_segmentados']} estado={fila['estado_lectura']} "
            f"texto={fila['texto_corregido'] or fila['texto_crudo']}"
        )

    csv_path = SALIDA / "rectificacion_placas_reales.csv"
    json_path = SALIDA / "rectificacion_placas_reales.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as archivo:
        campos = list(filas[0].keys()) if filas else [
            "archivo",
            "metodo_rectificacion",
            "puntaje_rectificacion",
            "cantidad_caracteres_segmentados",
            "texto_crudo",
            "texto_corregido",
            "confianza_cnn",
            "estado_lectura",
        ]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    json_path.write_text(json.dumps(detalles, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.consolidar_eventos:
        consolidados = []
        grupos = {}
        for detalle in detalles:
            grupos.setdefault(detalle["fila"]["grupo_evento"], []).append(detalle["lectura_evento"])
        for grupo, lecturas in grupos.items():
            consolidado = consolidar_lecturas_evento_placa(lecturas)
            ruta_debug = guardar_debug_votacion_evento(grupo, lecturas, consolidado)
            consolidados.append(
                {
                    "grupo_evento": grupo,
                    "lecturas_individuales": [lectura.get("texto_corregido_formato") or lectura.get("texto_crudo") for lectura in lecturas],
                    "lectura_final_consolidada": consolidado.get("texto_final"),
                    "confianza_final": consolidado.get("confianza_final"),
                    "cantidad_recortes_usados": len(lecturas),
                    "estado": consolidado.get("estado"),
                    "ruta_debug": ruta_debug,
                }
            )
            print(
                f"Consolidado {grupo}: final={consolidado.get('texto_final')} "
                f"conf={consolidado.get('confianza_final')} recortes={len(lecturas)}"
            )
        ruta_consolidados = SALIDA / "rectificacion_placas_reales_consolidado.json"
        ruta_consolidados.write_text(json.dumps(consolidados, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON consolidado: {ruta_consolidados}")

    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {len(filas)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


def _grupo_evento_desde_ruta(ruta: Path) -> str:
    texto = ruta.stem.lower()
    match = re.search(r"evento[_-]?(\d+)", texto)
    if match:
        return f"evento_{int(match.group(1)):04d}"
    return "grupo_general"


def _lectura_para_consolidar(fila: dict, resultado: dict) -> dict:
    lectura = {
        "ruta_recorte": fila.get("archivo"),
        "texto_crudo": fila.get("texto_crudo"),
        "texto_corregido_formato": fila.get("texto_corregido"),
        "confianza_cnn_caracteres": fila.get("confianza_cnn"),
        "puntaje_recorte": 0.0,
        "puntaje_segmentacion": fila.get("puntaje_segmentacion"),
        "cantidad_caracteres_segmentados": fila.get("cantidad_caracteres_segmentados"),
        "metodo_rectificacion": fila.get("metodo_rectificacion"),
        "estrategia_segmentacion": fila.get("estrategia_segmentacion"),
        "formato_valido": resultado.get("formato_valido"),
        "predicciones_caracteres": resultado.get("predicciones_caracteres", []),
    }
    lectura["calidad_lectura_evento"] = evaluar_calidad_lectura_evento(lectura)
    lectura["puntaje_lectura"] = lectura["calidad_lectura_evento"]["puntaje_lectura"]
    lectura["participa_votacion"] = lectura["calidad_lectura_evento"]["participa_votacion"]
    lectura["motivo_descarte_votacion"] = lectura["calidad_lectura_evento"]["motivo_descarte"]
    return lectura


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plate_reader import leer_placa_cnn_seguro_desde_monitoreo, leer_placa_desde_recorte


FUENTES = [
    Path("reports/evidencias/placas_capturadas"),
    Path("reports/evidencias/mejores_recortes_placa"),
    Path("reports/evidencias/video_anotado/mejores_recortes"),
    Path("reports/evidencias/eventos_placa"),
    Path("reports/evidencias/reconocimiento_caracteres/errores_monitoreo_cnn"),
]
SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_integracion_monitoreo_cnn")
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
            if ("recorte" in nombre or "placa" in nombre) and "frame_bbox" not in nombre and "mejor_frame_" not in nombre:
                rutas.append(ruta)
    return sorted(set(rutas))


def texto_historial(resultado: dict) -> str:
    texto = resultado.get("texto_corregido_formato") or resultado.get("texto_crudo") or ""
    estado = resultado.get("estado_lectura") or "sin_estado"
    if estado == "lectura_completa":
        return f"Lectura completa: {texto}"
    if estado == "lectura_parcial":
        return f"Lectura parcial: {texto}"
    if estado == "formato_dudoso":
        return f"Formato dudoso: {texto}"
    if estado == "segmentacion_incompleta":
        return f"Segmentacion incompleta: {texto or resultado.get('motivo')}"
    if estado == "error_cnn":
        return f"Error CNN: {resultado.get('etapa_error') or resultado.get('motivo')}"
    return f"Sin lectura: {resultado.get('motivo') or estado}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba integracion Monitoreo -> lector CNN.")
    parser.add_argument("--limit", type=int, default=10, help="Cantidad maxima de recortes. 0 procesa todos.")
    args = parser.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    recortes = buscar_recortes()
    if args.limit and args.limit > 0:
        recortes = recortes[: args.limit]

    filas = []
    detalles = []
    for idx, ruta in enumerate(recortes, start=1):
        directo = leer_placa_desde_recorte(str(ruta))
        monitoreo = leer_placa_cnn_seguro_desde_monitoreo(
            str(ruta),
            contexto={"funcion": "test_integracion_monitoreo_cnn", "indice": idx},
        )
        imagen_bgr = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        imagen_gris = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imagen_un_canal = np.expand_dims(imagen_gris, axis=2)
        variantes = {
            "bgr": leer_placa_cnn_seguro_desde_monitoreo(
                imagen_bgr, contexto={"funcion": "test_integracion_monitoreo_cnn", "variante": "bgr", "indice": idx}
            ),
            "grayscale": leer_placa_cnn_seguro_desde_monitoreo(
                imagen_gris, contexto={"funcion": "test_integracion_monitoreo_cnn", "variante": "grayscale", "indice": idx}
            ),
            "binaria": leer_placa_cnn_seguro_desde_monitoreo(
                imagen_binaria, contexto={"funcion": "test_integracion_monitoreo_cnn", "variante": "binaria", "indice": idx}
            ),
            "un_canal_3d": leer_placa_cnn_seguro_desde_monitoreo(
                imagen_un_canal, contexto={"funcion": "test_integracion_monitoreo_cnn", "variante": "un_canal_3d", "indice": idx}
            ),
        }
        variantes_sin_error = all(item.get("estado_lectura") != "error_cnn" for item in variantes.values())
        fila = {
            "archivo": str(ruta),
            "directo_ok": directo.get("ok"),
            "monitoreo_ok": monitoreo.get("ok"),
            "directo_estado": directo.get("estado_lectura"),
            "monitoreo_estado": monitoreo.get("estado_lectura"),
            "directo_texto": directo.get("texto_corregido_formato") or directo.get("texto_crudo"),
            "monitoreo_texto": monitoreo.get("texto_corregido_formato") or monitoreo.get("texto_crudo"),
            "monitoreo_historial": texto_historial(monitoreo),
            "confianza_cnn": monitoreo.get("confianza_cnn_caracteres"),
            "cantidad_caracteres": monitoreo.get("cantidad_caracteres_segmentados"),
            "estrategia_segmentacion": monitoreo.get("estrategia_segmentacion"),
            "error": monitoreo.get("error"),
            "etapa_error": monitoreo.get("etapa_error"),
            "variantes_canales_sin_error": variantes_sin_error,
            "estado_bgr": variantes["bgr"].get("estado_lectura"),
            "estado_grayscale": variantes["grayscale"].get("estado_lectura"),
            "estado_binaria": variantes["binaria"].get("estado_lectura"),
            "estado_un_canal_3d": variantes["un_canal_3d"].get("estado_lectura"),
            "coinciden": (
                directo.get("estado_lectura") == monitoreo.get("estado_lectura")
                and (directo.get("texto_corregido_formato") or directo.get("texto_crudo"))
                == (monitoreo.get("texto_corregido_formato") or monitoreo.get("texto_crudo"))
            ),
        }
        filas.append(fila)
        detalles.append({"fila": fila, "directo": directo, "monitoreo": monitoreo, "variantes_canales": variantes})
        print(
            f"{ruta.name}: directo={fila['directo_estado']} monitoreo={fila['monitoreo_estado']} "
            f"texto={fila['monitoreo_historial']} coincide={fila['coinciden']} "
            f"canales_sin_error={fila['variantes_canales_sin_error']}"
        )

    csv_path = SALIDA / "integracion_monitoreo_cnn.csv"
    json_path = SALIDA / "integracion_monitoreo_cnn.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()) if filas else [])
        if filas:
            writer.writeheader()
            writer.writerows(filas)
    json_path.write_text(json.dumps(detalles, ensure_ascii=False, indent=2), encoding="utf-8")
    errores = [fila for fila in filas if fila["monitoreo_estado"] == "error_cnn"]
    diferencias = [fila for fila in filas if not fila["coinciden"]]
    errores_canales = [fila for fila in filas if not fila["variantes_canales_sin_error"]]
    print("\nResumen")
    print("-------")
    print(f"Recortes evaluados: {len(filas)}")
    print(f"Errores CNN en integracion: {len(errores)}")
    print(f"Diferencias directo vs monitoreo: {len(diferencias)}")
    print(f"Recortes con error BGR/grayscale/binario/1-canal-3D: {len(errores_canales)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

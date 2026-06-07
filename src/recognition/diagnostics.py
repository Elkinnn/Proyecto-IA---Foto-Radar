"""Reportes del lector CNN de caracteres.

Contiene escritura de CSV/JSON para diagnostico de recortes y reportes
generales de reconocimiento, separada del pipeline de segmentacion.
"""

from datetime import datetime
from pathlib import Path
import csv
import json


RECONOCIMIENTO_CARACTERES_DIR = Path("reports") / "evidencias" / "reconocimiento_caracteres"
OCR_DIR = RECONOCIMIENTO_CARACTERES_DIR
DIAGNOSTICO_RECORTES_DIR = OCR_DIR / "diagnostico_recortes"


def registrar_diagnostico_recorte(resultado: dict, placa_esperada: str = "") -> dict:
    DIAGNOSTICO_RECORTES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_json = DIAGNOSTICO_RECORTES_DIR / "diagnostico_recortes.json"
    ruta_csv = DIAGNOSTICO_RECORTES_DIR / "diagnostico_recortes.csv"
    diagnostico = resultado.get("diagnostico", {})
    predicciones = resultado.get("predicciones_caracteres", [])
    filas_caracteres = []
    for item in diagnostico.get("diagnostico_caracteres", []):
        filas_caracteres.append(
            {
                "indice": item.get("indice"),
                "etiqueta_esperada": item.get("etiqueta_esperada"),
                "prediccion": item.get("prediccion"),
                "confianza": item.get("confianza"),
                "top3": item.get("top3"),
                "caracter_crudo": item.get("caracter_crudo"),
                "caracter_postprocesado": item.get("caracter_postprocesado"),
                "ruta_caracter": item.get("ruta_caracter"),
                "ruta_debug_normalizada": item.get("ruta_debug_normalizada"),
                "estado": item.get("estado"),
                "causa_probable": item.get("causa_probable"),
                "observacion": item.get("observacion"),
            }
        )

    registro = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "ruta_recorte": resultado.get("ruta_imagen"),
        "placa_esperada": placa_esperada,
        "placa_esperada_normalizada": diagnostico.get("placa_esperada_normalizada", ""),
        "texto_crudo": resultado.get("texto_detectado_crudo", ""),
        "texto_postprocesado": resultado.get("texto_postprocesado", ""),
        "formato_valido": resultado.get("formato", {}).get("valido"),
        "acierto": resultado.get("comparacion", {}).get("coincide"),
        "cantidad_esperada": diagnostico.get("cantidad_esperada", 0),
        "cantidad_detectada": diagnostico.get("cantidad_detectada", len(predicciones)),
        "predicciones_por_caracter": filas_caracteres,
        "confianza_por_caracter": [item.get("confianza") for item in filas_caracteres],
        "top3_por_caracter": [item.get("top3") for item in filas_caracteres],
        "causa_probable": diagnostico.get("causa_probable"),
        "mensaje": diagnostico.get("mensaje"),
    }

    registros = []
    if ruta_json.exists():
        try:
            registros = json.loads(ruta_json.read_text(encoding="utf-8"))
            if not isinstance(registros, list):
                registros = []
        except json.JSONDecodeError:
            registros = []
    registros.append(registro)
    ruta_json.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

    fila_csv = {
        "fecha_hora": registro["fecha_hora"],
        "ruta_recorte": registro["ruta_recorte"],
        "placa_esperada": registro["placa_esperada"],
        "texto_crudo": registro["texto_crudo"],
        "texto_postprocesado": registro["texto_postprocesado"],
        "formato_valido": registro["formato_valido"],
        "acierto": registro["acierto"],
        "cantidad_esperada": registro["cantidad_esperada"],
        "cantidad_detectada": registro["cantidad_detectada"],
        "prediccion_por_caracter": json.dumps([item.get("prediccion") for item in filas_caracteres], ensure_ascii=False),
        "confianza_por_caracter": json.dumps(registro["confianza_por_caracter"], ensure_ascii=False),
        "top3_por_caracter": json.dumps(registro["top3_por_caracter"], ensure_ascii=False),
        "causa_probable": registro["causa_probable"],
        "mensaje": registro["mensaje"],
    }
    existe_csv = ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(fila_csv.keys()))
        if not existe_csv:
            writer.writeheader()
        writer.writerow(fila_csv)

    return {
        "ruta_json": str(ruta_json),
        "ruta_csv": str(ruta_csv),
    }

def registrar_reporte_ocr(resultado: dict, fuente_recorte: str, placa_esperada: str = "") -> dict:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    ruta_json = OCR_DIR / "reporte_reconocimiento_caracteres.json"
    ruta_csv = OCR_DIR / "reporte_reconocimiento_caracteres.csv"
    fila = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "ruta_imagen": resultado.get("ruta_imagen"),
        "fuente_recorte": fuente_recorte,
        "metodo_segmentacion": resultado.get("metodo_segmentacion"),
        "placa_esperada": placa_esperada,
        "texto_detectado": resultado.get("texto_detectado"),
        "texto_detectado_crudo": resultado.get("texto_detectado_crudo"),
        "texto_postprocesado": resultado.get("texto_postprocesado"),
        "texto_normalizado": resultado.get("texto_normalizado"),
        "confianza_promedio": resultado.get("confianza_promedio", 0.0),
        "formato_valido": resultado.get("formato", {}).get("valido"),
        "valida_formato": resultado.get("formato", {}).get("valido"),
        "acierto": resultado.get("comparacion", {}).get("coincide"),
        "predicciones_caracteres": json.dumps(resultado.get("predicciones_caracteres", []), ensure_ascii=False),
        "cambios_postprocesamiento": json.dumps(resultado.get("cambios_postprocesamiento", []), ensure_ascii=False),
        "cantidad_caracteres_segmentados": resultado.get("cantidad_caracteres_segmentados", 0),
        "estado": resultado.get("estado"),
        "mensaje": resultado.get("mensaje"),
    }

    registros = []
    if ruta_json.exists():
        try:
            registros = json.loads(ruta_json.read_text(encoding="utf-8"))
            if not isinstance(registros, list):
                registros = []
        except json.JSONDecodeError:
            registros = []
    registros.append(fila)
    ruta_json.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

    existe_csv = ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(fila.keys()))
        if not existe_csv:
            writer.writeheader()
        writer.writerow(fila)

    return {
        "ruta_json": str(ruta_json),
        "ruta_csv": str(ruta_csv),
    }


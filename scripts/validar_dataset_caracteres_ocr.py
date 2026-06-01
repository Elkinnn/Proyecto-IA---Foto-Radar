from __future__ import annotations

import csv
from pathlib import Path
import sys

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador"
LABELS_CSV = DATASET_DIR / "labels.csv"
REPORTE_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "dataset_caracteres_ocr"
REPORTE_PATH = REPORTE_DIR / "reporte_validacion_dataset_caracteres_ocr.txt"
CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}
TAMANO_ESPERADO = (32, 32)


def listar_imagenes_clase(clase: str) -> list[Path]:
    clase_dir = DATASET_DIR / clase
    if not clase_dir.exists():
        return []
    return sorted(
        ruta
        for ruta in clase_dir.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
    )


def leer_labels() -> tuple[list[dict], list[str]]:
    errores = []
    if not LABELS_CSV.exists():
        return [], [f"No existe labels.csv: {LABELS_CSV}"]
    try:
        with open(LABELS_CSV, newline="", encoding="utf-8") as archivo:
            filas = list(csv.DictReader(archivo))
    except Exception as exc:
        return [], [f"No se pudo leer labels.csv: {exc}"]
    return filas, errores


def validar() -> dict:
    errores = []
    advertencias = []
    conteo_por_clase = {}
    imagenes_corruptas = []
    imagenes_tamano_invalido = []
    carpetas_faltantes = []

    if not DATASET_DIR.exists():
        errores.append(f"No existe el dataset: {DATASET_DIR}")

    for clase in CLASES:
        clase_dir = DATASET_DIR / clase
        if not clase_dir.exists():
            carpetas_faltantes.append(clase)
            conteo_por_clase[clase] = 0
            continue

        imagenes = listar_imagenes_clase(clase)
        conteo_por_clase[clase] = len(imagenes)
        for ruta in imagenes:
            imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
            if imagen is None:
                imagenes_corruptas.append(str(ruta.relative_to(ROOT_DIR)))
                continue
            alto, ancho = imagen.shape[:2]
            if (ancho, alto) != TAMANO_ESPERADO:
                imagenes_tamano_invalido.append(f"{ruta.relative_to(ROOT_DIR)} ({ancho}x{alto})")

    filas_labels, errores_labels = leer_labels()
    errores.extend(errores_labels)
    rutas_labels_invalidas = []
    clases_labels_invalidas = []

    for fila in filas_labels:
        ruta_rel = fila.get("ruta_imagen", "")
        etiqueta = fila.get("etiqueta", "")
        if etiqueta not in CLASES:
            clases_labels_invalidas.append(etiqueta)
        if not ruta_rel or not (ROOT_DIR / ruta_rel).exists():
            rutas_labels_invalidas.append(ruta_rel)

    clases_vacias = [clase for clase, cantidad in conteo_por_clase.items() if cantidad == 0]
    total_imagenes = sum(conteo_por_clase.values())

    if carpetas_faltantes:
        errores.append(f"Carpetas faltantes: {', '.join(carpetas_faltantes)}")
    if clases_vacias:
        errores.append(f"Clases vacias: {', '.join(clases_vacias)}")
    if imagenes_corruptas:
        errores.append(f"Imagenes corruptas: {len(imagenes_corruptas)}")
    if imagenes_tamano_invalido:
        errores.append(f"Imagenes con tamano distinto a 32x32: {len(imagenes_tamano_invalido)}")
    if rutas_labels_invalidas:
        errores.append(f"Rutas invalidas en labels.csv: {len(rutas_labels_invalidas)}")
    if clases_labels_invalidas:
        errores.append(f"Etiquetas invalidas en labels.csv: {len(set(clases_labels_invalidas))}")

    dataset_valido = not errores and total_imagenes > 0 and not clases_vacias
    return {
        "ruta": str(DATASET_DIR.relative_to(ROOT_DIR)),
        "total_imagenes": total_imagenes,
        "total_clases": len(CLASES),
        "conteo_por_clase": conteo_por_clase,
        "clases_vacias": clases_vacias,
        "carpetas_faltantes": carpetas_faltantes,
        "imagenes_corruptas": imagenes_corruptas,
        "imagenes_tamano_invalido": imagenes_tamano_invalido,
        "labels_total": len(filas_labels),
        "rutas_labels_invalidas": rutas_labels_invalidas,
        "errores": errores,
        "advertencias": advertencias,
        "dataset_valido": dataset_valido,
    }


def generar_reporte(resultado: dict) -> str:
    lineas = [
        "Validacion dataset OCR caracteres",
        "--------------------------------",
        f"Ruta: {resultado['ruta']}",
        f"Total imagenes: {resultado['total_imagenes']}",
        f"Total clases: {resultado['total_clases']}",
        f"Labels CSV filas: {resultado['labels_total']}",
        f"Clases vacias: {', '.join(resultado['clases_vacias']) if resultado['clases_vacias'] else 'Ninguna'}",
        f"Errores: {len(resultado['errores'])}",
        f"Dataset valido: {'si' if resultado['dataset_valido'] else 'no'}",
        "",
        "Cantidad por clase:",
    ]
    for clase, cantidad in resultado["conteo_por_clase"].items():
        lineas.append(f"- {clase}: {cantidad}")

    if resultado["advertencias"]:
        lineas.extend(["", "Advertencias:"])
        lineas.extend(f"- {item}" for item in resultado["advertencias"])

    if resultado["errores"]:
        lineas.extend(["", "Detalle de errores:"])
        lineas.extend(f"- {item}" for item in resultado["errores"])

    if resultado["imagenes_corruptas"][:20]:
        lineas.extend(["", "Imagenes corruptas (primeras 20):"])
        lineas.extend(f"- {item}" for item in resultado["imagenes_corruptas"][:20])

    if resultado["imagenes_tamano_invalido"][:20]:
        lineas.extend(["", "Imagenes con tamano invalido (primeras 20):"])
        lineas.extend(f"- {item}" for item in resultado["imagenes_tamano_invalido"][:20])

    REPORTE_DIR.mkdir(parents=True, exist_ok=True)
    texto = "\n".join(lineas)
    REPORTE_PATH.write_text(texto, encoding="utf-8")
    return texto


def main() -> None:
    resultado = validar()
    texto = generar_reporte(resultado)
    print(texto)
    print("")
    print(f"Reporte guardado en: {REPORTE_PATH.relative_to(ROOT_DIR)}")
    if not resultado["dataset_valido"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

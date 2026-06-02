from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_detector import PlateDetector  # noqa: E402
from src.plate_reader import preprocesar_placa, segmentar_caracteres_v2  # noqa: E402


PLACAS_DIR = ROOT_DIR / "datasets" / "placas_ecuador"
CSV_PLACAS = PLACAS_DIR / "placas_labels.csv"
CARACTERES_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador"
LABELS_CSV = CARACTERES_DIR / "labels.csv"
AUDITORIA_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "generacion_caracteres_ecuador"
RECORTES_TEMP_DIR = AUDITORIA_DIR / "recortes_placa"
DEBUG_DIR = AUDITORIA_DIR / "debug_segmentacion"

CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}
TAMANO_FINAL = 32
CAMPOS_LABELS = [
    "ruta_imagen",
    "etiqueta",
    "origen",
    "ruta_original",
    "ancho_original",
    "alto_original",
    "ancho_final",
    "alto_final",
    "fecha_procesamiento",
]
CAMPOS_REPORTE = [
    "ruta_imagen",
    "placa_esperada",
    "placa_normalizada",
    "estado",
    "motivo",
    "cantidad_caracteres_segmentados",
    "cantidad_caracteres_esperados",
    "caracteres_guardados",
    "rutas_guardadas",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera caracteres OCR desde placas ecuatorianas segmentadas."
    )
    parser.add_argument("--limit", type=int, default=None, help="Procesa solo N imagenes.")
    parser.add_argument("--dry-run", action="store_true", help="Prueba sin guardar caracteres ni labels.")
    parser.add_argument(
        "--usar-yolo",
        action="store_true",
        help="Intenta detectar placa con YOLO cuando la imagen no parece recorte de placa.",
    )
    parser.add_argument(
        "--reset-generados",
        action="store_true",
        help="Borra solo caracteres generados previamente con origen caracter_ecuador_segmentado.",
    )
    return parser.parse_args()


def normalizar_placa_esperada(texto: str | None) -> str:
    placa = re.sub(r"[^A-Za-z0-9]", "", texto or "").upper()
    if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", placa):
        return placa
    return ""


def validar_placa_para_guardado(placa: str) -> tuple[bool, str]:
    if not placa:
        return False, "No existe placa esperada normalizada."
    if not re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", placa):
        return False, "La placa esperada no tiene formato ecuatoriano valido."
    invalidos = [caracter for caracter in placa if caracter not in CLASES]
    if invalidos:
        return False, f"La placa contiene caracteres no permitidos: {', '.join(invalidos)}"
    return True, ""


def cargar_placas_csv() -> dict[str, str]:
    if not CSV_PLACAS.exists():
        return {}

    etiquetas = {}
    with open(CSV_PLACAS, newline="", encoding="utf-8") as archivo:
        reader = csv.DictReader(archivo)
        for fila in reader:
            ruta = (fila.get("ruta_imagen") or "").strip()
            placa = normalizar_placa_esperada(fila.get("placa"))
            if ruta and placa:
                etiquetas[_normalizar_clave_ruta(ruta)] = placa
    return etiquetas


def _normalizar_clave_ruta(ruta: str | Path) -> str:
    return str(Path(str(ruta)).as_posix()).lower()


def obtener_placa_esperada(ruta_imagen: Path, etiquetas_csv: dict[str, str]) -> tuple[str, str]:
    rel = ruta_imagen.relative_to(PLACAS_DIR)
    claves = [
        _normalizar_clave_ruta(rel),
        _normalizar_clave_ruta(ruta_imagen.name),
        _normalizar_clave_ruta(ruta_imagen),
    ]
    for clave in claves:
        placa = etiquetas_csv.get(clave)
        if placa:
            return placa, "csv"

    match = re.search(r"([A-Za-z]{3})[-_\s]?([0-9]{3,4})", ruta_imagen.stem)
    if match:
        placa = normalizar_placa_esperada(f"{match.group(1)}{match.group(2)}")
        if placa:
            return placa, "nombre_archivo"

    return "", ""


def listar_imagenes_placas() -> list[Path]:
    if not PLACAS_DIR.exists():
        return []
    return sorted(
        ruta
        for ruta in PLACAS_DIR.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
    )


def parece_recorte_placa(imagen: np.ndarray) -> bool:
    alto, ancho = imagen.shape[:2]
    if alto <= 0 or ancho <= 0:
        return False
    aspect = ancho / alto
    return 1.8 <= aspect <= 6.8 and ancho >= 80 and alto >= 18


def preparar_recorte_placa(
    ruta_imagen: Path,
    imagen: np.ndarray,
    usar_yolo: bool,
    detector: PlateDetector | None,
) -> tuple[Path | None, str, str]:
    if parece_recorte_placa(imagen):
        return ruta_imagen, "recorte_directo", ""

    if not usar_yolo:
        return None, "placa_no_detectada", "La imagen no parece recorte de placa y no se uso --usar-yolo."

    if detector is None or detector.model is None:
        return None, "placa_no_detectada", "YOLO no esta disponible o no se pudo cargar el modelo de placas."

    resultado = detector.detectar_en_frame(imagen, conf_min=0.35)
    detecciones = resultado.get("detecciones", [])
    if not detecciones:
        return None, "placa_no_detectada", resultado.get("mensaje", "No se detecto placa con YOLO.")

    mejor = max(detecciones, key=lambda item: float(item.get("confianza", 0.0)))
    recorte = mejor.get("recorte_placa")
    if recorte is None:
        return None, "placa_no_detectada", "YOLO detecto placa, pero no entrego recorte util."

    RECORTES_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    nombre = f"{ruta_imagen.stem}_recorte_yolo.jpg"
    ruta_recorte = RECORTES_TEMP_DIR / nombre
    cv2.imwrite(str(ruta_recorte), recorte)
    return ruta_recorte, "recorte_yolo", ""


def normalizar_caracter(ruta_caracter: Path) -> tuple[np.ndarray | None, tuple[int, int]]:
    imagen = cv2.imread(str(ruta_caracter), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return None, (0, 0)

    alto_original, ancho_original = imagen.shape[:2]
    imagen = cv2.GaussianBlur(imagen, (3, 3), 0)
    _, binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blancos = cv2.countNonZero(binaria)
    total = binaria.shape[0] * binaria.shape[1]
    if blancos > total * 0.5:
        binaria = cv2.bitwise_not(binaria)

    coords = cv2.findNonZero(binaria)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        binaria = binaria[y : y + h, x : x + w]

    alto, ancho = binaria.shape[:2]
    escala = min((TAMANO_FINAL - 6) / max(ancho, 1), (TAMANO_FINAL - 6) / max(alto, 1))
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    redimensionada = cv2.resize(binaria, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((TAMANO_FINAL, TAMANO_FINAL), dtype=np.uint8)
    x0 = (TAMANO_FINAL - nuevo_ancho) // 2
    y0 = (TAMANO_FINAL - nuevo_alto) // 2
    lienzo[y0 : y0 + nuevo_alto, x0 : x0 + nuevo_ancho] = redimensionada
    return lienzo, (ancho_original, alto_original)


def leer_labels() -> list[dict]:
    if not LABELS_CSV.exists():
        return []
    with open(LABELS_CSV, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def escribir_labels(filas: list[dict]) -> None:
    CARACTERES_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=CAMPOS_LABELS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)


def reset_generados(filas: list[dict]) -> list[dict]:
    for clase in CLASES:
        clase_dir = CARACTERES_DIR / clase
        if not clase_dir.exists():
            continue
        for ruta in clase_dir.glob(f"{clase}_ecuador_*.png"):
            if ruta.is_file():
                ruta.unlink()

    return [fila for fila in filas if fila.get("origen") != "caracter_ecuador_segmentado"]


def hash_fuente(ruta_imagen: Path, placa: str, indice: int) -> str:
    texto = f"{ruta_imagen.resolve()}|{placa}|{indice}"
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:12]


def guardar_caracteres(
    ruta_original: Path,
    placa: str,
    caracteres: list[dict],
    filas_labels: list[dict],
    existentes: set[str],
    dry_run: bool,
    indices_guardar: list[int] | None = None,
) -> tuple[int, list[str]]:
    placa = normalizar_placa_esperada(placa)
    valido, motivo = validar_placa_para_guardado(placa)
    if not valido:
        raise ValueError(motivo)

    indices = indices_guardar if indices_guardar is not None else list(range(len(caracteres)))
    caracteres_seleccionados = [caracteres[idx] for idx in indices if 0 <= idx < len(caracteres)]
    if len(caracteres_seleccionados) != len(placa):
        raise ValueError("No se guardan caracteres porque la cantidad segmentada no coincide con la placa esperada.")

    fecha = datetime.now().isoformat(timespec="seconds")
    guardados = []

    for indice, (etiqueta, caracter_segmentado) in enumerate(zip(placa, caracteres_seleccionados), start=1):
        if etiqueta not in CLASES:
            continue

        identificador = hash_fuente(ruta_original, placa, indice)
        nombre = f"{etiqueta}_ecuador_{identificador}_{indice:02d}.png"
        ruta_destino = CARACTERES_DIR / etiqueta / nombre
        ruta_rel = str(ruta_destino.relative_to(ROOT_DIR))
        if ruta_rel in existentes or ruta_destino.exists():
            continue

        ruta_segmentada = Path(caracter_segmentado.get("ruta_caracter", ""))
        imagen, (ancho_original, alto_original) = normalizar_caracter(ruta_segmentada)
        if imagen is None:
            continue

        if not dry_run:
            ruta_destino.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ruta_destino), imagen)
            filas_labels.append(
                {
                    "ruta_imagen": ruta_rel,
                    "etiqueta": etiqueta,
                    "origen": "caracter_ecuador_segmentado",
                    "ruta_original": str(ruta_original.relative_to(ROOT_DIR)),
                    "ancho_original": ancho_original,
                    "alto_original": alto_original,
                    "ancho_final": TAMANO_FINAL,
                    "alto_final": TAMANO_FINAL,
                    "fecha_procesamiento": fecha,
                }
            )
            existentes.add(ruta_rel)

        guardados.append(ruta_rel)

    return len(guardados), guardados


def registrar_auditoria_guardado(
    ruta_imagen: Path,
    placa_esperada: str,
    placa_normalizada: str,
    estado: str,
    motivo: str,
    cantidad_segmentada: int,
    cantidad_esperada: int,
    rutas_guardadas: list[str] | None = None,
) -> dict:
    AUDITORIA_DIR.mkdir(parents=True, exist_ok=True)
    ruta_csv = AUDITORIA_DIR / "reporte_generacion_caracteres.csv"
    fila = {
        "ruta_imagen": _ruta_relativa_o_absoluta(ruta_imagen),
        "placa_esperada": placa_esperada,
        "placa_normalizada": placa_normalizada,
        "estado": estado,
        "motivo": motivo,
        "cantidad_caracteres_segmentados": cantidad_segmentada,
        "cantidad_caracteres_esperados": cantidad_esperada,
        "caracteres_guardados": len(rutas_guardadas or []),
        "rutas_guardadas": ";".join(rutas_guardadas or []),
    }
    existe = ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=CAMPOS_REPORTE)
        if not existe:
            writer.writeheader()
        writer.writerow(fila)
    return {"ruta_csv": str(ruta_csv), "fila": fila}


def _ruta_relativa_o_absoluta(ruta: Path) -> str:
    try:
        return str(ruta.relative_to(ROOT_DIR))
    except ValueError:
        return str(ruta)


def procesar(args: argparse.Namespace) -> dict:
    if not PLACAS_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset fuente: {PLACAS_DIR}")

    for clase in CLASES:
        (CARACTERES_DIR / clase).mkdir(parents=True, exist_ok=True)
    AUDITORIA_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    filas_labels = leer_labels()
    if args.reset_generados and not args.dry_run:
        filas_labels = reset_generados(filas_labels)

    existentes = {
        fila.get("ruta_imagen", "")
        for fila in filas_labels
        if fila.get("origen") == "caracter_ecuador_segmentado"
    }

    etiquetas_csv = cargar_placas_csv()
    imagenes = listar_imagenes_placas()
    if args.limit is not None:
        imagenes = imagenes[: max(args.limit, 0)]

    detector = PlateDetector() if args.usar_yolo else None
    reporte = []
    resumen = {
        "procesadas": 0,
        "guardado": 0,
        "sin_etiqueta": 0,
        "placa_no_detectada": 0,
        "cantidad_no_coincide": 0,
        "error_preprocesamiento": 0,
        "error_segmentacion": 0,
    }

    for ruta_imagen in imagenes:
        resumen["procesadas"] += 1
        rel = str(ruta_imagen.relative_to(ROOT_DIR))
        placa, origen_placa = obtener_placa_esperada(ruta_imagen, etiquetas_csv)
        if not placa:
            _agregar_reporte(reporte, rel, "", "", "sin_etiqueta", "No se pudo obtener placa desde CSV ni nombre de archivo.")
            resumen["sin_etiqueta"] += 1
            continue

        imagen = cv2.imread(str(ruta_imagen))
        if imagen is None:
            _agregar_reporte(reporte, rel, placa, placa, "error_preprocesamiento", "No se pudo cargar la imagen.")
            resumen["error_preprocesamiento"] += 1
            continue

        ruta_recorte, origen_recorte, motivo_recorte = preparar_recorte_placa(
            ruta_imagen, imagen, args.usar_yolo, detector
        )
        if ruta_recorte is None:
            _agregar_reporte(reporte, rel, placa, placa, "placa_no_detectada", motivo_recorte)
            resumen["placa_no_detectada"] += 1
            continue

        nombre_base = f"{ruta_imagen.stem}_{placa}_{origen_placa}_{origen_recorte}"
        preprocesamiento = preprocesar_placa(str(ruta_recorte), nombre_base)
        ruta_procesada = preprocesamiento.get("ruta_imagen_procesada")
        if not ruta_procesada:
            _agregar_reporte(reporte, rel, placa, placa, "error_preprocesamiento", preprocesamiento.get("mensaje", "Error."))
            resumen["error_preprocesamiento"] += 1
            continue

        segmentacion = segmentar_caracteres_v2(ruta_procesada, nombre_base)
        if segmentacion.get("estado") != "ok":
            _agregar_reporte(reporte, rel, placa, placa, "error_segmentacion", segmentacion.get("mensaje", "Error."))
            resumen["error_segmentacion"] += 1
            continue

        copiar_debug(segmentacion, ruta_imagen.stem, placa)
        caracteres = segmentacion.get("caracteres", [])
        cantidad_segmentada = len(caracteres)
        cantidad_esperada = len(placa)
        if cantidad_segmentada != cantidad_esperada:
            _agregar_reporte(
                reporte,
                rel,
                placa,
                placa,
                "cantidad_no_coincide",
                "La cantidad segmentada no coincide con la placa esperada.",
                cantidad_segmentada,
                cantidad_esperada,
            )
            resumen["cantidad_no_coincide"] += 1
            continue

        cantidad, rutas = guardar_caracteres(ruta_imagen, placa, caracteres, filas_labels, existentes, args.dry_run)
        _agregar_reporte(
            reporte,
            rel,
            placa,
            placa,
            "guardado",
            "Caracteres guardados." if not args.dry_run else "Dry-run: caracteres validos, no guardados.",
            cantidad_segmentada,
            cantidad_esperada,
            cantidad,
            rutas,
        )
        resumen["guardado"] += 1

    if not args.dry_run:
        escribir_labels(filas_labels)
    escribir_reporte(reporte, resumen, args)
    return resumen


def copiar_debug(segmentacion: dict, stem: str, placa: str) -> None:
    for clave in ("ruta_banda", "ruta_debug"):
        ruta = segmentacion.get(clave)
        if not ruta:
            continue
        origen = Path(ruta)
        if origen.exists():
            destino = DEBUG_DIR / f"{stem}_{placa}_{origen.name}"
            shutil.copy2(origen, destino)


def _agregar_reporte(
    reporte: list[dict],
    ruta_imagen: str,
    placa_esperada: str,
    placa_normalizada: str,
    estado: str,
    motivo: str,
    cantidad_segmentada: int = 0,
    cantidad_esperada: int = 0,
    caracteres_guardados: int = 0,
    rutas_guardadas: list[str] | None = None,
) -> None:
    reporte.append(
        {
            "ruta_imagen": ruta_imagen,
            "placa_esperada": placa_esperada,
            "placa_normalizada": placa_normalizada,
            "estado": estado,
            "motivo": motivo,
            "cantidad_caracteres_segmentados": cantidad_segmentada,
            "cantidad_caracteres_esperados": cantidad_esperada,
            "caracteres_guardados": caracteres_guardados,
            "rutas_guardadas": ";".join(rutas_guardadas or []),
        }
    )


def escribir_reporte(reporte: list[dict], resumen: dict, args: argparse.Namespace) -> None:
    AUDITORIA_DIR.mkdir(parents=True, exist_ok=True)
    ruta_csv = AUDITORIA_DIR / "reporte_generacion_caracteres.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=CAMPOS_REPORTE)
        writer.writeheader()
        writer.writerows(reporte)

    lineas = [
        "Generacion de caracteres desde placas ecuatorianas",
        "--------------------------------------------------",
        f"Fuente: {PLACAS_DIR.relative_to(ROOT_DIR)}",
        f"Destino: {CARACTERES_DIR.relative_to(ROOT_DIR)}",
        f"Dry-run: {'si' if args.dry_run else 'no'}",
        f"Usar YOLO: {'si' if args.usar_yolo else 'no'}",
        f"Limit: {args.limit if args.limit is not None else 'sin limite'}",
        "",
        f"Imagenes procesadas: {resumen['procesadas']}",
        f"Placas con caracteres guardados: {resumen['guardado']}",
        f"Sin etiqueta: {resumen['sin_etiqueta']}",
        f"Placa no detectada: {resumen['placa_no_detectada']}",
        f"Cantidad no coincide: {resumen['cantidad_no_coincide']}",
        f"Error preprocesamiento: {resumen['error_preprocesamiento']}",
        f"Error segmentacion: {resumen['error_segmentacion']}",
        "",
        f"Reporte CSV: {ruta_csv.relative_to(ROOT_DIR)}",
    ]
    (AUDITORIA_DIR / "resumen_generacion_caracteres.txt").write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        resumen = procesar(args)
    except Exception as exc:
        print(f"Error generando caracteres desde placas ecuatorianas: {exc}")
        sys.exit(1)

    print("Generacion de caracteres desde placas ecuatorianas")
    print("--------------------------------------------------")
    print(f"Procesadas: {resumen['procesadas']}")
    print(f"Guardado: {resumen['guardado']}")
    print(f"Sin etiqueta: {resumen['sin_etiqueta']}")
    print(f"Placa no detectada: {resumen['placa_no_detectada']}")
    print(f"Cantidad no coincide: {resumen['cantidad_no_coincide']}")
    print(f"Reporte: {(AUDITORIA_DIR / 'reporte_generacion_caracteres.csv').relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

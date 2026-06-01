from __future__ import annotations

from pathlib import Path
import sys

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT_DIR / "datasets" / "caracteres_ecuador_split"
REPORTE_DIR = ROOT_DIR / "reports" / "evidencias" / "ocr" / "dataset_caracteres_ocr"
REPORTE_PATH = REPORTE_DIR / "reporte_validacion_split_caracteres_ocr.txt"
CLASES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPLITS = ["train", "valid", "test"]
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}
TAMANO_ESPERADO = (32, 32)


def listar_imagenes(split: str, clase: str) -> list[Path]:
    carpeta = SPLIT_DIR / split / clase
    if not carpeta.exists():
        return []
    return sorted(
        ruta
        for ruta in carpeta.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
    )


def validar() -> dict:
    errores = []
    conteos = {split: {} for split in SPLITS}
    corruptas = []
    tamano_invalido = []
    carpetas_faltantes = []
    clases_vacias = []

    if not SPLIT_DIR.exists():
        errores.append(f"No existe split: {SPLIT_DIR}")

    for split in SPLITS:
        split_dir = SPLIT_DIR / split
        if not split_dir.exists():
            errores.append(f"No existe carpeta {split}: {split_dir}")
            continue
        for clase in CLASES:
            clase_dir = split_dir / clase
            if not clase_dir.exists():
                carpetas_faltantes.append(f"{split}/{clase}")
                conteos[split][clase] = 0
                continue

            imagenes = listar_imagenes(split, clase)
            conteos[split][clase] = len(imagenes)
            if not imagenes:
                clases_vacias.append(f"{split}/{clase}")

            for ruta in imagenes:
                imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
                if imagen is None:
                    corruptas.append(str(ruta.relative_to(ROOT_DIR)))
                    continue
                alto, ancho = imagen.shape[:2]
                if (ancho, alto) != TAMANO_ESPERADO:
                    tamano_invalido.append(f"{ruta.relative_to(ROOT_DIR)} ({ancho}x{alto})")

    if carpetas_faltantes:
        errores.append(f"Carpetas faltantes: {len(carpetas_faltantes)}")
    if clases_vacias:
        errores.append(f"Clases vacias en split: {len(clases_vacias)}")
    if corruptas:
        errores.append(f"Imagenes corruptas: {len(corruptas)}")
    if tamano_invalido:
        errores.append(f"Imagenes con tamano distinto a 32x32: {len(tamano_invalido)}")

    total = sum(sum(clases.values()) for clases in conteos.values())
    return {
        "total": total,
        "conteos": conteos,
        "carpetas_faltantes": carpetas_faltantes,
        "clases_vacias": clases_vacias,
        "corruptas": corruptas,
        "tamano_invalido": tamano_invalido,
        "errores": errores,
        "valido": not errores and total > 0,
    }


def generar_reporte(resultado: dict) -> str:
    lineas = [
        "Validacion split OCR caracteres",
        "--------------------------------",
        f"Ruta: {SPLIT_DIR.relative_to(ROOT_DIR)}",
        f"Total imagenes: {resultado['total']}",
        f"Total clases: {len(CLASES)}",
        f"Errores: {len(resultado['errores'])}",
        f"Split valido: {'si' if resultado['valido'] else 'no'}",
        "",
        "Cantidad por split/clase:",
    ]
    for clase in CLASES:
        train = resultado["conteos"].get("train", {}).get(clase, 0)
        valid = resultado["conteos"].get("valid", {}).get(clase, 0)
        test = resultado["conteos"].get("test", {}).get(clase, 0)
        lineas.append(f"- {clase}: train={train} valid={valid} test={test}")

    if resultado["errores"]:
        lineas.extend(["", "Detalle de errores:"])
        lineas.extend(f"- {item}" for item in resultado["errores"])

    if resultado["clases_vacias"][:30]:
        lineas.extend(["", "Clases vacias (primeras 30):"])
        lineas.extend(f"- {item}" for item in resultado["clases_vacias"][:30])

    if resultado["corruptas"][:20]:
        lineas.extend(["", "Imagenes corruptas (primeras 20):"])
        lineas.extend(f"- {item}" for item in resultado["corruptas"][:20])

    if resultado["tamano_invalido"][:20]:
        lineas.extend(["", "Tamano invalido (primeras 20):"])
        lineas.extend(f"- {item}" for item in resultado["tamano_invalido"][:20])

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
    if not resultado["valido"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

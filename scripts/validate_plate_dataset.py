from pathlib import Path

import yaml


DATASET_DIR = Path("datasets/placas_ecuador")
REPORT_PATH = Path("reports/evidencias/dataset_placas_ecuador/reporte_validacion_dataset_placas.txt")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_clases(data_yaml: Path) -> set[int]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return set(range(len(names)))
    if isinstance(names, dict):
        return {int(class_id) for class_id in names.keys()}
    return set()


def validar_label(label_path: Path, clases_validas: set[int]) -> tuple[list[str], int]:
    errores = []
    anotaciones = 0

    for numero_linea, linea in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        linea = linea.strip()
        if not linea:
            continue

        partes = linea.split()
        if len(partes) != 5:
            errores.append(f"{label_path}:{numero_linea} debe tener 5 valores y tiene {len(partes)}")
            continue

        try:
            class_id = int(float(partes[0]))
            coordenadas = [float(valor) for valor in partes[1:]]
        except ValueError:
            errores.append(f"{label_path}:{numero_linea} contiene valores no numericos")
            continue

        if class_id not in clases_validas:
            errores.append(f"{label_path}:{numero_linea} class_id invalido: {class_id}")

        for valor in coordenadas:
            if valor < 0 or valor > 1:
                errores.append(f"{label_path}:{numero_linea} coordenada fuera de [0, 1]: {valor}")

        anotaciones += 1

    return errores, anotaciones


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lineas = ["Validacion de dataset de placas ecuatorianas", "=" * 52, ""]
    errores = []
    total_anotaciones = 0

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        errores.append(f"No existe data.yaml en {DATASET_DIR}")
        clases_validas = set()
    else:
        lineas.append(f"data.yaml: OK ({data_yaml})")
        clases_validas = cargar_clases(data_yaml)
        lineas.append(f"Clases validas: {sorted(clases_validas)}")

    for split in SPLITS:
        images_dir = DATASET_DIR / "images" / split
        labels_dir = DATASET_DIR / "labels" / split
        lineas.extend(["", f"Split: {split}", "-" * 20])

        if not images_dir.exists():
            errores.append(f"No existe {images_dir}")
            imagenes = []
        else:
            imagenes = [ruta for ruta in images_dir.iterdir() if ruta.suffix.lower() in IMAGE_EXTENSIONS]

        if not labels_dir.exists():
            errores.append(f"No existe {labels_dir}")
            labels = []
        else:
            labels = list(labels_dir.glob("*.txt"))

        imagen_stems = {ruta.stem for ruta in imagenes}
        label_stems = {ruta.stem for ruta in labels}
        imagenes_sin_label = sorted(imagen_stems - label_stems)
        labels_sin_imagen = sorted(label_stems - imagen_stems)

        lineas.append(f"Imagenes: {len(imagenes)}")
        lineas.append(f"Labels: {len(labels)}")
        lineas.append(f"Imagenes sin label: {len(imagenes_sin_label)}")
        lineas.append(f"Labels sin imagen: {len(labels_sin_imagen)}")

        for stem in imagenes_sin_label[:50]:
            errores.append(f"{split}: imagen sin label: {stem}")
        for stem in labels_sin_imagen[:50]:
            errores.append(f"{split}: label sin imagen: {stem}")

        for label_path in labels:
            errores_label, anotaciones = validar_label(label_path, clases_validas)
            errores.extend(errores_label)
            total_anotaciones += anotaciones

    lineas.extend(["", f"Total de anotaciones: {total_anotaciones}", ""])
    if errores:
        lineas.append("Errores encontrados:")
        lineas.extend(f"- {error}" for error in errores)
    else:
        lineas.append("Resultado: dataset valido para entrenamiento YOLO.")

    REPORT_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Reporte guardado en: {REPORT_PATH}")
    print(f"Errores encontrados: {len(errores)}")


if __name__ == "__main__":
    main()

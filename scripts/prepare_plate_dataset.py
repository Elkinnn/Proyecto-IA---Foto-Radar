import shutil
from pathlib import Path

import yaml


SOURCE_DIR = Path(r"C:\Users\elkin\OneDrive\Desktop\Placas_Ecuador\Placas_Ecuador")
DEST_DIR = Path("datasets/placas_ecuador")


def copiar_contenido(origen: Path, destino: Path) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    if not origen.exists():
        raise FileNotFoundError(f"No existe la carpeta fuente: {origen}")

    cantidad = 0
    for archivo in origen.iterdir():
        if archivo.is_file():
            shutil.copy2(archivo, destino / archivo.name)
            cantidad += 1
    return cantidad


def escribir_data_yaml() -> None:
    data = {
        "path": "datasets/placas_ecuador",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "placa"},
    }
    with open(DEST_DIR / "data.yaml", "w", encoding="utf-8") as archivo:
        yaml.safe_dump(data, archivo, sort_keys=False, allow_unicode=True)


def escribir_readme(conteos: dict[str, dict[str, int]]) -> None:
    lineas = [
        "# Dataset Placas Ecuador",
        "",
        "Dataset preparado para deteccion de placas ecuatorianas completas con YOLO.",
        "",
        "Estructura normalizada:",
        "",
        "- `images/train`, `labels/train`",
        "- `images/val`, `labels/val`",
        "- `images/test`, `labels/test`",
        "",
        "Conteos copiados:",
    ]
    for split, valores in conteos.items():
        lineas.append(f"- {split}: {valores['images']} imagenes, {valores['labels']} labels")

    (DEST_DIR / "README.md").write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    mapeo = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }
    conteos = {}

    for split_origen, split_destino in mapeo.items():
        imagenes = copiar_contenido(SOURCE_DIR / split_origen / "images", DEST_DIR / "images" / split_destino)
        labels = copiar_contenido(SOURCE_DIR / split_origen / "labels", DEST_DIR / "labels" / split_destino)
        conteos[split_destino] = {"images": imagenes, "labels": labels}

    escribir_data_yaml()
    escribir_readme(conteos)

    print(f"Dataset preparado en: {DEST_DIR}")
    for split, valores in conteos.items():
        print(f"{split}: {valores['images']} imagenes, {valores['labels']} labels")


if __name__ == "__main__":
    main()

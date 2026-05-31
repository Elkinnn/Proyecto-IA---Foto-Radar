from pathlib import Path
from uuid import uuid4

import yaml


def cargar_config(ruta_config: str = "config.yaml") -> dict:
    with open(ruta_config, "r", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def asegurar_directorio(ruta: str | Path) -> Path:
    ruta_path = Path(ruta)
    ruta_path.mkdir(parents=True, exist_ok=True)
    return ruta_path


def guardar_archivo_subido(archivo_subido, directorio_destino: str | Path) -> str:
    destino = asegurar_directorio(directorio_destino)
    extension = Path(archivo_subido.name).suffix.lower()
    ruta_archivo = destino / f"{uuid4().hex}{extension}"
    ruta_archivo.write_bytes(archivo_subido.getbuffer())
    return str(ruta_archivo)


def normalizar_placa(placa: str | None) -> str:
    if not placa:
        return ""
    return placa.strip().upper().replace("-", "").replace(" ", "")

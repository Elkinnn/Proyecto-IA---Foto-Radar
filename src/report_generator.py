import json
from datetime import datetime
from pathlib import Path


def guardar_reporte(resultado: dict, reports_dir: str) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    placa = resultado.get("texto_placa") or "sin_placa"
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_reporte = Path(reports_dir) / f"evidencia_{placa}_{marca_tiempo}.json"

    with open(ruta_reporte, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2)

    return str(ruta_reporte)

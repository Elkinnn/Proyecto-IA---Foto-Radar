from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_reader import leer_placa_desde_recorte, registrar_diagnostico_recorte  # noqa: E402


BUSCAR_EN = [
    ROOT_DIR / "reports" / "evidencias" / "eventos_placa",
    ROOT_DIR / "reports" / "evidencias" / "placas_detectadas",
    ROOT_DIR / "data" / "input",
]
PLACAS_PRUEBA = ["PDZ279", "PDP6236", "TDH493"]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostica OCR experimental sobre recortes de placa.")
    parser.add_argument(
        "--caso",
        action="append",
        default=[],
        help="Caso manual con formato PLACA=ruta_recorte. Puede repetirse.",
    )
    return parser.parse_args()


def listar_recortes() -> list[Path]:
    rutas = []
    for carpeta in BUSCAR_EN:
        if carpeta.exists():
            rutas.extend(
                ruta
                for ruta in carpeta.rglob("*")
                if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES
            )
    return sorted(rutas)


def buscar_por_placa(placa: str, rutas: list[Path]) -> Path | None:
    placa_lower = placa.lower()
    for ruta in rutas:
        if placa_lower in ruta.stem.lower():
            return ruta
    return None


def main() -> None:
    args = parse_args()
    rutas = listar_recortes()
    print("Diagnostico OCR sobre recortes")
    print("------------------------------")
    casos_manuales = []
    for caso in args.caso:
        if "=" not in caso:
            print(f"Caso ignorado por formato invalido: {caso}")
            continue
        placa, ruta_txt = caso.split("=", 1)
        ruta = Path(ruta_txt)
        if not ruta.is_absolute():
            ruta = ROOT_DIR / ruta
        casos_manuales.append((placa.strip().upper(), ruta))

    if not rutas and not casos_manuales:
        print("No se encontraron recortes en eventos_placa, placas_detectadas ni data/input.")
        return

    casos = casos_manuales or [(placa, buscar_por_placa(placa, rutas)) for placa in PLACAS_PRUEBA]
    for placa, ruta in casos:
        if ruta is None:
            print(f"{placa}: no se encontro recorte con esa placa en el nombre.")
            continue
        if not ruta.exists():
            print(f"{placa}: no existe el recorte indicado: {ruta}")
            continue

        resultado = leer_placa_desde_recorte(str(ruta), placa_esperada=placa, metodo_segmentacion="v2_banda_caracteres")
        reporte = registrar_diagnostico_recorte(resultado, placa)
        diagnostico = resultado.get("diagnostico", {})
        print(f"{placa}:")
        print(f"  Recorte: {ruta.relative_to(ROOT_DIR)}")
        print(f"  Texto crudo: {resultado.get('texto_detectado_crudo')}")
        print(f"  Texto postprocesado: {resultado.get('texto_postprocesado')}")
        print(f"  Esperados/detectados: {diagnostico.get('cantidad_esperada')} / {diagnostico.get('cantidad_detectada')}")
        print(f"  Causa probable: {diagnostico.get('causa_probable')}")
        print(f"  Mensaje: {diagnostico.get('mensaje')}")
        print(f"  Diagnostico CSV: {Path(reporte['ruta_csv']).relative_to(ROOT_DIR)}")

        for item in diagnostico.get("diagnostico_caracteres", []):
            print(
                "    "
                f"#{item.get('indice')} esperado={item.get('etiqueta_esperada') or '-'} "
                f"pred={item.get('prediccion') or '-'} conf={float(item.get('confianza', 0.0)):.2f} "
                f"estado={item.get('estado')} causa={item.get('causa_probable')}"
            )
        print("")


if __name__ == "__main__":
    main()

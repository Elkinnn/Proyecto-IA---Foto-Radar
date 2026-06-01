from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_reader import leer_placa_desde_recorte, registrar_reporte_ocr


def buscar_recortes() -> tuple[list[Path], str]:
    eventos_dir = Path("reports") / "evidencias" / "eventos_placa"
    placas_dir = Path("reports") / "evidencias" / "placas_detectadas"
    patrones = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    recortes_eventos = []
    for patron in patrones:
        recortes_eventos.extend(eventos_dir.glob(patron) if eventos_dir.exists() else [])
    recortes_eventos = [ruta for ruta in recortes_eventos if "recorte" in ruta.name.lower()]
    if recortes_eventos:
        return sorted(recortes_eventos, key=lambda ruta: ruta.stat().st_mtime, reverse=True), "eventos_placa"

    recortes_placas = []
    for patron in patrones:
        recortes_placas.extend(placas_dir.glob(patron) if placas_dir.exists() else [])
    return sorted(recortes_placas, key=lambda ruta: ruta.stat().st_mtime, reverse=True), "placas_detectadas"


def main() -> None:
    recortes, fuente = buscar_recortes()
    if not recortes:
        print("No hay recortes de placa para probar segmentacion v2.")
        print("Genere detecciones primero desde Monitoreo o cargue una imagen desde Streamlit.")
        return

    print(f"Fuente de recortes: {fuente}")
    for ruta in recortes[:5]:
        resultado = leer_placa_desde_recorte(str(ruta), metodo_segmentacion="v2_banda_caracteres")
        registrar_reporte_ocr(resultado, fuente)
        print("-" * 72)
        print(f"Ruta imagen: {ruta}")
        print(f"Caracteres aceptados: {resultado.get('cantidad_caracteres_segmentados', 0)}")
        print(f"Contornos rechazados: {resultado.get('cantidad_contornos_rechazados', 0)}")
        print(f"Ruta banda: {resultado.get('ruta_banda')}")
        print(f"Ruta debug: {resultado.get('ruta_debug_segmentacion')}")
        print(f"Motivos rechazo: {resultado.get('motivos_rechazo')}")
        print(f"Estado: {resultado.get('estado')}")


if __name__ == "__main__":
    main()

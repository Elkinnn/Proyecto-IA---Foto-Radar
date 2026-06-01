from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.plate_reader import leer_placa_desde_recorte, registrar_reporte_ocr


def buscar_recortes() -> list[Path]:
    eventos_dir = Path("reports") / "evidencias" / "eventos_placa"
    placas_dir = Path("reports") / "evidencias" / "placas_detectadas"
    patrones = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    recortes = []
    for patron in patrones:
        recortes.extend(eventos_dir.glob(patron) if eventos_dir.exists() else [])
    recortes = [ruta for ruta in recortes if "recorte" in ruta.name.lower()]
    if recortes:
        return sorted(recortes, key=lambda ruta: ruta.stat().st_mtime, reverse=True)

    for patron in patrones:
        recortes.extend(placas_dir.glob(patron) if placas_dir.exists() else [])
    return sorted(recortes, key=lambda ruta: ruta.stat().st_mtime, reverse=True)


def main() -> None:
    recortes = buscar_recortes()
    if not recortes:
        print("No hay recortes para probar OCR con CNN.")
        return

    print("Prueba OCR experimental con CNN propia")
    print("-------------------------------------")
    for ruta in recortes[:10]:
        resultado = leer_placa_desde_recorte(str(ruta), metodo_segmentacion="v2_banda_caracteres")
        registrar_reporte_ocr(resultado, "eventos_placa")
        print(f"Ruta imagen: {ruta}")
        print(f"Texto crudo: {resultado.get('texto_detectado_crudo')}")
        print(f"Texto postprocesado: {resultado.get('texto_postprocesado')}")
        print(f"Confianza: {resultado.get('confianza_promedio', 0.0):.4f}")
        print(f"Formato valido: {resultado.get('formato', {}).get('valido')}")
        print(f"Caracteres segmentados: {resultado.get('cantidad_caracteres_segmentados', 0)}")
        print(f"Estado: {resultado.get('estado')}")
        print("-" * 72)


if __name__ == "__main__":
    main()

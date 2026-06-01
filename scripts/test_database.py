from datetime import datetime
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd, listar_eventos, listar_vehiculos
from src.fuzzy_system import clasificar_velocidad


def main() -> None:
    ruta_bd = "data/database/fotorradar.db"
    inicializar_bd(ruta_bd)
    print(f"Base de datos inicializada: {ruta_bd}")

    print("\nVehiculos registrados:")
    vehiculos = listar_vehiculos(ruta_bd)
    for vehiculo in vehiculos:
        print(
            f"- {vehiculo['placa']} | {vehiculo.get('marca')} | {vehiculo.get('modelo')} | "
            f"{vehiculo.get('color')} | {vehiculo.get('propietario')} | {vehiculo.get('correo')} | {vehiculo.get('estado')}"
        )

    placa = "PBC1234"
    vehiculo = buscar_vehiculo_por_placa(placa, ruta_bd)
    print(f"\nBusqueda placa {placa}:")
    print(vehiculo)

    velocidad = 40.0
    limite = 30.0
    difuso = clasificar_velocidad(velocidad, limite)
    evento_id = guardar_evento(
        {
            "fecha_hora": datetime.now().isoformat(timespec="seconds"),
            "placa": placa,
            "vehiculo": vehiculo,
            "velocidad_kmh": velocidad,
            "limite_kmh": limite,
            "estado_difuso": difuso["estado"],
            "nivel_infraccion": difuso["nivel_infraccion"],
            "sancion": difuso["sancion"],
            "horas_suspension": difuso["horas_suspension"],
            "mensaje": difuso["mensaje"],
            "evidencia_frame": "reports/evidencias/test/frame.jpg",
            "evidencia_placa": "reports/evidencias/test/placa.jpg",
            "fuente": "test_database.py",
        },
        ruta_bd,
    )
    print(f"\nEvento de prueba guardado con ID: {evento_id}")

    print("\nEventos recientes:")
    for evento in listar_eventos(ruta_bd, limit=10):
        print(
            f"- #{evento['id']} | {evento.get('fecha_hora')} | {evento.get('placa')} | "
            f"{evento.get('velocidad_kmh')} km/h | {evento.get('estado_difuso')} | {evento.get('sancion')}"
        )


if __name__ == "__main__":
    main()

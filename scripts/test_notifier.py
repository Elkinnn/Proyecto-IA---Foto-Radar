from datetime import datetime
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import buscar_vehiculo_por_placa, guardar_evento, inicializar_bd
from src.fuzzy_system import clasificar_velocidad
from src.notifier import generar_notificacion_simulada, guardar_notificacion_simulada


def main() -> None:
    ruta_bd = "data/database/fotorradar.db"
    inicializar_bd(ruta_bd)

    placa = "PBC1234"
    vehiculo = buscar_vehiculo_por_placa(placa, ruta_bd)
    if not vehiculo:
        raise RuntimeError(f"No se encontro vehiculo de prueba para placa {placa}.")

    velocidad = 40.0
    limite = 30.0
    resultado_difuso = clasificar_velocidad(velocidad, limite)
    evento = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "placa": placa,
        "vehiculo": vehiculo,
        "velocidad_kmh": velocidad,
        "limite_kmh": limite,
        "estado_difuso": resultado_difuso["estado"],
        "nivel_infraccion": resultado_difuso["nivel_infraccion"],
        "sancion": resultado_difuso["sancion"],
        "horas_suspension": resultado_difuso["horas_suspension"],
        "mensaje": resultado_difuso["mensaje"],
        "evidencia_frame": "reports/evidencias/test_notifier/frame.jpg",
        "evidencia_placa": "reports/evidencias/test_notifier/placa.jpg",
        "fuente": "test_notifier.py",
    }
    evento_id = guardar_evento(evento, ruta_bd)

    notificacion = generar_notificacion_simulada(evento, vehiculo, resultado_difuso)
    if not notificacion:
        raise RuntimeError("No se genero notificacion, pero el caso de prueba debe tener infraccion.")

    rutas = guardar_notificacion_simulada(notificacion, evento_id)
    notificacion.update(rutas)

    print(f"Evento guardado: {evento_id}")
    print(f"Destinatario: {notificacion['destinatario']}")
    print(f"Asunto: {notificacion['asunto']}")
    print(f"TXT: {notificacion['ruta_txt']}")
    print(f"JSON: {notificacion['ruta_json']}")


if __name__ == "__main__":
    main()

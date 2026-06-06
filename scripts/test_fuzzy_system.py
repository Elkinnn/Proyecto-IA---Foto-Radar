from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.fuzzy_system import clasificar_velocidad
from src.utils import cargar_config


def main() -> None:
    config = cargar_config()
    limite = float(config["speed"]["campus_speed_limit_kmh"])
    casos = [15, 28, 32, 40, 52, 65]

    print("Prueba de logica difusa (multa/sancion)")
    print(f"Limite del campus: {limite:.1f} km/h")
    print("-" * 72)

    for velocidad in casos:
        resultado = clasificar_velocidad(velocidad, limite, config)
        print(f"Velocidad: {velocidad:.1f} km/h")
        print(f"Estado: {resultado['estado']}")
        print(f"Multa: {resultado['multa_texto']}")
        print(f"Nivel: {resultado['nivel_infraccion']}")
        print(f"Horas suspension: {resultado['horas_suspension']}")
        print(f"Mensaje: {resultado['mensaje']}")
        print(f"Grados: {resultado['grados']}")
        print("-" * 72)


if __name__ == "__main__":
    main()

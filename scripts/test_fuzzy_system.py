from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.fuzzy_system import clasificar_velocidad, obtener_datos_visualizacion
from src.utils import cargar_config


def main() -> None:
    config = cargar_config()
    limite = float(config["speed"]["campus_speed_limit_kmh"])
    casos = [15, 28, 32, 35, 40, 52, 65]
    resultados = {velocidad: clasificar_velocidad(velocidad, limite, config) for velocidad in casos}

    assert resultados[15]["metodo"] == "mamdani_centroide"
    assert resultados[15]["estado"] == "Seguro"
    assert resultados[15]["multa_usd"] == 0
    assert resultados[15]["nivel_sancion_defuzzificado"] < 10
    assert resultados[28]["sancion_aplica"] is False
    assert resultados[32]["nivel_infraccion"] == "Advertencia"
    assert resultados[35]["multa_usd"] == 0
    assert resultados[40]["sancion_aplica"] is True
    assert resultados[40]["nivel_infraccion"] == "Leve"
    assert resultados[52]["nivel_infraccion"] == "Grave"
    assert resultados[52]["multa_usd"] > resultados[40]["multa_usd"]

    limite_alternativo = clasificar_velocidad(40, 40, config)
    assert limite_alternativo["estado"] == "Permitido"
    assert limite_alternativo["sancion_aplica"] is False
    assert limite_alternativo["multa_usd"] == 0

    for resultado in resultados.values():
        assert resultado["grados_entrada"]
        assert resultado["activaciones_salida"]
        assert resultado["reglas_activas"]
        assert resultado["operadores"] == {
            "implicacion": "minimo",
            "agregacion": "maximo",
            "defuzzificacion": "centroide",
        }

    visual = obtener_datos_visualizacion(40, limite, config)
    assert visual["curvas_entrada"]
    assert visual["curvas_salida"]
    assert len(visual["universo"]) == len(visual["agregada"])

    print("Prueba de logica difusa (multa/sancion)")
    print(f"Limite del campus: {limite:.1f} km/h")
    print("-" * 72)

    for velocidad in casos:
        resultado = resultados[velocidad]
        print(f"Velocidad: {velocidad:.1f} km/h")
        print(f"Estado: {resultado['estado']}")
        print(f"Multa: {resultado['multa_texto']}")
        print(f"Nivel: {resultado['nivel_infraccion']}")
        print(f"Horas suspension: {resultado['horas_suspension']}")
        print(f"Mensaje: {resultado['mensaje']}")
        print(f"Grados: {resultado['grados']}")
        print("-" * 72)
    print("Resultado: OK - fuzzificacion, reglas, agregacion y defuzzificacion verificadas.")


if __name__ == "__main__":
    main()

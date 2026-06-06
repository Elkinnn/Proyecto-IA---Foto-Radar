from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.notifier import evaluar_calidad_evento, notificar_evento_placa
from src.utils import cargar_config


def main() -> None:
    config = cargar_config()
    destinatario = sys.argv[1] if len(sys.argv) > 1 else "destino@example.com"
    placa = "PBC1234"

    # Caso 1: placa completa y valida -> debe pasar el candado de calidad.
    metricas_ok = {
        "formato_valido": True,
        "confianza_final": 0.92,
        "aspect_ratio": 3.1,
        "nitidez": 120.0,
        "cerca_borde": False,
        "cantidad_caracteres": 7,
    }
    print("Evaluacion placa completa:", evaluar_calidad_evento({**metricas_ok, "placa": placa}, config))

    # Caso 2: media placa -> debe ser descartada.
    metricas_media = {
        "formato_valido": False,
        "confianza_final": 0.95,
        "aspect_ratio": 1.1,
        "nitidez": 120.0,
        "cerca_borde": True,
        "cantidad_caracteres": 4,
    }
    print("Evaluacion media placa:", evaluar_calidad_evento({**metricas_media, "placa": "PB1"}, config))

    resultado = notificar_evento_placa(
        destinatario=destinatario,
        placa=placa,
        metricas_calidad=metricas_ok,
        adjuntos=[],
        config=config,
        evento_id="test",
    )
    print("Resultado envio:", resultado.get("estado"), "| modo:", resultado.get("modo"))
    print(resultado.get("mensaje_estado"))


if __name__ == "__main__":
    main()

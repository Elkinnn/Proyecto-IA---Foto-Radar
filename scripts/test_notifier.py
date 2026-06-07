from pathlib import Path
from unittest.mock import patch
import sys

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.notifier import evaluar_calidad_evento, notificar_evento_placa
from src.utils import cargar_config


def main() -> None:
    config = cargar_config()
    placa = "PBC1234"
    metricas_ok = {
        "formato_valido": True,
        "confianza_final": 0.92,
        "aspect_ratio": 3.1,
        "nitidez": 120.0,
        "cerca_borde": False,
        "cantidad_caracteres": 7,
    }
    assert evaluar_calidad_evento({**metricas_ok, "placa": placa}, config)["apto"]

    evidencia = Path("reports/evidencias/notificaciones/test_evento_real.jpg")
    evidencia.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(evidencia), np.zeros((120, 320, 3), dtype=np.uint8))

    contexto = {
        "evento_id": 77,
        "fecha_hora": "2026-06-06T14:30:15",
        "fuente": "Camara en vivo",
        "velocidad_kmh": 40.0,
        "limite_kmh": 30.0,
        "distancia_metros": 10.0,
        "tiempo_entre_lineas": 0.9,
        "confianza_ocr": 0.92,
        "confianza_yolo": 0.88,
        "evidencia_principal": str(evidencia),
    }
    capturado = {}

    def enviar_falso(**kwargs):
        capturado.update(kwargs)
        return {
            "enviado": True,
            "modo": "prueba",
            "estado": "enviado",
            "destinatario": kwargs["destinatario"],
            "mensaje": kwargs["mensaje"],
            "mensaje_html": kwargs["mensaje_html"],
            "adjuntos": kwargs["adjuntos"],
        }

    with patch("src.notifier.enviar_correo", side_effect=enviar_falso):
        resultado = notificar_evento_placa(
            destinatario="destino@example.com",
            placa=placa,
            metricas_calidad=metricas_ok,
            adjuntos=[str(evidencia)],
            config=config,
            evento_id=77,
            contexto=contexto,
            velocidad_kmh=40.0,
            limite_kmh=30.0,
        )

    assert resultado["enviado"]
    assert resultado["clasificacion_difusa"]["velocidad_kmh"] == 40.0
    assert resultado["clasificacion_difusa"]["sancion_aplica"] is True
    assert resultado["contexto_evento"]["fecha_hora"] == contexto["fecha_hora"]
    assert "40.00 km/h" in capturado["mensaje"]
    assert "10.00 m" in capturado["mensaje"]
    assert "valores de demostracion" not in capturado["mensaje"].lower()
    assert "40.00 km/h" in capturado["mensaje_html"]
    assert "Multa leve" in capturado["mensaje_html"]
    assert "cid:evidencia_evento" in capturado["mensaje_html"]
    assert capturado["evidencia_principal"] == str(evidencia)
    assert capturado["contexto_evento"]["fecha_hora"] == contexto["fecha_hora"]

    sin_velocidad = notificar_evento_placa(
        destinatario="destino@example.com",
        placa=placa,
        metricas_calidad=metricas_ok,
        adjuntos=[str(evidencia)],
        config=config,
        evento_id=78,
        contexto={**contexto, "velocidad_kmh": None},
        velocidad_kmh=None,
        limite_kmh=30.0,
    )
    assert sin_velocidad["estado"] == "velocidad_no_medida"
    assert sin_velocidad["enviado"] is False

    print("Notificador con datos reales")
    print("----------------------------")
    print("Velocidad real y logica difusa: OK")
    print("Fecha, placa y evidencia del evento: OK")
    print("Plantilla HTML profesional: OK")
    print("Evento sin velocidad real bloqueado: OK")
    print("Resultado: OK")


if __name__ == "__main__":
    main()

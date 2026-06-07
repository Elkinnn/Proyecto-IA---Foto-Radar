from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from src.utils import cargar_config


def _entry(evento_id: int, placa: str, frame: int, ruta_recorte: str, ruta_frame: str) -> dict:
    return {
        "estado": "lectura_disponible",
        "evento_cerrado": True,
        "ruta_recorte": ruta_recorte,
        "resumen_base": {
            "evento_id": evento_id,
            "fecha_hora_evento": f"2026-06-06T12:{evento_id:02d}:00",
            "frame_actual": frame,
            "frame_mejor_evento": frame,
            "ruta_mejor_frame_evento": ruta_frame,
            "mejor_recorte_placa": ruta_recorte,
            "ruta_mejor_recorte_evento": ruta_recorte,
            "limite_velocidad_kmh": 37.0,
            "velocidad": {
                "velocidad_kmh": 40.0 + evento_id,
                "distancia_metros": 10.0,
                "tiempo_entre_lineas": 0.9,
                "frame_cruce_linea_1": frame - 20,
                "frame_cruce_linea_2": frame,
            },
            "mejor_recorte_placa_info": {
                "aspect_ratio": 3.0,
                "nitidez": 120.0,
                "cerca_borde": False,
            },
        },
        "datos": {
            "placa_consolidada_evento": placa,
            "placa_individual": placa,
            "formato_consolidado_valido": True,
            "formato_ocr_valido": True,
            "confianza_final_evento": 0.95,
            "confianza_ocr": 0.95,
            "cantidad_caracteres_segmentados_ocr": len(placa),
            "lecturas_usadas_evento": 2,
        },
    }


def main() -> None:
    config = cargar_config()
    config["notificaciones"]["envio_automatico"] = True
    llamadas: list[tuple[int, str]] = []
    payloads: list[dict] = []

    def notificar_falso(**kwargs):
        llamadas.append((int(kwargs["evento_id"]), str(kwargs["placa"])))
        payloads.append(kwargs)
        return {
            "enviado": True,
            "modo": "smtp",
            "estado": "enviado",
            "destinatario": kwargs["destinatario"],
            "clasificacion_difusa": {"estado": "Prueba"},
        }

    app.notificar_evento_placa = notificar_falso
    app._inicializar_estado_monitoreo()
    app._iniciar_contexto_lectura_monitoreo()
    app.st.session_state.correo_destino_monitoreo = "destino@example.com"

    ruta_recorte = Path("reports/evidencias/notificaciones/test_recorte_email.jpg")
    ruta_frame = Path("reports/evidencias/notificaciones/test_mejor_frame_email.jpg")
    ruta_recorte.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_recorte), np.zeros((64, 180, 3), dtype=np.uint8))
    cv2.imwrite(str(ruta_frame), np.zeros((360, 640, 3), dtype=np.uint8))

    app.st.session_state.ocr_eventos_cache = {
        1: _entry(1, "TBH4543", 100, str(ruta_recorte), str(ruta_frame)),
        2: _entry(2, "ABC0123", 110, str(ruta_recorte), str(ruta_frame)),
    }
    app._procesar_email_inmediato_eventos_reconocidos(config, "PBC1234")
    for _ in range(100):
        if not app._hay_email_asincrono_pendiente():
            break
        time.sleep(0.01)
    app._fusionar_emails_asincronos_monitoreo()

    assert llamadas == [(1, "TBH4543"), (2, "ABC0123")], llamadas
    assert payloads[0]["velocidad_kmh"] == 41.0
    assert payloads[1]["velocidad_kmh"] == 42.0
    assert payloads[0]["contexto"]["fecha_hora"] == "2026-06-06T12:01:00"
    assert payloads[1]["contexto"]["fecha_hora"] == "2026-06-06T12:02:00"
    assert payloads[0]["limite_kmh"] == 37.0
    assert payloads[0]["contexto"]["evidencia_principal"] == str(ruta_frame)

    app._procesar_email_inmediato_eventos_reconocidos(config, "PBC1234")
    assert len(llamadas) == 2, llamadas

    app._iniciar_contexto_lectura_monitoreo()
    app.st.session_state.ocr_eventos_cache = {
        10: _entry(10, "PBC1234", 200, str(ruta_recorte), str(ruta_frame)),
        11: _entry(11, "PBC1234", 205, str(ruta_recorte), str(ruta_frame)),
    }
    app._procesar_email_inmediato_eventos_reconocidos(config, "PBC1234")
    for _ in range(100):
        if not app._hay_email_asincrono_pendiente():
            break
        time.sleep(0.01)
    app._fusionar_emails_asincronos_monitoreo()
    assert llamadas[-1] == (10, "PBC1234"), llamadas
    assert len([llamada for llamada in llamadas if llamada[1] == "PBC1234"]) == 1, llamadas
    assert app.st.session_state.ocr_eventos_cache[11]["email_estado"] == "suprimido_duplicado"

    cache = app.st.session_state.ocr_eventos_cache
    cache[3] = _entry(3, "PBC1234", 220, str(ruta_recorte), str(ruta_frame))
    app.st.session_state.ocr_eventos_cache = cache
    app._procesar_email_inmediato_eventos_reconocidos(config, "PBC1234")
    assert len(llamadas) == 3, llamadas
    assert app.st.session_state.ocr_eventos_cache[3]["email_estado"] == "suprimido_duplicado"

    grupos_distintos = app._agrupar_cola_por_paso_vehiculo(
        [
            {"evento_id": 1, "frame_mejor": 100, "resumen": {"placa_consolidada_evento": "TBH4543"}},
            {"evento_id": 2, "frame_mejor": 110, "resumen": {"placa_consolidada_evento": "ABC0123"}},
        ],
        150,
    )
    assert len(grupos_distintos) == 2

    print("Envio automatico monitoreo")
    print("--------------------------")
    print("Dos placas distintas: 2 correos")
    print("Mismo evento repetido: 0 correos adicionales")
    print("Misma placa lista simultaneamente: 1 solo correo")
    print("Misma placa fragmentada: correo duplicado suprimido")
    print("Vehiculos consecutivos con placas distintas: grupos separados")
    print("Velocidad, fecha y evidencia: pertenecen al evento correcto")
    print("Resultado: OK")


if __name__ == "__main__":
    main()

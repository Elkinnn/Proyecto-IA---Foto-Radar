def enviar_notificacion_sancion(vehiculo: dict | None, resultado: dict, config: dict) -> dict:
    email_config = config.get("email", {})
    if email_config.get("simulation_mode", True):
        return {
            "enviado": False,
            "modo": "simulacion",
            "destinatario": vehiculo.get("correo") if vehiculo else None,
            "mensaje": "Notificacion simulada; no se usaron credenciales SMTP.",
        }

    return {
        "enviado": False,
        "modo": "pendiente",
        "destinatario": vehiculo.get("correo") if vehiculo else None,
        "mensaje": "Envio real pendiente de configurar.",
    }

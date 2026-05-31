def clasificar_velocidad(velocidad_kmh: float, config: dict) -> dict:
    reglas = config["fuzzy"]
    suspensiones = reglas.get("suspension_horas", {})

    if velocidad_kmh <= reglas["seguro_hasta"]:
        estado = "Seguro"
        clave = "seguro"
        mensaje = "Velocidad segura para circulacion dentro del campus."
    elif velocidad_kmh <= reglas["permitido_hasta"]:
        estado = "Permitido"
        clave = "permitido"
        mensaje = "Velocidad dentro del limite permitido."
    elif velocidad_kmh <= reglas["precaucion_hasta"]:
        estado = "Precaucion"
        clave = "precaucion"
        mensaje = "Velocidad cercana al exceso; se recomienda advertencia."
    elif velocidad_kmh <= reglas["exceso_leve_hasta"]:
        estado = "Exceso leve"
        clave = "exceso_leve"
        mensaje = "Exceso leve de velocidad en zona universitaria."
    elif velocidad_kmh <= reglas["exceso_medio_hasta"]:
        estado = "Exceso medio"
        clave = "exceso_medio"
        mensaje = "Exceso medio de velocidad; corresponde sancion."
    else:
        estado = "Exceso alto"
        clave = "exceso_alto"
        mensaje = "Exceso alto de velocidad; corresponde sancion maxima configurada."

    horas = int(suspensiones.get(clave, 0))
    sancion = horas > 0

    return {
        "estado": estado,
        "sancion": sancion,
        "horas_suspension": horas,
        "mensaje": mensaje,
    }

def triangular(x: float, a: float, b: float, c: float) -> float:
    if a == b and x <= b:
        return 1.0
    if b == c and x >= b:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def _obtener_limite(limite_kmh=30.0) -> float:
    if isinstance(limite_kmh, dict):
        speed = limite_kmh.get("speed", {})
        return float(speed.get("campus_speed_limit_kmh", 30.0))
    return float(limite_kmh)


def _calcular_grados(velocidad_kmh: float, limite_kmh: float) -> dict:
    return {
        "seguro": round(trapezoidal(velocidad_kmh, 0.0, 0.0, limite_kmh * 0.45, limite_kmh * 0.70), 4),
        "permitido": round(triangular(velocidad_kmh, limite_kmh * 0.60, limite_kmh * 0.88, limite_kmh), 4),
        "precaucion": round(triangular(velocidad_kmh, limite_kmh * 0.95, limite_kmh + 2.5, limite_kmh + 5.0), 4),
        "exceso_leve": round(trapezoidal(velocidad_kmh, limite_kmh + 3.0, limite_kmh + 5.0, limite_kmh + 12.0, limite_kmh + 15.0), 4),
        "exceso_medio": round(trapezoidal(velocidad_kmh, limite_kmh + 12.0, limite_kmh + 15.0, limite_kmh + 25.0, limite_kmh + 30.0), 4),
        "exceso_alto": round(trapezoidal(velocidad_kmh, limite_kmh + 25.0, limite_kmh + 30.0, limite_kmh + 120.0, limite_kmh + 140.0), 4),
    }


def clasificar_velocidad(velocidad_kmh: float, limite_kmh: float = 30.0) -> dict:
    limite = _obtener_limite(limite_kmh)
    velocidad = float(velocidad_kmh)

    if velocidad <= limite * 0.70:
        estado = "Seguro"
        nivel = "Sin infracción"
        sancion = "No aplica"
        horas = 0
        mensaje = "Velocidad segura para circulación dentro del campus."
    elif velocidad <= limite:
        estado = "Permitido"
        nivel = "Sin infracción"
        sancion = "No aplica"
        horas = 0
        mensaje = "Velocidad dentro del límite permitido."
    elif velocidad <= limite + 5:
        estado = "Precaución"
        nivel = "Advertencia"
        sancion = "Advertencia verbal o notificación preventiva"
        horas = 0
        mensaje = "Velocidad ligeramente sobre el límite; corresponde advertencia preventiva."
    elif velocidad <= limite + 15:
        estado = "Exceso leve"
        nivel = "Leve"
        sancion = "Notificación de infracción leve"
        horas = 0
        mensaje = "Exceso leve de velocidad en zona universitaria."
    elif velocidad <= limite + 30:
        estado = "Exceso medio"
        nivel = "Media"
        sancion = "Suspensión temporal de acceso vehicular"
        horas = 24
        mensaje = "Exceso medio de velocidad; corresponde suspensión temporal."
    else:
        estado = "Exceso alto"
        nivel = "Grave"
        sancion = "Suspensión extendida de acceso vehicular"
        horas = 72
        mensaje = "Exceso alto de velocidad; corresponde suspensión extendida."

    return {
        "velocidad_kmh": velocidad,
        "limite_kmh": limite,
        "estado": estado,
        "nivel_infraccion": nivel,
        "sancion": sancion,
        "sancion_aplica": horas > 0 or nivel != "Sin infracción",
        "horas_suspension": horas,
        "mensaje": mensaje,
        "grados": _calcular_grados(velocidad, limite),
    }

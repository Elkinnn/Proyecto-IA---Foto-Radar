def calcular_velocidad_kmh(distancia_metros: float, tiempo_segundos: float) -> float:
    if distancia_metros <= 0:
        raise ValueError("La distancia debe ser mayor que cero.")
    if tiempo_segundos <= 0:
        raise ValueError("El tiempo debe ser mayor que cero.")

    velocidad_m_s = distancia_metros / tiempo_segundos
    return velocidad_m_s * 3.6


def estimar_velocidad(config: dict, tiempo_segundos: float | None = None) -> float:
    distancia = float(config["speed"]["default_distance_meters"])
    tiempo = float(tiempo_segundos or config["speed"]["default_time_seconds"])
    return calcular_velocidad_kmh(distancia, tiempo)

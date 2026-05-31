def calcular_velocidad_kmh(distancia_metros: float, tiempo_segundos: float) -> float:
    if distancia_metros <= 0:
        raise ValueError("La distancia debe ser mayor que cero.")
    if tiempo_segundos <= 0:
        raise ValueError("El tiempo debe ser mayor que cero.")

    velocidad_m_s = distancia_metros / tiempo_segundos
    return velocidad_m_s * 3.6


class SpeedTracker:
    def __init__(self, linea_1_y: int, linea_2_y: int, distancia_metros: float, fps: float):
        if distancia_metros <= 0:
            raise ValueError("La distancia debe ser mayor que cero.")

        self.linea_1_y = int(linea_1_y)
        self.linea_2_y = int(linea_2_y)
        self.distancia_metros = float(distancia_metros)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_cruce_linea_1 = None
        self.frame_cruce_linea_2 = None
        self.tiempo_cruce_linea_1 = None
        self.tiempo_cruce_linea_2 = None
        self.tiempo_entre_lineas = None
        self.velocidad_kmh = None
        self.estado = "esperando_linea_1"
        self.centro_y_anterior = None

    def actualizar(self, bbox: list[int], numero_frame: int) -> dict:
        x1, y1, x2, y2 = bbox
        centro_x = (x1 + x2) / 2
        centro_y = (y1 + y2) / 2

        if self.centro_y_anterior is not None:
            if self.estado == "esperando_linea_1" and self._cruzo_hacia_abajo(self.linea_1_y, centro_y):
                self.frame_cruce_linea_1 = numero_frame
                self.tiempo_cruce_linea_1 = numero_frame / self.fps
                self.estado = "esperando_linea_2"
            elif self.estado == "esperando_linea_2" and self._cruzo_hacia_abajo(self.linea_2_y, centro_y):
                self.frame_cruce_linea_2 = numero_frame
                self.tiempo_cruce_linea_2 = numero_frame / self.fps
                self.tiempo_entre_lineas = self.tiempo_cruce_linea_2 - self.tiempo_cruce_linea_1
                self.velocidad_kmh = calcular_velocidad_kmh(self.distancia_metros, self.tiempo_entre_lineas)
                self.estado = "velocidad_calculada"

        self.centro_y_anterior = centro_y
        return self.resumen(centro_x=centro_x, centro_y=centro_y)

    def _cruzo_hacia_abajo(self, linea_y: int, centro_y_actual: float) -> bool:
        return self.centro_y_anterior < linea_y <= centro_y_actual

    def resumen(self, centro_x: float | None = None, centro_y: float | None = None) -> dict:
        return {
            "estado": self.estado,
            "frame_cruce_linea_1": self.frame_cruce_linea_1,
            "frame_cruce_linea_2": self.frame_cruce_linea_2,
            "tiempo_cruce_linea_1": self.tiempo_cruce_linea_1,
            "tiempo_cruce_linea_2": self.tiempo_cruce_linea_2,
            "tiempo_entre_lineas": self.tiempo_entre_lineas,
            "distancia_metros": self.distancia_metros,
            "fps": self.fps,
            "velocidad_kmh": self.velocidad_kmh,
            "centro_x": centro_x,
            "centro_y": centro_y,
        }


def estimar_velocidad(config: dict, tiempo_segundos: float | None = None) -> float:
    distancia = float(config["speed"]["default_distance_meters"])
    tiempo = float(tiempo_segundos or config["speed"]["default_time_seconds"])
    return calcular_velocidad_kmh(distancia, tiempo)

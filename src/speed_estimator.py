"""Medicion de velocidad tipo tic-toc entre dos lineas virtuales."""

import math

METODO_MEDICION = "tic_toc_dos_lineas_subframe"
METODO_MEDICION_TIEMPO_REAL = "tic_toc_dos_lineas_reloj_monotonico"


def calcular_velocidad_kmh(distancia_metros: float, tiempo_segundos: float) -> float:
    if distancia_metros <= 0:
        raise ValueError("La distancia debe ser mayor que cero.")
    if tiempo_segundos <= 0:
        raise ValueError("El tiempo debe ser mayor que cero.")

    velocidad_m_s = distancia_metros / tiempo_segundos
    return velocidad_m_s * 3.6


def interpolar_frame_cruce(
    frame_anterior: int,
    frame_actual: int,
    y_anterior: float,
    y_actual: float,
    linea_y: float,
) -> float:
    """Estima el frame fraccionario exacto donde el centro cruza linea_y."""
    if y_anterior == y_actual:
        return float(frame_actual)
    if not (y_anterior < linea_y <= y_actual):
        return float(frame_actual)
    fraccion = (linea_y - y_anterior) / (y_actual - y_anterior)
    fraccion = max(0.0, min(1.0, fraccion))
    return float(frame_anterior) + fraccion * (float(frame_actual) - float(frame_anterior))


def interpolar_tiempo_cruce(
    tiempo_anterior: float,
    tiempo_actual: float,
    y_anterior: float,
    y_actual: float,
    linea_y: float,
) -> float:
    """Interpola el instante tic/toc entre dos observaciones de la placa."""
    if tiempo_actual <= tiempo_anterior or y_anterior == y_actual:
        return float(tiempo_actual)
    if not (y_anterior < linea_y <= y_actual):
        return float(tiempo_actual)
    fraccion = (linea_y - y_anterior) / (y_actual - y_anterior)
    fraccion = max(0.0, min(1.0, fraccion))
    return float(tiempo_anterior) + fraccion * (float(tiempo_actual) - float(tiempo_anterior))


class SpeedTracker:
    def __init__(self, linea_1_y: int, linea_2_y: int, distancia_metros: float, fps: float):
        if distancia_metros <= 0:
            raise ValueError("La distancia debe ser mayor que cero.")

        self.linea_1_y = int(linea_1_y)
        self.linea_2_y = int(linea_2_y)
        self.distancia_metros = float(distancia_metros)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.min_tiempo_entre_lineas = max(2.0 / self.fps, 0.04)
        self.max_tiempo_entre_lineas = 30.0
        self.max_salto_y_por_frame = max(120.0, abs(self.linea_2_y - self.linea_1_y) * 2.5)

        self.frame_cruce_linea_1 = None
        self.frame_cruce_linea_2 = None
        self.frame_cruce_linea_1_exacto = None
        self.frame_cruce_linea_2_exacto = None
        self.tiempo_cruce_linea_1 = None
        self.tiempo_cruce_linea_2 = None
        self.tiempo_entre_lineas = None
        self.velocidad_kmh = None
        self.estado = "esperando_linea_1"
        self.motivo_invalido = None
        self.metodo_medicion = METODO_MEDICION

        self.frame_anterior = None
        self.centro_y_anterior = None
        self.centro_x_anterior = None
        self.timestamp_origen = None
        self.timestamp_anterior = None
        self.modo_tiempo = None

    def actualizar(
        self,
        bbox: list[int],
        numero_frame: int,
        timestamp_segundos: float | None = None,
    ) -> dict:
        x1, y1, x2, y2 = bbox
        centro_x = (x1 + x2) / 2.0
        centro_y = (y1 + y2) / 2.0
        timestamp = self._normalizar_timestamp(timestamp_segundos)
        tiempo_observacion = self._tiempo_observacion(numero_frame, timestamp)

        if self.frame_anterior is not None and self.centro_y_anterior is not None:
            if self.estado == "esperando_linea_1" and self._cruzo_hacia_abajo(self.linea_1_y, centro_y):
                frame_exacto = interpolar_frame_cruce(
                    self.frame_anterior,
                    numero_frame,
                    self.centro_y_anterior,
                    centro_y,
                    float(self.linea_1_y),
                )
                self.frame_cruce_linea_1_exacto = frame_exacto
                self.frame_cruce_linea_1 = int(numero_frame)
                self.tiempo_cruce_linea_1 = self._interpolar_tiempo_cruce(
                    frame_exacto,
                    tiempo_observacion,
                    centro_y,
                    self.linea_1_y,
                )
                self.estado = "esperando_linea_2"
                self.motivo_invalido = None
            if self.estado == "esperando_linea_2" and self._cruzo_hacia_abajo(self.linea_2_y, centro_y):
                frame_exacto = interpolar_frame_cruce(
                    self.frame_anterior,
                    numero_frame,
                    self.centro_y_anterior,
                    centro_y,
                    float(self.linea_2_y),
                )
                self.frame_cruce_linea_2_exacto = frame_exacto
                self.frame_cruce_linea_2 = int(numero_frame)
                self.tiempo_cruce_linea_2 = self._interpolar_tiempo_cruce(
                    frame_exacto,
                    tiempo_observacion,
                    centro_y,
                    self.linea_2_y,
                )
                self.tiempo_entre_lineas = self.tiempo_cruce_linea_2 - self.tiempo_cruce_linea_1
                if self._tiempo_entre_lineas_valido(self.tiempo_entre_lineas):
                    self.velocidad_kmh = calcular_velocidad_kmh(self.distancia_metros, self.tiempo_entre_lineas)
                    self.estado = "velocidad_calculada"
                    self.motivo_invalido = None
                else:
                    self.velocidad_kmh = None
                    self.estado = "medicion_invalida"
                    self.motivo_invalido = self._motivo_tiempo_invalido(self.tiempo_entre_lineas)

        self.frame_anterior = int(numero_frame)
        self.centro_y_anterior = centro_y
        self.centro_x_anterior = centro_x
        self.timestamp_anterior = tiempo_observacion
        return self.resumen(centro_x=centro_x, centro_y=centro_y)

    def _normalizar_timestamp(self, timestamp_segundos: float | None) -> float | None:
        timestamp_valido = None
        if timestamp_segundos is not None:
            candidato = float(timestamp_segundos)
            if math.isfinite(candidato):
                timestamp_valido = candidato
        if self.modo_tiempo is None:
            self.modo_tiempo = "reloj_monotonico" if timestamp_valido is not None else "frames_fps"
            if timestamp_valido is not None:
                self.metodo_medicion = METODO_MEDICION_TIEMPO_REAL
        if self.modo_tiempo != "reloj_monotonico" or timestamp_valido is None:
            return None
        if self.timestamp_origen is None:
            self.timestamp_origen = timestamp_valido
        return max(0.0, timestamp_valido - self.timestamp_origen)

    def _tiempo_observacion(self, numero_frame: int, timestamp: float | None) -> float:
        if self.modo_tiempo == "reloj_monotonico" and timestamp is not None:
            return float(timestamp)
        if self.modo_tiempo == "reloj_monotonico" and self.timestamp_anterior is not None and self.frame_anterior is not None:
            frames_transcurridos = max(int(numero_frame) - int(self.frame_anterior), 0)
            return float(self.timestamp_anterior) + frames_transcurridos / self.fps
        return float(numero_frame) / self.fps

    def _interpolar_tiempo_cruce(
        self,
        frame_exacto: float,
        tiempo_actual: float,
        centro_y_actual: float,
        linea_y: float,
    ) -> float:
        if (
            self.modo_tiempo == "reloj_monotonico"
            and self.timestamp_anterior is not None
            and tiempo_actual >= self.timestamp_anterior
        ):
            return interpolar_tiempo_cruce(
                self.timestamp_anterior,
                tiempo_actual,
                float(self.centro_y_anterior),
                centro_y_actual,
                linea_y,
            )
        return float(frame_exacto) / self.fps

    def _tiempo_entre_lineas_valido(self, tiempo: float | None) -> bool:
        if tiempo is None:
            return False
        return self.min_tiempo_entre_lineas <= float(tiempo) <= self.max_tiempo_entre_lineas

    def _motivo_tiempo_invalido(self, tiempo: float | None) -> str:
        if tiempo is None:
            return "No se pudo calcular el tiempo entre lineas."
        if tiempo < self.min_tiempo_entre_lineas:
            return "Tiempo demasiado corto (posible salto del bbox)."
        if tiempo > self.max_tiempo_entre_lineas:
            return "Tiempo demasiado largo entre lineas."
        return "Medicion invalida."

    def _cruzo_hacia_abajo(self, linea_y: int, centro_y_actual: float) -> bool:
        if self.centro_y_anterior is None:
            return False
        if not (self.centro_y_anterior < linea_y <= centro_y_actual):
            return False
        salto = abs(centro_y_actual - self.centro_y_anterior)
        if salto > self.max_salto_y_por_frame:
            return False
        if centro_y_actual <= self.centro_y_anterior:
            return False
        return True

    def resumen(self, centro_x: float | None = None, centro_y: float | None = None) -> dict:
        formula = None
        if self.velocidad_kmh is not None and self.tiempo_entre_lineas:
            formula = (
                f"v = {self.distancia_metros:.2f} m / {self.tiempo_entre_lineas:.4f} s × 3.6 "
                f"= {self.velocidad_kmh:.2f} km/h"
            )
        return {
            "estado": self.estado,
            "metodo_medicion": self.metodo_medicion,
            "frame_cruce_linea_1": self.frame_cruce_linea_1,
            "frame_cruce_linea_2": self.frame_cruce_linea_2,
            "frame_cruce_linea_1_exacto": self.frame_cruce_linea_1_exacto,
            "frame_cruce_linea_2_exacto": self.frame_cruce_linea_2_exacto,
            "tiempo_cruce_linea_1": self.tiempo_cruce_linea_1,
            "tiempo_cruce_linea_2": self.tiempo_cruce_linea_2,
            "tiempo_entre_lineas": self.tiempo_entre_lineas,
            "distancia_metros": self.distancia_metros,
            "fps": self.fps,
            "fuente_tiempo": self.modo_tiempo or "pendiente",
            "velocidad_kmh": self.velocidad_kmh,
            "motivo_invalido": self.motivo_invalido,
            "formula_medicion": formula,
            "centro_x": centro_x,
            "centro_y": centro_y,
        }


def estimar_velocidad(config: dict, tiempo_segundos: float | None = None) -> float:
    distancia = float(config["speed"]["default_distance_meters"])
    tiempo = float(tiempo_segundos or config["speed"]["default_time_seconds"])
    return calcular_velocidad_kmh(distancia, tiempo)

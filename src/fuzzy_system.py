"""Sistema de inferencia difusa Mamdani para velocidad y nivel de sancion.

Entrada:
    velocidad del vehiculo en km/h, evaluada respecto al limite configurado.

Salida difusa:
    nivel de sancion en una escala 0-100.

Consecuencias operativas:
    multa monetaria, suspension, nivel de infraccion y mensaje.
"""

from __future__ import annotations

import math

import numpy as np


CATEGORIAS = (
    "seguro",
    "permitido",
    "precaucion",
    "exceso_leve",
    "exceso_medio",
    "exceso_alto",
)
SALIDAS_DIFUSAS = (
    "sin_multa",
    "advertencia",
    "multa_leve",
    "multa_moderada",
    "multa_grave",
)
SEVERIDAD = {categoria: indice for indice, categoria in enumerate(CATEGORIAS)}
SEVERIDAD_SALIDA = {categoria: indice for indice, categoria in enumerate(SALIDAS_DIFUSAS)}
SALIDA_A_CATEGORIA_OPERATIVA = {
    "sin_multa": "permitido",
    "advertencia": "precaucion",
    "multa_leve": "exceso_leve",
    "multa_moderada": "exceso_medio",
    "multa_grave": "exceso_alto",
}

REGLAS_DIFUSAS = (
    ("seguro", "sin_multa", "Si la velocidad es segura, entonces no existe multa."),
    ("permitido", "sin_multa", "Si la velocidad esta permitida, entonces no existe multa."),
    ("precaucion", "advertencia", "Si la velocidad requiere precaucion, entonces corresponde advertencia."),
    ("exceso_leve", "multa_leve", "Si existe exceso leve, entonces corresponde multa leve."),
    ("exceso_medio", "multa_moderada", "Si existe exceso medio, entonces corresponde multa moderada."),
    ("exceso_alto", "multa_grave", "Si existe exceso alto, entonces corresponde multa grave."),
)

ETIQUETAS_ENTRADA_ACADEMICA = {
    "seguro": "Muy baja / segura",
    "permitido": "Normal / permitida",
    "precaucion": "Precaucion",
    "exceso_leve": "Alta / exceso leve",
    "exceso_medio": "Muy alta / exceso medio",
    "exceso_alto": "Critica / exceso alto",
}
ETIQUETAS_SALIDA = {
    "sin_multa": "Sin multa",
    "advertencia": "Advertencia",
    "multa_leve": "Multa leve",
    "multa_moderada": "Multa moderada",
    "multa_grave": "Multa grave",
}

DEFAULT_FUZZY = {
    "seguro_hasta": 20.0,
    "permitido_hasta": 30.0,
    "precaucion_hasta": 35.0,
    "exceso_leve_hasta": 40.0,
    "exceso_medio_hasta": 50.0,
    "solapamiento_kmh": 2.0,
    "multa_usd": {
        "seguro": 0.0,
        "permitido": 0.0,
        "precaucion": 0.0,
        "exceso_leve": 15.0,
        "exceso_medio": 35.0,
        "exceso_alto": 60.0,
    },
    "suspension_horas": {
        "seguro": 0,
        "permitido": 0,
        "precaucion": 0,
        "exceso_leve": 2,
        "exceso_medio": 6,
        "exceso_alto": 12,
    },
    "etiquetas": {
        "seguro": "Seguro",
        "permitido": "Permitido",
        "precaucion": "Precaucion",
        "exceso_leve": "Exceso leve",
        "exceso_medio": "Exceso medio",
        "exceso_alto": "Exceso alto",
    },
    "nivel_infraccion": {
        "seguro": "Sin infraccion",
        "permitido": "Sin infraccion",
        "precaucion": "Advertencia",
        "exceso_leve": "Leve",
        "exceso_medio": "Media",
        "exceso_alto": "Grave",
    },
    "sancion": {
        "seguro": "No aplica",
        "permitido": "No aplica",
        "precaucion": "Advertencia preventiva",
        "exceso_leve": "Multa leve por exceso de velocidad",
        "exceso_medio": "Multa media y suspension temporal",
        "exceso_alto": "Multa grave y suspension extendida",
    },
    "mensajes": {
        "seguro": "Velocidad muy por debajo del limite; no corresponde sancion.",
        "permitido": "Velocidad dentro del limite permitido.",
        "precaucion": "Velocidad ligeramente sobre el limite; corresponde advertencia.",
        "exceso_leve": "Exceso leve de velocidad.",
        "exceso_medio": "Exceso medio de velocidad; corresponde multa y suspension temporal.",
        "exceso_alto": "Exceso alto de velocidad; corresponde multa grave y suspension extendida.",
    },
}


def triangular(x: float, a: float, b: float, c: float) -> float:
    """Funcion de membresia triangular, incluidos hombros degenerados."""
    valor = float(x)
    if a == b and valor <= b:
        return 1.0
    if b == c and valor >= b:
        return 1.0
    if valor <= a or valor >= c:
        return 0.0
    if valor == b:
        return 1.0
    if valor < b:
        return (valor - a) / max(b - a, 1e-12)
    return (c - valor) / max(c - b, 1e-12)


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    """Funcion trapezoidal robusta para hombro izquierdo o derecho."""
    valor = float(x)
    if a == b and valor <= b:
        izquierda = 1.0
    elif valor <= a:
        izquierda = 0.0
    elif valor < b:
        izquierda = (valor - a) / max(b - a, 1e-12)
    else:
        izquierda = 1.0

    if c == d and valor >= c:
        derecha = 1.0
    elif valor >= d:
        derecha = 0.0
    elif valor > c:
        derecha = (d - valor) / max(d - c, 1e-12)
    else:
        derecha = 1.0
    return max(0.0, min(1.0, izquierda, derecha))


def _obtener_limite(limite_kmh=30.0, config: dict | None = None) -> float:
    if isinstance(limite_kmh, dict):
        config = limite_kmh
        limite_kmh = None
    if limite_kmh is None and config:
        limite_kmh = (config.get("speed") or {}).get("campus_speed_limit_kmh", 30.0)
    limite = 30.0 if limite_kmh is None else float(limite_kmh)
    if not math.isfinite(limite) or limite <= 0:
        raise ValueError("El limite de velocidad debe ser un numero mayor que cero.")
    return limite


def _config_fuzzy(config: dict | None, limite_kmh: float | None = None) -> dict:
    base = {
        "seguro_hasta": DEFAULT_FUZZY["seguro_hasta"],
        "permitido_hasta": DEFAULT_FUZZY["permitido_hasta"],
        "precaucion_hasta": DEFAULT_FUZZY["precaucion_hasta"],
        "exceso_leve_hasta": DEFAULT_FUZZY["exceso_leve_hasta"],
        "exceso_medio_hasta": DEFAULT_FUZZY["exceso_medio_hasta"],
        "solapamiento_kmh": DEFAULT_FUZZY["solapamiento_kmh"],
        "multa_usd": dict(DEFAULT_FUZZY["multa_usd"]),
        "suspension_horas": dict(DEFAULT_FUZZY["suspension_horas"]),
        "etiquetas": dict(DEFAULT_FUZZY["etiquetas"]),
        "nivel_infraccion": dict(DEFAULT_FUZZY["nivel_infraccion"]),
        "sancion": dict(DEFAULT_FUZZY["sancion"]),
        "mensajes": dict(DEFAULT_FUZZY["mensajes"]),
    }
    fuzzy = (config or {}).get("fuzzy") or {}
    for clave in (
        "seguro_hasta",
        "permitido_hasta",
        "precaucion_hasta",
        "exceso_leve_hasta",
        "exceso_medio_hasta",
        "solapamiento_kmh",
    ):
        if fuzzy.get(clave) is not None:
            base[clave] = float(fuzzy[clave])
    for mapa in ("multa_usd", "suspension_horas", "etiquetas", "nivel_infraccion", "sancion", "mensajes"):
        if fuzzy.get(mapa):
            base[mapa].update(fuzzy[mapa])

    limite = _obtener_limite(limite_kmh, config)
    desplazamiento = limite - float(base["permitido_hasta"])
    for clave in ("seguro_hasta", "permitido_hasta", "precaucion_hasta", "exceso_leve_hasta", "exceso_medio_hasta"):
        base[clave] = max(0.0, float(base[clave]) + desplazamiento)
    base["permitido_hasta"] = limite
    base["solapamiento_kmh"] = max(float(base["solapamiento_kmh"]), 0.1)

    umbrales = [
        base["seguro_hasta"],
        base["permitido_hasta"],
        base["precaucion_hasta"],
        base["exceso_leve_hasta"],
        base["exceso_medio_hasta"],
    ]
    if any(actual >= siguiente for actual, siguiente in zip(umbrales, umbrales[1:])):
        raise ValueError("Los umbrales fuzzy deben estar ordenados de menor a mayor.")
    return base


def _calcular_grados(velocidad_kmh: float, cfg: dict) -> dict[str, float]:
    v = float(velocidad_kmh)
    o = float(cfg["solapamiento_kmh"])
    s = float(cfg["seguro_hasta"])
    p = float(cfg["permitido_hasta"])
    pr = float(cfg["precaucion_hasta"])
    el = float(cfg["exceso_leve_hasta"])
    em = float(cfg["exceso_medio_hasta"])
    return {
        "seguro": round(trapezoidal(v, 0.0, 0.0, max(s - o, 1.0), s + o * 0.5), 6),
        "permitido": round(triangular(v, max(0.0, s - o), p, p + o), 6),
        "precaucion": round(triangular(v, p - o, pr, pr + o), 6),
        "exceso_leve": round(trapezoidal(v, pr - o, el - o * 0.5, el, el + o), 6),
        "exceso_medio": round(trapezoidal(v, el - o, em - o * 0.5, em, em + o), 6),
        "exceso_alto": round(trapezoidal(v, em - o, em + o * 0.5, em + 25.0, em + 25.0), 6),
    }


def _categoria_crisp(velocidad_kmh: float, cfg: dict) -> str:
    velocidad = float(velocidad_kmh)
    for categoria, clave in (
        ("seguro", "seguro_hasta"),
        ("permitido", "permitido_hasta"),
        ("precaucion", "precaucion_hasta"),
        ("exceso_leve", "exceso_leve_hasta"),
        ("exceso_medio", "exceso_medio_hasta"),
    ):
        if velocidad <= float(cfg[clave]):
            return categoria
    return "exceso_alto"


def _defuzzificar_ponderado(activaciones: dict[str, float], valores: dict[str, float]) -> float:
    """Consecuencia operativa suave para multa USD y horas de suspension."""
    peso_total = sum(float(activaciones.get(categoria, 0.0)) for categoria in CATEGORIAS)
    if peso_total <= 1e-9:
        return 0.0
    acumulado = sum(
        float(activaciones.get(categoria, 0.0)) * float(valores.get(categoria, 0.0))
        for categoria in CATEGORIAS
    )
    return acumulado / peso_total


def _categoria_dominante(activaciones: dict[str, float], categorias=CATEGORIAS, severidad=SEVERIDAD) -> str:
    return max(categorias, key=lambda categoria: (float(activaciones.get(categoria, 0.0)), severidad[categoria]))


def _curvas_salida(universo: np.ndarray) -> dict[str, np.ndarray]:
    return {
        # Conjunto cerrado 0-10. No se usa un trapecio degenerado porque
        # c == d representa un hombro derecho en la funcion generica.
        "sin_multa": np.where(universo <= 10.0, 1.0, 0.0),
        "advertencia": np.array([triangular(x, 10.0, 25.0, 40.0) for x in universo]),
        "multa_leve": np.array([triangular(x, 30.0, 45.0, 60.0) for x in universo]),
        "multa_moderada": np.array([triangular(x, 50.0, 67.5, 82.0) for x in universo]),
        "multa_grave": np.array([trapezoidal(x, 72.0, 88.0, 100.0, 100.0) for x in universo]),
    }


def _evaluar_mamdani(grados: dict[str, float]) -> dict:
    activaciones_salida = {salida: 0.0 for salida in SALIDAS_DIFUSAS}
    reglas = []
    for antecedente, consecuente, descripcion in REGLAS_DIFUSAS:
        activacion = float(grados.get(antecedente, 0.0))
        activaciones_salida[consecuente] = max(activaciones_salida[consecuente], activacion)
        reglas.append(
            {
                "antecedente": antecedente,
                "consecuente": consecuente,
                "activacion": round(activacion, 6),
                "descripcion": descripcion,
            }
        )

    universo = np.linspace(0.0, 100.0, 1001)
    curvas = _curvas_salida(universo)
    recortadas = {
        salida: np.minimum(curvas[salida], float(activaciones_salida[salida]))
        for salida in SALIDAS_DIFUSAS
    }
    agregada = np.maximum.reduce(list(recortadas.values()))
    area = float(np.trapezoid(agregada, universo))
    centroide = float(np.trapezoid(universo * agregada, universo) / area) if area > 1e-12 else 0.0
    categoria_salida = _categoria_dominante(activaciones_salida, SALIDAS_DIFUSAS, SEVERIDAD_SALIDA)
    return {
        "activaciones_salida": activaciones_salida,
        "reglas": reglas,
        "universo": universo,
        "curvas_salida": curvas,
        "curvas_recortadas": recortadas,
        "agregada": agregada,
        "area_agregada": area,
        "centroide": centroide,
        "categoria_salida": categoria_salida,
    }


def obtener_reglas_difusas() -> list[dict]:
    return [
        {
            "numero": indice,
            "si_velocidad_es": ETIQUETAS_ENTRADA_ACADEMICA[antecedente],
            "entonces_sancion_es": ETIQUETAS_SALIDA[consecuente],
            "antecedente": antecedente,
            "consecuente": consecuente,
            "descripcion": descripcion,
        }
        for indice, (antecedente, consecuente, descripcion) in enumerate(REGLAS_DIFUSAS, start=1)
    ]


def _ejecutar_inferencia(
    velocidad_kmh: float,
    limite_kmh: float | dict | None,
    config: dict | None,
) -> dict:
    if isinstance(limite_kmh, dict) and config is None:
        config = limite_kmh
        limite_kmh = None
    limite = _obtener_limite(limite_kmh, config)
    cfg = _config_fuzzy(config, limite)
    velocidad = max(0.0, float(velocidad_kmh))
    grados = _calcular_grados(velocidad, cfg)
    if sum(grados.values()) <= 1e-9:
        categoria = _categoria_crisp(velocidad, cfg)
        grados = {cat: (1.0 if cat == categoria else 0.0) for cat in CATEGORIAS}
    return {
        "velocidad_kmh": velocidad,
        "limite_kmh": limite,
        "configuracion": cfg,
        "grados_entrada": grados,
        **_evaluar_mamdani(grados),
    }


def obtener_datos_visualizacion(
    velocidad_kmh: float,
    limite_kmh: float | dict | None = 30.0,
    config: dict | None = None,
) -> dict:
    """Datos completos para graficas y explicacion tecnica de la inferencia."""
    diagnostico = _ejecutar_inferencia(velocidad_kmh, limite_kmh, config)
    cfg = diagnostico["configuracion"]
    velocidad = diagnostico["velocidad_kmh"]
    max_entrada = max(100.0, cfg["exceso_medio_hasta"] + 30.0, velocidad + 10.0)
    universo_entrada = np.linspace(0.0, max_entrada, int(max_entrada * 10) + 1)
    curvas_entrada = {
        categoria: np.array([_calcular_grados(float(valor), cfg)[categoria] for valor in universo_entrada])
        for categoria in CATEGORIAS
    }
    return {
        **diagnostico,
        "universo_entrada": universo_entrada,
        "curvas_entrada": curvas_entrada,
    }


def clasificar_velocidad(
    velocidad_kmh: float,
    limite_kmh: float | dict | None = 30.0,
    config: dict | None = None,
) -> dict:
    """Ejecuta fuzzificacion, reglas, agregacion y defuzzificacion Mamdani."""
    diagnostico = _ejecutar_inferencia(velocidad_kmh, limite_kmh, config)
    velocidad = diagnostico["velocidad_kmh"]
    limite = diagnostico["limite_kmh"]
    cfg = diagnostico["configuracion"]
    grados = diagnostico["grados_entrada"]
    categoria = _categoria_dominante(grados)
    categoria_salida = diagnostico["categoria_salida"]
    categoria_operativa = SALIDA_A_CATEGORIA_OPERATIVA[categoria_salida]

    multa_usd = round(_defuzzificar_ponderado(grados, cfg["multa_usd"]), 2)
    horas = int(round(_defuzzificar_ponderado(grados, cfg["suspension_horas"])))
    if categoria_salida in {"sin_multa", "advertencia"}:
        multa_usd = 0.0
        horas = 0
    estado = cfg["etiquetas"].get(categoria, categoria.replace("_", " ").title())
    nivel = cfg["nivel_infraccion"].get(categoria_operativa, "Sin infraccion")
    sancion = cfg["sancion"].get(categoria_operativa, "No aplica")
    mensaje = cfg["mensajes"].get(categoria, "")
    sancion_aplica = categoria_salida in {"multa_leve", "multa_moderada", "multa_grave"}

    if multa_usd > 0 and horas > 0:
        multa_texto = f"Multa ${multa_usd:.2f} USD + {horas} h suspension"
    elif multa_usd > 0:
        multa_texto = f"Multa ${multa_usd:.2f} USD"
    elif horas > 0:
        multa_texto = f"Suspension {horas} h (sin multa monetaria)"
    else:
        multa_texto = "Sin multa"

    reglas_activas = [
        regla for regla in diagnostico["reglas"] if float(regla["activacion"]) > 0.0
    ]
    explicacion = [
        f"Fuzzificacion: {len(reglas_activas)} regla(s) activa(s) para {velocidad:.2f} km/h.",
        "Evaluacion: cada consecuente se recorta con el grado de activacion de su regla.",
        "Agregacion: se aplica el maximo entre todos los consecuentes recortados.",
        f"Defuzzificacion por centroide: nivel de sancion = {diagnostico['centroide']:.2f}/100.",
    ]
    return {
        "velocidad_kmh": velocidad,
        "limite_kmh": limite,
        "categoria_fuzzy": categoria,
        "categoria_operativa": categoria_operativa,
        "categoria_salida": categoria_salida,
        "etiqueta_salida": ETIQUETAS_SALIDA[categoria_salida],
        "estado": estado,
        "nivel_infraccion": nivel,
        "sancion": sancion,
        "multa_usd": multa_usd,
        "multa_texto": multa_texto,
        "horas_suspension": horas,
        "sancion_aplica": bool(sancion_aplica),
        "mensaje": mensaje,
        "grados": grados,
        "grados_entrada": grados,
        "activaciones_salida": {
            clave: round(float(valor), 6)
            for clave, valor in diagnostico["activaciones_salida"].items()
        },
        "reglas_activas": reglas_activas,
        "nivel_sancion_defuzzificado": round(float(diagnostico["centroide"]), 4),
        "metodo": "mamdani_centroide",
        "operadores": {
            "implicacion": "minimo",
            "agregacion": "maximo",
            "defuzzificacion": "centroide",
        },
        "explicacion_inferencia": explicacion,
        "peso_total_activacion": round(sum(grados.values()), 6),
    }

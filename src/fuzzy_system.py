"""Logica difusa para multa/sancion a partir de la velocidad medida (km/h).

Entrada: velocidad_kmh (+ limite de referencia y config).
Salida: multa_usd, horas_suspension, estado, sancion, grados de pertenencia.

Metodo: inferencia Mamdani simplificada — activacion por conjunto borroso y
defuzzificacion ponderada de multa y horas de suspension.
"""

from __future__ import annotations

CATEGORIAS = (
    "seguro",
    "permitido",
    "precaucion",
    "exceso_leve",
    "exceso_medio",
    "exceso_alto",
)

SEVERIDAD = {cat: idx for idx, cat in enumerate(CATEGORIAS)}

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
        "permitido": "Velocidad dentro del limite permitido del campus.",
        "precaucion": "Velocidad ligeramente sobre el limite; corresponde advertencia.",
        "exceso_leve": "Exceso leve de velocidad en zona universitaria.",
        "exceso_medio": "Exceso medio de velocidad; corresponde multa y suspension temporal.",
        "exceso_alto": "Exceso alto de velocidad; corresponde multa grave y suspension extendida.",
    },
}


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


def _obtener_limite(limite_kmh=30.0, config: dict | None = None) -> float:
    if isinstance(limite_kmh, dict):
        config = limite_kmh
        limite_kmh = None
    if config:
        speed = config.get("speed") or {}
        if limite_kmh is None:
            return float(speed.get("campus_speed_limit_kmh", 30.0))
    if limite_kmh is None:
        return 30.0
    if isinstance(limite_kmh, dict):
        speed = limite_kmh.get("speed", {})
        return float(speed.get("campus_speed_limit_kmh", 30.0))
    return float(limite_kmh)


def _config_fuzzy(config: dict | None) -> dict:
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
    if not config:
        return base
    fuzzy = config.get("fuzzy") or {}
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
    if fuzzy.get("suspension_horas"):
        base["suspension_horas"].update(fuzzy["suspension_horas"])
    return base


def _calcular_grados(velocidad_kmh: float, cfg: dict) -> dict[str, float]:
    v = float(velocidad_kmh)
    o = float(cfg.get("solapamiento_kmh", 2.0))
    s = float(cfg["seguro_hasta"])
    p = float(cfg["permitido_hasta"])
    pr = float(cfg["precaucion_hasta"])
    el = float(cfg["exceso_leve_hasta"])
    em = float(cfg["exceso_medio_hasta"])
    tope_alto = em + max(20.0, (em - el) * 2.0)

    return {
        "seguro": round(trapezoidal(v, 0.0, 0.0, max(s - o, 1.0), s + o * 0.5), 4),
        "permitido": round(triangular(v, max(0.0, s - o), p, p + o), 4),
        "precaucion": round(triangular(v, p - o, pr, pr + o), 4),
        "exceso_leve": round(trapezoidal(v, pr - o, el - o * 0.5, el, el + o), 4),
        "exceso_medio": round(trapezoidal(v, el - o, em - o * 0.5, em, em + o), 4),
        "exceso_alto": round(trapezoidal(v, em - o, em + o * 0.5, tope_alto, tope_alto + 20.0), 4),
    }


def _categoria_crisp(velocidad_kmh: float, cfg: dict) -> str:
    v = float(velocidad_kmh)
    if v <= float(cfg["seguro_hasta"]):
        return "seguro"
    if v <= float(cfg["permitido_hasta"]):
        return "permitido"
    if v <= float(cfg["precaucion_hasta"]):
        return "precaucion"
    if v <= float(cfg["exceso_leve_hasta"]):
        return "exceso_leve"
    if v <= float(cfg["exceso_medio_hasta"]):
        return "exceso_medio"
    return "exceso_alto"


def _defuzzificar_ponderado(activaciones: dict[str, float], valores: dict[str, float]) -> float:
    peso_total = sum(float(activaciones.get(cat, 0.0)) for cat in CATEGORIAS)
    if peso_total <= 1e-9:
        return 0.0
    acumulado = sum(float(activaciones.get(cat, 0.0)) * float(valores.get(cat, 0.0)) for cat in CATEGORIAS)
    return acumulado / peso_total


def _categoria_dominante(activaciones: dict[str, float]) -> str:
    return max(
        CATEGORIAS,
        key=lambda cat: (float(activaciones.get(cat, 0.0)), SEVERIDAD[cat]),
    )


def clasificar_velocidad(
    velocidad_kmh: float,
    limite_kmh: float | dict | None = 30.0,
    config: dict | None = None,
) -> dict:
    """Evalua multa/sancion difusa. `config` puede pasarse como 2do arg (compat pipeline)."""
    if isinstance(limite_kmh, dict) and config is None:
        config = limite_kmh
        limite_kmh = None

    cfg = _config_fuzzy(config)
    limite = _obtener_limite(limite_kmh, config)
    velocidad = max(0.0, float(velocidad_kmh))

    grados = _calcular_grados(velocidad, cfg)
    peso_total = sum(grados.values())
    if peso_total <= 1e-9:
        categoria = _categoria_crisp(velocidad, cfg)
        grados = {cat: (1.0 if cat == categoria else 0.0) for cat in CATEGORIAS}
        peso_total = 1.0

    categoria = _categoria_dominante(grados)
    multa_usd = round(_defuzzificar_ponderado(grados, cfg["multa_usd"]), 2)
    horas_float = _defuzzificar_ponderado(grados, cfg["suspension_horas"])
    horas = int(round(horas_float))

    estado = cfg["etiquetas"].get(categoria, categoria.replace("_", " ").title())
    nivel = cfg["nivel_infraccion"].get(categoria, "Sin infraccion")
    sancion = cfg["sancion"].get(categoria, "No aplica")
    mensaje = cfg["mensajes"].get(categoria, "")
    sancion_aplica = multa_usd > 0.0 or horas > 0 or nivel not in {"Sin infraccion", "Advertencia"}

    if multa_usd > 0 and horas > 0:
        multa_texto = f"Multa ${multa_usd:.2f} USD + {horas} h suspension"
    elif multa_usd > 0:
        multa_texto = f"Multa ${multa_usd:.2f} USD"
    elif horas > 0:
        multa_texto = f"Suspension {horas} h (sin multa monetaria)"
    else:
        multa_texto = "Sin multa"

    return {
        "velocidad_kmh": velocidad,
        "limite_kmh": limite,
        "categoria_fuzzy": categoria,
        "estado": estado,
        "nivel_infraccion": nivel,
        "sancion": sancion,
        "multa_usd": multa_usd,
        "multa_texto": multa_texto,
        "horas_suspension": horas,
        "sancion_aplica": bool(sancion_aplica),
        "mensaje": mensaje,
        "grados": grados,
        "metodo": "mamdani_ponderado",
        "peso_total_activacion": round(peso_total, 4),
    }

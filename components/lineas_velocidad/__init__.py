import os

import streamlit.components.v1 as components

_PARENT = os.path.dirname(os.path.abspath(__file__))
_FRONTEND = os.path.join(_PARENT, "frontend")
_lineas_component = components.declare_component("lineas_velocidad", path=_FRONTEND)


def editor_lineas_velocidad(
    image_base64: str,
    line1: float = 0.45,
    line2: float = 0.65,
    height: int = 420,
    key: str | None = None,
) -> dict | None:
    """Editor visual: dos lineas horizontales arrastrables sobre la imagen."""
    if not image_base64:
        return None
    result = _lineas_component(
        imageBase64=image_base64,
        line1=float(line1),
        line2=float(line2),
        key=key,
        default={"line1": float(line1), "line2": float(line2)},
        height=int(height),
    )
    if isinstance(result, dict):
        return result
    return None

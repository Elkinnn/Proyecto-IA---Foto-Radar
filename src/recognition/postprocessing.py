"""Postprocesamiento y consolidacion del lector CNN de caracteres.

Este modulo contiene funciones puras de formato ecuatoriano, correcciones
por posicion y votacion de lecturas por evento. Se separa del pipeline visual
para que la segmentacion y la inferencia queden mas faciles de explicar.
"""

from datetime import datetime
from pathlib import Path
import json
import re

from src.utils import normalizar_placa


RECONOCIMIENTO_CARACTERES_DIR = Path("reports") / "evidencias" / "reconocimiento_caracteres"
OCR_DIR = RECONOCIMIENTO_CARACTERES_DIR
VOTACION_EVENTOS_DIR = OCR_DIR / "votacion_eventos"


DIGITOS_PLACA = set("0123456789")

LETRAS_PLACA = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _tipo_posicion_formato(idx: int) -> str:
    """Formato Ecuador: 3 letras + 3/4 digitos. Posiciones 0-2 = letra, resto = numero."""
    return "letra" if idx < 3 else "numero"

def _mejor_caracter_por_tipo(probs_por_clase: dict, tipo: str) -> tuple[str, float]:
    """Devuelve (caracter, confianza) restringiendo el argmax al universo del tipo."""
    universo = LETRAS_PLACA if tipo == "letra" else DIGITOS_PLACA
    candidatos = [(c, p) for c, p in (probs_por_clase or {}).items() if c in universo]
    if not candidatos:
        return "", 0.0
    candidatos.sort(key=lambda kv: kv[1], reverse=True)
    return candidatos[0][0], float(candidatos[0][1])

def _top3_por_tipo(probs_por_clase: dict, tipo: str) -> list:
    universo = LETRAS_PLACA if tipo == "letra" else DIGITOS_PLACA
    candidatos = sorted(
        ((c, p) for c, p in (probs_por_clase or {}).items() if c in universo),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    return [{"caracter": c, "confianza": float(p)} for c, p in candidatos]

def _confianzas_enmascaradas_ventana(ventana: list) -> tuple[str, list]:
    """Calcula texto y confianzas aplicando la mascara de tipo por posicion, SIN mutar."""
    chars, confs = [], []
    for i, pred in enumerate(ventana):
        tipo = _tipo_posicion_formato(i)
        ch, conf = _mejor_caracter_por_tipo(pred.get("probs_por_clase"), tipo)
        chars.append(ch)
        confs.append(conf)
    return "".join(chars), confs

def _puntuar_ventana_lectura(predicciones: list) -> float:
    """Puntaje de una ventana usando la prediccion enmascarada por formato."""
    if not predicciones:
        return -999.0
    texto, confs = _confianzas_enmascaradas_ventana(predicciones)
    puntaje = sum(confs)
    if confs:
        puntaje += min(confs) * 2.0
    if validar_formato_placa_ecuador(texto).get("valido"):
        puntaje += 40.0
    if len(predicciones) in (6, 7):
        puntaje += 10.0
    return puntaje

def _aplicar_mascara_formato(ventana: list) -> None:
    """Reemplaza caracter_predicho/confianza/top3 por la version restringida al tipo
    de cada posicion (letra en 0-2, numero en 3+). Conserva la prediccion cruda."""
    for i, pred in enumerate(ventana):
        tipo = _tipo_posicion_formato(i)
        ch, conf = _mejor_caracter_por_tipo(pred.get("probs_por_clase"), tipo)
        if "caracter_predicho_crudo" not in pred:
            pred["caracter_predicho_crudo"] = pred.get("caracter_predicho", "")
            pred["confianza_cruda"] = float(pred.get("confianza", 0.0))
        pred["caracter_predicho"] = ch
        pred["confianza"] = conf
        pred["tipo_posicion"] = tipo
        top3 = _top3_por_tipo(pred.get("probs_por_clase"), tipo)
        if top3:
            pred["top3_predicciones"] = top3

def postprocesar_texto_placa(texto: str) -> dict:
    crudo = normalizar_placa((texto or "").replace("-", ""))
    letras = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}
    numeros = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8"}
    corregido = []
    for idx, caracter in enumerate(crudo):
        if idx < 3:
            corregido.append(letras.get(caracter, caracter))
        else:
            corregido.append(numeros.get(caracter, caracter))
    return {
        "texto_detectado_crudo": crudo,
        "texto_postprocesado": "".join(corregido),
    }

def postprocesar_por_formato_ecuador(texto_crudo: str, predicciones_caracteres: list) -> dict:
    crudo = normalizar_placa((texto_crudo or "").replace("-", ""))
    letras_directas = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B", "7": "T", "6": "G"}
    numeros_directos = {"O": "0", "I": "1", "S": "5", "Z": "2", "B": "8", "G": "6", "T": "7"}
    caracteres = []
    cambios = []

    for idx, caracter in enumerate(crudo):
        pred = predicciones_caracteres[idx] if idx < len(predicciones_caracteres) else {}
        confianza = float(pred.get("confianza", 0.0))
        top3 = pred.get("top3_predicciones", []) or []
        top3_map = {str(item.get("caracter")): float(item.get("confianza", 0.0)) for item in top3}
        esperado = "letra" if idx < 3 else "numero"
        nuevo = caracter
        motivo = ""

        if esperado == "letra":
            if caracter.isalpha():
                nuevo = caracter
            else:
                candidato = letras_directas.get(caracter)
                if candidato and (candidato in top3_map or confianza < 0.65):
                    nuevo = candidato
                    motivo = f"{caracter}->{candidato} por posicion de letra"
                elif caracter == "6" and "G" in top3_map and confianza < 0.75:
                    nuevo = "G"
                    motivo = "6->G respaldado por top3"
                else:
                    mejor_letra = _mejor_top3_por_tipo(top3, tipo="letra", confianza_actual=confianza)
                    if mejor_letra:
                        nuevo = mejor_letra
                        motivo = f"{caracter}->{mejor_letra} por top3 en posicion de letra"
        else:
            if caracter.isdigit():
                nuevo = caracter
            else:
                candidato = numeros_directos.get(caracter)
                if candidato and (candidato in top3_map or confianza < 0.65):
                    nuevo = candidato
                    motivo = f"{caracter}->{candidato} por posicion numerica"
                else:
                    mejor_numero = _mejor_top3_por_tipo(top3, tipo="numero", confianza_actual=confianza)
                    if mejor_numero:
                        nuevo = mejor_numero
                        motivo = f"{caracter}->{mejor_numero} por top3 en posicion numerica"

        caracteres.append(nuevo)
        if nuevo != caracter:
            cambios.append(
                {
                    "indice": idx + 1,
                    "original": caracter,
                    "corregido": nuevo,
                    "confianza_original": confianza,
                    "motivo": motivo,
                }
            )

    texto = "".join(caracteres)
    formato = validar_formato_placa_ecuador(texto)
    return {
        "texto_detectado_crudo": crudo,
        "texto_postprocesado": texto,
        "formato_valido": formato["valido"],
        "cambios": cambios,
    }

def _tipo_posicion_placa(indice: int) -> str:
    return "letra" if indice < 3 else "numero"

def _es_tipo_caracter(caracter: str, tipo: str) -> bool:
    return bool(caracter) and ((tipo == "letra" and caracter.isalpha()) or (tipo == "numero" and caracter.isdigit()))

def _mapa_confusion_para_tipo(caracter: str, tipo: str) -> str:
    letras = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "T", "8": "B"}
    numeros = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"}
    return (letras if tipo == "letra" else numeros).get(str(caracter), "")

def _normalizar_topk_prediccion(pred: dict, caracter_base: str = "") -> list[dict]:
    top = pred.get("top3_predicciones") or pred.get("topk") or []
    mejores = {}
    for item in top:
        caracter = normalizar_placa(str(item.get("caracter", "")))[:1]
        if caracter:
            confianza = float(item.get("confianza", 0.0) or 0.0)
            mejores[caracter] = max(mejores.get(caracter, 0.0), confianza)
    if not mejores and caracter_base:
        mejores[caracter_base] = float(pred.get("confianza", 0.0) or 0.0)
    return [{"caracter": ch, "confianza": conf} for ch, conf in sorted(mejores.items(), key=lambda item: item[1], reverse=True)]

def _opciones_por_posicion(pred: dict, caracter_base: str, indice: int) -> tuple[list[dict], list[dict]]:
    tipo = _tipo_posicion_placa(indice)
    top = _normalizar_topk_prediccion(pred, caracter_base)
    correcciones = []
    opciones = []
    for item in top:
        caracter = item["caracter"]
        confianza = float(item.get("confianza", 0.0))
        if _es_tipo_caracter(caracter, tipo):
            opciones.append({"caracter": caracter, "confianza": confianza, "origen": "topk_tipo_correcto"})
    if opciones:
        return opciones, correcciones
    respaldo = _mapa_confusion_para_tipo(caracter_base, tipo)
    if respaldo:
        correcciones.append(
            {
                "posicion": indice + 1,
                "original": caracter_base,
                "corregido": respaldo,
                "motivo": f"mapa_confusion_{tipo}_sin_alternativa_topk",
            }
        )
        return [{"caracter": respaldo, "confianza": float(pred.get("confianza", 0.0) or 0.0) * 0.72, "origen": "mapa_confusion"}], correcciones
    return [], correcciones

def _peso_lectura_evento(lectura: dict) -> float:
    evaluacion = evaluar_calidad_lectura_evento(lectura)
    if not evaluacion.get("participa_votacion"):
        return 0.0
    if evaluacion.get("puntaje_lectura", 0.0) <= 0.0:
        return 0.0
    return max(0.01, round(float(evaluacion["puntaje_lectura"]) / 100.0, 5))

def evaluar_calidad_lectura_evento(lectura: dict) -> dict:
    texto = normalizar_placa(
        lectura.get("texto_corregido_formato")
        or lectura.get("texto_postprocesado")
        or lectura.get("texto_corregido")
        or lectura.get("texto_crudo")
        or lectura.get("texto_detectado_crudo")
        or ""
    )
    confianza_cnn = float(
        lectura.get("confianza_cnn_caracteres")
        or lectura.get("confianza_promedio")
        or lectura.get("confianza_cnn")
        or lectura.get("confianza")
        or 0.0
    )
    puntaje_recorte = float(lectura.get("puntaje_recorte") or lectura.get("puntaje_total") or 0.0)
    puntaje_segmentacion = float(lectura.get("puntaje_segmentacion") or 0.0)
    puntaje_rectificacion = float(lectura.get("puntaje_rectificacion") or lectura.get("confianza_rectificacion") or 0.0)
    conf_yolo = float(lectura.get("confianza_yolo") or lectura.get("conf_yolo") or 0.0)
    cantidad = int(lectura.get("cantidad_caracteres_segmentados") or len(texto))
    formato = bool(lectura.get("formato_valido") or validar_formato_placa_ecuador(texto).get("valido"))
    estado = str(lectura.get("estado_lectura") or lectura.get("estado") or "").lower()
    predicciones = lectura.get("predicciones_caracteres") or []
    motivos_rechazo = lectura.get("motivos_rechazo") or {}
    correcciones = postprocesar_por_formato_ecuador(texto, predicciones).get("cambios", []) if texto else []
    imposibles = sum(1 for idx, ch in enumerate(texto) if ch and not _es_tipo_caracter(ch, _tipo_posicion_placa(idx)))
    guion_ruido = int(motivos_rechazo.get("posible_guion_o_ruido", 0) or 0) + int(motivos_rechazo.get("aspecto_muy_delgado", 0) or 0)
    unidos = int(motivos_rechazo.get("posible_caracter_unido", 0) or 0) + int(motivos_rechazo.get("aspecto_muy_ancho", 0) or 0)

    puntaje = 10.0
    puntaje += confianza_cnn * 26.0
    if confianza_cnn < 0.70:
        puntaje -= (0.70 - confianza_cnn) * 55.0
    elif confianza_cnn >= 0.82:
        puntaje += 8.0
    puntaje += conf_yolo * 8.0
    puntaje += min(max(puntaje_recorte, 0.0) / 100.0, 1.0) * 10.0
    puntaje += min(max(puntaje_segmentacion, 0.0) / 100.0, 1.0) * 18.0
    puntaje += min(max(puntaje_rectificacion, 0.0) / 100.0, 1.0) * 6.0
    if cantidad in (6, 7):
        puntaje += 18.0
    elif cantidad < 6:
        puntaje -= (6 - cantidad) * 14.0
    elif cantidad > 7:
        puntaje -= (cantidad - 7) * 12.0
    if formato:
        puntaje += 18.0
    else:
        puntaje -= min(18.0, imposibles * 4.0)
    if estado == "lectura_completa":
        puntaje += 10.0
    elif estado == "formato_dudoso":
        puntaje -= 3.0
    elif estado == "lectura_parcial":
        puntaje -= 18.0
    elif estado in {"segmentacion_incompleta", "sin_caracteres", "error_cnn"}:
        puntaje -= 45.0
    puntaje -= min(18.0, len(correcciones) * 3.5)
    puntaje -= min(16.0, guion_ruido * 4.0)
    puntaje -= min(18.0, unidos * 6.0)
    if len(texto) not in (6, 7):
        puntaje -= 16.0

    motivos = []
    if cantidad < 6:
        motivos.append("menos_de_6_caracteres")
    if cantidad > 7:
        motivos.append("mas_de_7_caracteres")
    if confianza_cnn < 0.35:
        motivos.append("confianza_cnn_baja")
    if estado in {"sin_caracteres", "error_cnn"}:
        motivos.append(estado)
    if guion_ruido:
        motivos.append("posible_guion_borde_o_ruido")
    if unidos:
        motivos.append("posible_bloque_de_caracteres_unidos")
    if imposibles >= 3:
        motivos.append("demasiados_caracteres_imposibles_por_posicion")
    participa = puntaje >= 35.0 and estado not in {"sin_caracteres", "error_cnn"} and cantidad >= 6 and cantidad <= 7
    if not participa and not motivos:
        motivos.append("puntaje_lectura_bajo")
    return {
        "puntaje_lectura": round(max(0.0, min(100.0, puntaje)), 4),
        "participa_votacion": bool(participa),
        "motivo_descarte": "; ".join(motivos) if not participa else "",
        "cantidad_caracteres": cantidad,
        "formato_valido": formato,
        "correcciones_formato": len(correcciones),
        "caracteres_imposibles": imposibles,
        "guion_ruido": guion_ruido,
        "caracteres_unidos": unidos,
    }

def consolidar_lecturas_evento_placa(lecturas_evento: list[dict], max_lecturas: int = 5) -> dict:
    lecturas = [lectura for lectura in (lecturas_evento or []) if lectura]
    if not lecturas:
        return {
            "texto_final": "",
            "texto_crudo_mas_confiable": "",
            "formato_detectado": "",
            "formato_valido": False,
            "confianza_final": 0.0,
            "votos_por_posicion": [],
            "correcciones_por_formato": [],
            "lectura_base_usada": {},
            "estado": "segmentacion_incompleta",
            "motivo": "sin_lecturas_evento",
        }

    for lectura in lecturas:
        calidad = evaluar_calidad_lectura_evento(lectura)
        lectura["puntaje_lectura"] = calidad["puntaje_lectura"]
        lectura["participa_votacion"] = calidad["participa_votacion"]
        lectura["motivo_descarte_votacion"] = calidad["motivo_descarte"]
        lectura["calidad_lectura_evento"] = calidad
        lectura["_peso_evento"] = _peso_lectura_evento(lectura)
        texto = normalizar_placa(
            lectura.get("texto_corregido_formato")
            or lectura.get("texto_postprocesado")
            or lectura.get("texto_corregido")
            or lectura.get("texto_crudo")
            or lectura.get("texto_detectado_crudo")
            or ""
        )
        lectura["_texto_evento"] = texto

    lecturas_validas = [lectura for lectura in lecturas if lectura.get("participa_votacion") and lectura.get("_peso_evento", 0.0) > 0.0]
    if not lecturas_validas:
        lecturas_validas = sorted(lecturas, key=lambda item: item.get("puntaje_lectura", 0.0), reverse=True)[:1]
    lecturas_ordenadas = sorted(lecturas_validas, key=lambda item: item.get("_peso_evento", 0.0), reverse=True)[: max(1, int(max_lecturas))]
    base = lecturas_ordenadas[0]
    longitudes = {}
    for lectura in lecturas_ordenadas:
        largo = len(lectura.get("_texto_evento", ""))
        if 6 <= largo <= 7:
            longitudes[largo] = longitudes.get(largo, 0.0) + lectura.get("_peso_evento", 0.0)
    if longitudes:
        longitud_objetivo = max(longitudes.items(), key=lambda item: (item[1], item[0]))[0]
    else:
        longitud_objetivo = min(max(len(base.get("_texto_evento", "")), 0), 7)

    votos_por_posicion = []
    correcciones = []
    texto_final = []
    confianzas_pos = []
    for indice in range(longitud_objetivo):
        tipo = _tipo_posicion_placa(indice)
        votos = {}
        detalles = []
        for lectura in lecturas_ordenadas:
            texto = lectura.get("_texto_evento", "")
            if indice >= len(texto):
                continue
            caracter_base = texto[indice]
            predicciones = lectura.get("predicciones_caracteres") or []
            pred = predicciones[indice] if indice < len(predicciones) else {"confianza": lectura.get("confianza_cnn_caracteres", 0.0)}
            opciones, corr = _opciones_por_posicion(pred, caracter_base, indice)
            correcciones.extend(corr)
            peso_lectura = float(lectura.get("_peso_evento", 0.0))
            if not opciones and _es_tipo_caracter(caracter_base, tipo):
                opciones = [{"caracter": caracter_base, "confianza": float(pred.get("confianza", 0.0) or 0.0), "origen": "texto_base"}]
            for opcion in opciones:
                peso = peso_lectura * max(float(opcion.get("confianza", 0.0)), 0.05)
                votos[opcion["caracter"]] = votos.get(opcion["caracter"], 0.0) + peso
                detalles.append(
                    {
                        "lectura": texto,
                        "caracter": opcion["caracter"],
                        "peso": round(peso, 5),
                        "origen": opcion.get("origen"),
                    }
                )
        if votos:
            elegido, peso_elegido = max(votos.items(), key=lambda item: item[1])
            total = sum(votos.values())
            confianza_pos = peso_elegido / total if total > 0 else 0.0
        else:
            elegido = ""
            confianza_pos = 0.0
        texto_final.append(elegido)
        confianzas_pos.append(confianza_pos)
        votos_por_posicion.append(
            {
                "posicion": indice + 1,
                "tipo_esperado": tipo,
                "elegido": elegido,
                "confianza_posicion": round(confianza_pos, 5),
                "votos": {k: round(v, 5) for k, v in sorted(votos.items(), key=lambda item: item[1], reverse=True)},
                "detalles": detalles,
            }
        )

    texto = "".join(texto_final)
    formato = validar_formato_placa_ecuador(texto)
    consistencia = sum(confianzas_pos) / len(confianzas_pos) if confianzas_pos else 0.0
    cantidad_validas = sum(1 for lectura in lecturas_ordenadas if len(lectura.get("_texto_evento", "")) in (6, 7))
    promedio_peso = sum(float(l.get("_peso_evento", 0.0)) for l in lecturas_ordenadas) / len(lecturas_ordenadas)
    penalizacion_correcciones = min(0.25, len(correcciones) * 0.025)
    confianza_final = max(
        0.0,
        min(1.0, consistencia * 0.45 + promedio_peso * 0.35 + min(cantidad_validas / max(len(lecturas_ordenadas), 1), 1.0) * 0.20 - penalizacion_correcciones),
    )

    if len(texto) < 6:
        estado = "lectura_parcial"
    elif not formato.get("valido"):
        estado = "formato_dudoso"
    elif confianza_final < 0.45:
        estado = "baja_confianza"
    else:
        estado = "lectura_completa"

    return {
        "texto_final": texto,
        "texto_crudo_mas_confiable": base.get("texto_crudo") or base.get("texto_detectado_crudo") or base.get("_texto_evento", ""),
        "formato_detectado": "LLLDDDD" if len(texto) == 7 else "LLLDDD" if len(texto) == 6 else "parcial",
        "formato_valido": bool(formato.get("valido")),
        "confianza_final": round(float(confianza_final), 5),
        "votos_por_posicion": votos_por_posicion,
        "correcciones_por_formato": correcciones,
        "lectura_base_usada": {k: v for k, v in base.items() if not k.startswith("_") and k not in {"predicciones_caracteres"}},
        "estado": estado,
        "motivo": formato.get("mensaje") if not formato.get("valido") else "lectura_consolidada_por_evento",
        "lecturas_usadas": [
            {
                "texto": lectura.get("_texto_evento"),
                "peso": lectura.get("_peso_evento"),
                "confianza_cnn": lectura.get("confianza_cnn_caracteres") or lectura.get("confianza_promedio"),
                "puntaje_recorte": lectura.get("puntaje_recorte") or lectura.get("puntaje_total"),
                "puntaje_segmentacion": lectura.get("puntaje_segmentacion"),
                "puntaje_lectura": lectura.get("puntaje_lectura"),
                "participa_votacion": lectura.get("participa_votacion"),
                "cantidad_caracteres_segmentados": lectura.get("cantidad_caracteres_segmentados"),
            }
            for lectura in lecturas_ordenadas
        ],
        "lecturas_descartadas": [
            {
                "texto": lectura.get("_texto_evento"),
                "puntaje_lectura": lectura.get("puntaje_lectura"),
                "motivo_descarte": lectura.get("motivo_descarte_votacion"),
                "estado_lectura": lectura.get("estado_lectura") or lectura.get("estado"),
            }
            for lectura in lecturas
            if lectura not in lecturas_ordenadas
        ],
        "cantidad_lecturas_usadas": len(lecturas_ordenadas),
        "cantidad_lecturas_descartadas": max(0, len(lecturas) - len(lecturas_ordenadas)),
    }

def guardar_debug_votacion_evento(event_id: int | str, lecturas_evento: list[dict], consolidado: dict) -> str:
    VOTACION_EVENTOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = VOTACION_EVENTOS_DIR / f"evento_{str(event_id).zfill(4)}_votacion.json"
    ruta.write_text(
        json.dumps(
            {
                "event_id": event_id,
                "lecturas_evento": lecturas_evento,
                "consolidado": consolidado,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return str(ruta)

def _mejor_top3_por_tipo(top3: list, tipo: str, confianza_actual: float) -> str:
    for item in top3:
        candidato = str(item.get("caracter", ""))
        confianza = float(item.get("confianza", 0.0))
        if tipo == "letra" and not candidato.isalpha():
            continue
        if tipo == "numero" and not candidato.isdigit():
            continue
        if confianza_actual < 0.80 or confianza >= confianza_actual - 0.20:
            return candidato
    return ""

def validar_formato_placa_ecuador(texto: str) -> dict:
    normalizada = normalizar_placa((texto or "").replace("-", ""))
    valido = bool(re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", normalizada))
    return {
        "texto_normalizado": normalizada if valido else normalizada,
        "valido": valido,
        "mensaje": "Formato ecuatoriano valido." if valido else "Formato no valido para placa ecuatoriana ABC123 o ABC1234.",
    }

def comparar_con_esperada(texto_normalizado: str, placa_esperada: str) -> dict:
    esperada = validar_formato_placa_ecuador(placa_esperada).get("texto_normalizado", "")
    detectada = validar_formato_placa_ecuador(texto_normalizado).get("texto_normalizado", "")
    if not esperada:
        return {
            "placa_esperada_normalizada": "",
            "coincide": False,
            "mensaje": "No se ingreso placa esperada para comparar.",
        }
    coincide = bool(detectada and detectada == esperada)
    return {
        "placa_esperada_normalizada": esperada,
        "coincide": coincide,
        "mensaje": "La placa coincide con la esperada." if coincide else "La placa detectada no coincide o aun no fue reconocida.",
    }


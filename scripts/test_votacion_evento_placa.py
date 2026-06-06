from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plate_reader import consolidar_lecturas_evento_placa, guardar_debug_votacion_evento


def _predicciones_desde_texto(texto: str, alternativas: dict[int, list[str]] | None = None) -> list[dict]:
    alternativas = alternativas or {}
    salida = []
    for idx, caracter in enumerate(texto):
        top = [{"caracter": caracter, "confianza": 0.72}]
        for alt_idx, alt in enumerate(alternativas.get(idx, []), start=1):
            top.append({"caracter": alt, "confianza": max(0.15, 0.65 - alt_idx * 0.12)})
        salida.append(
            {
                "indice": idx + 1,
                "caracter_predicho": caracter,
                "confianza": top[0]["confianza"],
                "top3_predicciones": top[:3],
            }
        )
    return salida


def _lectura(texto: str, confianza: float = 0.75, alternativas: dict[int, list[str]] | None = None) -> dict:
    return {
        "texto_crudo": texto,
        "texto_corregido_formato": texto,
        "confianza_cnn_caracteres": confianza,
        "confianza_yolo": 0.82,
        "puntaje_recorte": 0.70,
        "puntaje_segmentacion": 78.0,
        "cantidad_caracteres_segmentados": len(texto),
        "predicciones_caracteres": _predicciones_desde_texto(texto, alternativas),
    }


def main() -> None:
    casos = [
        {
            "nombre": "caso_1_pdk7282",
            "esperado": "PDK7282",
            "lecturas": [
                _lectura("BDK7282", 0.66, {0: ["P", "B"]}),
                _lectura("P0K7282", 0.68, {1: ["D", "O"]}),
                _lectura("PDK6282", 0.74, {3: ["7", "6"]}),
                _lectura("PDK7282", 0.88),
                _lectura("PDKA282", 0.63, {3: ["7", "4"]}),
            ],
        },
        {
            "nombre": "caso_2_abc0123",
            "esperado": "ABC0123",
            "lecturas": [
                _lectura("ABCO123", 0.67, {3: ["0", "O"]}),
                _lectura("ABC0123", 0.86),
                _lectura("A8C0123", 0.64, {1: ["B", "8"]}),
                _lectura("ABCO12B", 0.61, {3: ["0", "O"], 6: ["3", "8"]}),
            ],
        },
        {
            "nombre": "caso_3_pdz279",
            "esperado": "PDZ279",
            "lecturas": [
                _lectura("PDZ279", 0.87),
                _lectura("P0Z279", 0.67, {1: ["D", "O"]}),
                _lectura("PD2279", 0.68, {2: ["Z", "2"]}),
                _lectura("PDZ27S", 0.63, {5: ["9", "5"]}),
            ],
        },
        {
            "nombre": "caso_4_filtrar_lecturas_malas",
            "esperado": "ABC0123",
            "lecturas": [
                _lectura("AAG0123", 0.62, {1: ["B", "A"], 2: ["C", "G"]}),
                _lectura("ABG4126", 0.61, {2: ["C", "G"], 3: ["0", "4"], 6: ["3", "6"]}),
                {**_lectura("IL1472", 0.42), "estado_lectura": "lectura_parcial", "cantidad_caracteres_segmentados": 5},
                {**_lectura("UL4720", 0.40), "estado_lectura": "lectura_parcial", "cantidad_caracteres_segmentados": 5},
                {**_lectura("FABC032", 0.44), "estado_lectura": "formato_dudoso", "cantidad_caracteres_segmentados": 8},
                _lectura("ABC0123", 0.82),
            ],
        },
        {
            "nombre": "caso_5_no_conservar_lectura_antigua",
            "esperado": "TBH4543",
            "lecturas": [
                {
                    **_lectura("BTA8990", 0.62),
                    "puntaje_recorte": 0.60,
                    "puntaje_segmentacion": 62.0,
                },
                {
                    **_lectura("TBH4543", 0.96),
                    "puntaje_recorte": 1.0,
                    "puntaje_segmentacion": 100.0,
                    "estado_lectura": "lectura_completa",
                    "formato_valido": True,
                },
            ],
        },
    ]

    resultados = []
    for idx, caso in enumerate(casos, start=1):
        consolidado = consolidar_lecturas_evento_placa(caso["lecturas"])
        ok = consolidado["texto_final"] == caso["esperado"]
        ruta_debug = guardar_debug_votacion_evento(idx, caso["lecturas"], consolidado)
        resultados.append({**caso, "consolidado": consolidado, "ok": ok, "ruta_debug": ruta_debug})
        print(
            f"{caso['nombre']}: esperado={caso['esperado']} obtenido={consolidado['texto_final']} "
            f"estado={consolidado['estado']} confianza={consolidado['confianza_final']:.3f} ok={ok}"
        )
    salida = Path("reports/evidencias/reconocimiento_caracteres/votacion_eventos/test_votacion_evento_placa.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    errores = [item for item in resultados if not item["ok"]]
    print("\nResumen")
    print("-------")
    print(f"Casos: {len(resultados)}")
    print(f"Errores: {len(errores)}")
    print(f"JSON: {salida}")
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

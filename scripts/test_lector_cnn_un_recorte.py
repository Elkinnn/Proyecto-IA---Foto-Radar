from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plate_reader import (
    cargar_modelo_caracteres,
    leer_placa_desde_recorte,
    preprocesar_recorte_placa_para_ocr,
    rectificar_placa_para_lector_cnn,
    segmentar_caracteres_v3_estrategias,
)


SALIDA = Path("reports/evidencias/reconocimiento_caracteres/test_un_recorte")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba minima del lector CNN con un recorte de placa.")
    parser.add_argument("--image", required=True, help="Ruta del recorte de placa.")
    parser.add_argument("--placa-esperada", default="", help="Placa esperada opcional para diagnostico.")
    parser.add_argument("--expected", default="", help="Alias de --placa-esperada.")
    args = parser.parse_args()
    placa_esperada = args.expected or args.placa_esperada

    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = Path(args.image)
    imagen = cv2.imread(str(ruta))
    modelo_info, mensaje_modelo = cargar_modelo_caracteres()
    print(f"Modelo cargado: {'si' if modelo_info is not None else 'no'}")
    print(f"Mensaje modelo: {mensaje_modelo}")
    print(f"Recorte: {ruta}")
    print(f"Shape recorte: {list(imagen.shape) if imagen is not None else 'no_cargado'}")
    if imagen is not None:
        rectificacion = rectificar_placa_para_lector_cnn(imagen, ruta.stem)
        pre_sin = preprocesar_recorte_placa_para_ocr(imagen, f"{ruta.stem}_sin_rectificacion")
        seg_sin = segmentar_caracteres_v3_estrategias(pre_sin["ruta_imagen_procesada"], f"{ruta.stem}_sin_rectificacion") if pre_sin.get("ruta_imagen_procesada") else {}
        pre_con = preprocesar_recorte_placa_para_ocr(rectificacion["placa_rectificada"], f"{ruta.stem}_con_rectificacion")
        seg_con = segmentar_caracteres_v3_estrategias(pre_con["ruta_imagen_procesada"], f"{ruta.stem}_con_rectificacion") if pre_con.get("ruta_imagen_procesada") else {}
        print("Rectificacion previa:")
        print(f"  Metodo: {rectificacion.get('metodo_rectificacion')}")
        print(f"  Puntaje: {rectificacion.get('confianza_rectificacion')}")
        print(f"  Imagen seleccionada: {rectificacion.get('ruta_seleccionada')}")
        print(f"  Chars sin rectificar: {len(seg_sin.get('caracteres', []))} | puntaje={seg_sin.get('puntaje_segmentacion')}")
        print(f"  Chars con rectificacion: {len(seg_con.get('caracteres', []))} | puntaje={seg_con.get('puntaje_segmentacion')}")

    resultado = leer_placa_desde_recorte(str(ruta), placa_esperada=placa_esperada)
    print(f"OK: {resultado.get('ok')}")
    print(f"Estado lectura: {resultado.get('estado_lectura')}")
    print(f"Motivo: {resultado.get('motivo') or resultado.get('motivo_sin_lectura')}")
    print(f"Caracteres segmentados: {resultado.get('cantidad_caracteres_segmentados')}")
    print(f"Texto crudo: {resultado.get('texto_detectado_crudo') or resultado.get('texto_crudo')}")
    print(f"Texto corregido: {resultado.get('texto_postprocesado') or resultado.get('texto_corregido_formato')}")
    print(f"Confianza: {resultado.get('confianza_promedio') or resultado.get('confianza_cnn_caracteres')}")
    print(f"Formato valido: {resultado.get('formato_valido')}")
    print(f"Metodo rectificacion: {resultado.get('metodo_rectificacion') or (resultado.get('rectificacion') or {}).get('metodo_rectificacion')}")
    print(f"Puntaje rectificacion: {resultado.get('puntaje_rectificacion') or (resultado.get('rectificacion') or {}).get('confianza_rectificacion')}")
    print(f"Debug rectificacion: {(resultado.get('rectificacion') or {}).get('ruta_seleccionada')}")
    print(f"Estrategia: {(resultado.get('segmentacion') or {}).get('estrategia_segmentacion')}")
    print(f"Puntaje segmentacion: {(resultado.get('segmentacion') or {}).get('puntaje_segmentacion')}")
    print(f"Debug segmentacion: {resultado.get('ruta_debug_segmentacion')}")
    print(f"Debug sin lectura: {resultado.get('ruta_debug_sin_lectura')}")

    for pred in resultado.get("predicciones_caracteres", []):
        top3 = ", ".join([f"{item.get('caracter')}:{float(item.get('confianza', 0.0)):.2f}" for item in pred.get("top3_predicciones", [])])
        print(
            f"Char {pred.get('indice')}: pred={pred.get('caracter_predicho')} "
            f"conf={float(pred.get('confianza', 0.0)):.3f} top3=[{top3}] "
            f"shape={pred.get('shape_array')}"
        )
    if placa_esperada:
        esperado = "".join(ch for ch in placa_esperada.upper() if ch.isalnum())
        detectado = resultado.get("texto_postprocesado") or resultado.get("texto_corregido_formato") or ""
        print("Comparacion esperada:")
        for idx, esperado_char in enumerate(esperado):
            detectado_char = detectado[idx] if idx < len(detectado) else "-"
            marca = "OK" if esperado_char == detectado_char else "X"
            print(f"  {idx + 1}: esperado={esperado_char} detectado={detectado_char} {marca}")

    salida_json = SALIDA / f"{ruta.stem}_resultado.json"
    salida_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {salida_json}")


if __name__ == "__main__":
    main()

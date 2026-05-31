# Fotorradar Ecuador IA

Sistema integrado para reconocimiento de placas ecuatorianas y control de velocidad vehicular en contexto de campus universitario.

## Objetivo

Construir una base limpia, organizada y funcional para un fotorradar controlado desde interfaz grafica. El proyecto integra deteccion de placa, lectura OCR, estimacion de velocidad, clasificacion difusa, consulta de base de datos, registro de eventos y generacion de evidencia.

## Arquitectura

- `app.py`: interfaz Streamlit organizada como consola de monitoreo por video/camara.
- `src/pipeline.py`: coordinador principal del flujo completo.
- `src/plate_detector.py`: detector de placas, preparado para cargar un modelo entrenado localmente.
- `src/plate_reader.py`: lector de placa con modo `manual_controlado` y estructura para OCR automatico futuro.
- `src/speed_estimator.py`: calculo de velocidad en km/h.
- `src/fuzzy_system.py`: clasificacion de velocidad y sancion segun rangos configurables.
- `src/database.py`: base SQLite con vehiculos y eventos.
- `src/notifier.py`: notificaciones en modo simulacion.
- `src/report_generator.py`: evidencia en JSON.
- `scripts/`: utilidades conectadas al sistema final.

## Placas ecuatorianas

El proyecto trabaja solo con placas ecuatorianas para evitar mezclar patrones visuales, formatos y distribuciones de datos de otros paises. Esta restriccion ayuda a mantener coherencia entre el dataset, el entrenamiento futuro y la evaluacion del sistema.

## Modelos

No se entrenan modelos todavia. El detector no descarga pesos ni usa pesos preentrenados. Cuando se entrene YOLO, debe iniciarse desde arquitectura `.yaml` y con `pretrained=False`.

## Interfaz

La pantalla principal es `Monitoreo`, con fuentes para `Video de prueba` o `Camara en vivo`. La entrada por imagen queda separada en la pestaña `Pruebas`, dentro de `Pruebas con imagen`.

La pestaña de monitoreo dibuja dos lineas virtuales horizontales sobre el frame procesado y deja preparada la estructura para calcular el tiempo real de cruce entre ambas lineas cuando se integre tracking vehicular.

## OCR

El OCR tiene dos modos:

- `manual_controlado`: modo temporal para pruebas internas, separado del flujo principal.
- `automatico`: estructura preparada para integrar despues un OCR real entrenado con datos ecuatorianos.

## Ejecutar

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/test_pipeline.py
streamlit run app.py
```

## Prueba del pipeline

`scripts/test_pipeline.py` crea una imagen controlada, usa una placa ecuatoriana de prueba, calcula una velocidad simulada, consulta SQLite, clasifica la velocidad y guarda una evidencia JSON.

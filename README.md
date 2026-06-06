# Fotorradar Ecuador IA

Sistema integrado para reconocimiento de placas ecuatorianas y control de velocidad vehicular en contexto de campus universitario.

## Objetivo

Construir una base limpia, organizada y funcional para un fotorradar controlado desde interfaz grafica. El proyecto integra deteccion de placa, reconocimiento de caracteres con CNN propia, estimacion de velocidad, clasificacion difusa, notificacion por correo de la mejor lectura por evento y generacion de evidencia.

## Arquitectura

- `app.py`: interfaz Streamlit organizada como consola de monitoreo por video/camara.
- `src/pipeline.py`: coordinador principal del flujo completo.
- `src/plate_detector.py`: detector de placas, preparado para cargar un modelo entrenado localmente.
- `src/plate_reader.py`: lector de placa con segmentacion OpenCV y clasificacion de caracteres mediante CNN propia.
- `src/speed_estimator.py`: calculo de velocidad en km/h.
- `src/fuzzy_system.py`: clasificacion de velocidad y sancion segun rangos configurables.
- `src/notifier.py`: envio de notificaciones por correo (SMTP) con control de calidad anti media-placa; cae en modo simulado si no hay credenciales.
- `src/report_generator.py`: evidencia en JSON.
- `scripts/`: utilidades conectadas al sistema final.

## Placas ecuatorianas

El proyecto trabaja solo con placas ecuatorianas para evitar mezclar patrones visuales, formatos y distribuciones de datos de otros paises. Esta restriccion ayuda a mantener coherencia entre el dataset, el entrenamiento futuro y la evaluacion del sistema.

## Modelos

El proyecto trabaja con modelos locales. El detector de placas usa YOLO y el lector de caracteres usa una CNN propia entrenada desde cero. No se usan motores OCR externos.

## Interfaz

La pantalla principal es `Monitoreo`, con fuentes para `Video de prueba` o `Camara en vivo`. La entrada por imagen queda separada en la pestaña `Pruebas`, dentro de `Pruebas con imagen`.

La pestaña de monitoreo dibuja dos lineas virtuales horizontales sobre el frame procesado y deja preparada la estructura para calcular el tiempo real de cruce entre ambas lineas cuando se integre tracking vehicular.

## Reconocimiento de caracteres con CNN

El sistema no usa OCR externo. La lectura de caracteres se realiza mediante segmentacion con OpenCV y clasificacion con una CNN propia entrenada desde cero.

Modos relevantes:

- `manual_controlado`: modo temporal para pruebas internas y consultas controladas.
- lector CNN automatico: reconoce caracteres desde recortes de placa cuando existe un recorte valido.

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

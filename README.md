# Reconocimiento de Patrones
## Clasificador SVC y k-nn hecho en Python

A partir de un set de datos hemos entrenado un modelo clasificador que reconoce una serie de etiquetas para predecir una etiqueta resultado.

## ¿Como ejecutar?
Solo depende de las librerias:
	
	1. sklearn
	2. pandas

Si se tienen instaladas debería poder hacerse directamente:

`pyton main.py`

En caso contrario se puede generar un entorno virtual y ejectuarse con:

```
python3 -m venv venv
source venv/bin/source # En linux
pip install -r requirements.txt
python main.py
```

## Pasos del proyecto
	1. Proceso de limpieza de datos
	2. Proceso de normalización de datos
	3. Busqueda de mejores parámetros para cada clasificador
	4. Display y guardado en documentos de métricas
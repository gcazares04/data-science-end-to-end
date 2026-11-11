# La imagen del servicio: el mismo patron de la Semana 3 (dependencias antes
# que codigo), con el modelo ya entrenado adentro. Antes de construirla,
# corre python entrenar.py para que exista modelo_tarifa.joblib.
#
#     docker build -t servicio-tarifa .
#     docker run -p 8000:8000 servicio-tarifa
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY preparacion_taxis.py features_taxis.py servicio.py modelo_tarifa.joblib ./
EXPOSE 8000

# 0.0.0.0: escucha en todas las interfaces del contenedor, no solo en su
# propio localhost, para que docker run -p lo alcance desde afuera.
CMD ["uvicorn", "servicio:app", "--host", "0.0.0.0", "--port", "8000"]

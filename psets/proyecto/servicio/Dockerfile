# Se construye desde psets/proyecto/, despues de exportar el modelo:
#   python servicio/exportar_modelo.py
#   docker build -f servicio/Dockerfile -t grantmatch-monto .
#   docker run --rm -p 8000:8000 grantmatch-monto
FROM python:3.11-slim

WORKDIR /app

COPY servicio/requirements.txt servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt

COPY modelado/pipeline_proyecto.py modelado/pipeline_proyecto.py
COPY servicio/api.py servicio/modelo.joblib servicio/modelo_info.json servicio/

EXPOSE 8000
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]

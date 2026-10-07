FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir spacy==3.8.2 fastapi "uvicorn[standard]" pydantic python-multipart

COPY src src
COPY checkpoints/ner_model_final checkpoints/ner_model_final
ENV NER_MODEL=/app/checkpoints/ner_model_final
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

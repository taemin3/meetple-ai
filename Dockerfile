FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
RUN useradd --create-home appuser
USER appuser
EXPOSE 8001
CMD ["uvicorn", "meetple_ai.app:app", "--host", "0.0.0.0", "--port", "8001"]

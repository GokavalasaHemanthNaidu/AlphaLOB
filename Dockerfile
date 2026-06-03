FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OMP_NUM_THREADS=1
EXPOSE 7860
CMD uvicorn src.api.main:app --host 0.0.0.0 --port 

FROM python:3.11-slim

WORKDIR /app

# Install dependencies as root so they are globally available
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the rest of the application with correct ownership
COPY --chown=user . $HOME/app

ENV OMP_NUM_THREADS=1
EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

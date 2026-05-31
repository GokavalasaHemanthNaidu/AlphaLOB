# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Ensure the ONNX runtime uses purely CPU
ENV OMP_NUM_THREADS=1

# Expose the port Uvicorn will run on
EXPOSE 8000

# Run Uvicorn server when the container launches, respecting Render's dynamic PORT
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}

# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /pipeline

# setting env variable
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_REQUIRE_HASHES=false
ENV PIP_CONFIG_FILE=/dev/null

# Copy requirements and install dependencies
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# Set enviroment variable
ENV GOOGLE_CLOUD_PROJECT=dataservices-dev-stage

# Command to run when the job starts
CMD ["python", "main.py"]
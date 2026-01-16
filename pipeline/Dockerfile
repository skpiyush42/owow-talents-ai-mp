# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /pipeline

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set enviroment variable
ENV GOOGLE_CLOUD_PROJECT=dataservices-dev-stage

# Command to run when the job starts
CMD ["python", "main.py"]
FROM tensorflow/tensorflow:2.16.1-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    google-cloud-aiplatform \
    google-cloud-bigquery[pandas] \
    pandas \
    pyarrow \
    scikit-learn \
    numpy

# Set working directory
WORKDIR /app

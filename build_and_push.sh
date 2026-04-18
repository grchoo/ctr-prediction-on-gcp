#!/bin/bash
set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
# Do not commit your actual Project ID to GitHub.
PROJECT_ID="${GCP_PROJECT_ID:-"YOUR_PROJECT_ID"}"
REGION="${GCP_REGION:-"asia-northeast3"}"
REPO_NAME="${GCP_ARTIFACT_REPO:-"ml-images"}"
IMAGE_NAME="ctr-pipeline-base"
TAG="latest"

if [ "$PROJECT_ID" == "YOUR_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID is not set. Please set it as an environment variable or edit this script."
    exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

echo "Building image for linux/amd64: ${IMAGE_URI}"
docker build --platform linux/amd64 -t ${IMAGE_URI} .

echo "Configuring docker authentication for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo "Pushing image to Artifact Registry..."
docker push ${IMAGE_URI}

echo "✅ Image successfully pushed to ${IMAGE_URI}"

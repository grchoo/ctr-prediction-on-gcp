terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_project_service" "services" {
  for_each = toset([
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com"
  ])
  project = var.project_id
  service = each.value
}

resource "google_storage_bucket" "data_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  depends_on = [google_project_service.services]
}

resource "google_bigquery_dataset" "dataset" {
  dataset_id = var.bq_dataset
  location   = var.region

  depends_on = [google_project_service.services]
}

resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = var.artifact_repo
  format        = "DOCKER"

  depends_on = [google_project_service.services]
}

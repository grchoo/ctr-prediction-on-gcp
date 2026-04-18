output "bucket_name" {
  value = google_storage_bucket.data_bucket.name
}

output "dataset_id" {
  value = google_bigquery_dataset.dataset.dataset_id
}

output "artifact_repo" {
  value = google_artifact_registry_repository.repo.repository_id
}

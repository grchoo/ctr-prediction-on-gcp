variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-northeast3"
}

variable "zone" {
  type    = string
  default = "asia-northeast3-a"
}

variable "bucket_name" {
  type = string
}

variable "bq_dataset" {
  type    = string
  default = "ad_ml"
}

variable "artifact_repo" {
  type    = string
  default = "ml-images"
}


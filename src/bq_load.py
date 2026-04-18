from google.cloud import bigquery
import argparse

SCHEMA = [
    bigquery.SchemaField("id", "STRING"),
    bigquery.SchemaField("click", "INT64"),
    bigquery.SchemaField("hour", "STRING"),
    bigquery.SchemaField("C1", "STRING"),
    bigquery.SchemaField("banner_pos", "STRING"),
    bigquery.SchemaField("site_id", "STRING"),
    bigquery.SchemaField("site_domain", "STRING"),
    bigquery.SchemaField("site_category", "STRING"),
    bigquery.SchemaField("app_id", "STRING"),
    bigquery.SchemaField("app_domain", "STRING"),
    bigquery.SchemaField("app_category", "STRING"),
    bigquery.SchemaField("device_id", "STRING"),
    bigquery.SchemaField("device_ip", "STRING"),
    bigquery.SchemaField("device_model", "STRING"),
    bigquery.SchemaField("device_type", "STRING"),
    bigquery.SchemaField("device_conn_type", "STRING"),
    bigquery.SchemaField("C14", "STRING"),
    bigquery.SchemaField("C15", "STRING"),
    bigquery.SchemaField("C16", "STRING"),
    bigquery.SchemaField("C17", "STRING"),
    bigquery.SchemaField("C18", "STRING"),
    bigquery.SchemaField("C19", "STRING"),
    bigquery.SchemaField("C20", "STRING"),
    bigquery.SchemaField("C21", "STRING"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table", default="avazu_raw")
    parser.add_argument("--gcs_uri", required=True)
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    table_id = f"{args.project}.{args.dataset}.{args.table}"

    job_config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    load_job = client.load_table_from_uri(
        args.gcs_uri,
        table_id,
        job_config=job_config,
    )
    load_job.result()
    print(f"Loaded {args.gcs_uri} into {table_id}")

if __name__ == "__main__":
    main()
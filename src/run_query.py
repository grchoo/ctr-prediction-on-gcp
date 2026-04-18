import argparse
from google.cloud import bigquery
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run BigQuery query from file with variable substitution")
    parser.add_argument("--sql_file", required=True, help="Path to SQL file")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--dataset", required=True, help="Dataset ID")
    args = parser.parse_args()

    # Read SQL file
    sql_path = Path(args.sql_file)
    if not sql_path.exists():
        print(f"Error: {args.sql_file} not found")
        return

    sql_content = sql_path.read_text()

    # Simple variable substitution
    # Replacing ${PROJECT_ID} and ${DATASET_ID}
    sql_content = sql_content.replace("${PROJECT_ID}", args.project)
    sql_content = sql_content.replace("${DATASET_ID}", args.dataset)

    client = bigquery.Client(project=args.project)

    print(f"Executing query from {args.sql_file} for project {args.project}, dataset {args.dataset}...")
    
    try:
        query_job = client.query(sql_content)
        query_job.result()  # Wait for query to complete
        print("Query executed successfully.")
    except Exception as e:
        print(f"Error executing query: {e}")

if __name__ == "__main__":
    main()

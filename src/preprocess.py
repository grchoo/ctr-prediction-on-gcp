import argparse
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple
from dotenv import load_dotenv, find_dotenv

import pandas as pd
from google.cloud import bigquery

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL, TIME_COL

def get_vocab_for_col(args: Tuple[str, pd.Series]) -> Tuple[str, List[str]]:
    col_name, series = args
    # Optimized: use value_counts() instead of Counter
    vocab = series.astype(str).value_counts().head(50000).index.tolist()
    return col_name, vocab

load_dotenv(find_dotenv())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--dataset", default=os.getenv("DATASET_ID"))
    parser.add_argument("--view", default="avazu_feature")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rows", type=int, default=2000000)
    args = parser.parse_args()

    if not args.project or not args.dataset:
        print("Error: project or dataset not set. Check .env or pass as arguments.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    client = bigquery.Client(project=args.project)

    query = f"""
    SELECT * FROM `{args.project}.{args.dataset}.{args.view}`
    ORDER BY {TIME_COL}
    LIMIT {args.sample_rows}
    """
    
    print(f"Reading {args.sample_rows} rows from BigQuery using Storage Read API...")
    query_job = client.query(query)
    # create_bqstorage_client=True enables the Fast Storage Read API (Arrow)
    df = query_job.to_dataframe(create_bqstorage_client=True)
    
    if df.empty:
        print("No data found.")
        return

    print(f"Data loaded. Shape: {df.shape}. Processing...")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # Define paths
    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    test_path = os.path.join(args.out_dir, "test.csv")

    # 1. Parallel Vocab Counting
    print("Accumulating vocabulary in parallel...")
    vocab = {}
    # Passing only required columns to reduce memory overhead during serialization
    tasks = [(col, df[col]) for col in CATEGORICAL_FEATURES]
    
    with ProcessPoolExecutor(max_workers=min(len(CATEGORICAL_FEATURES), os.cpu_count())) as executor:
        results = list(executor.map(get_vocab_for_col, tasks))
        for col_name, col_vocab in results:
            vocab[col_name] = col_vocab

    # 2. Optimized Splitting and Writing
    print("Splitting data and writing to CSV...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    df.iloc[:train_end].to_csv(train_path, index=False)
    print(f"Saved {train_end} rows to {train_path}")
    
    df.iloc[train_end:val_end].to_csv(val_path, index=False)
    print(f"Saved {val_end - train_end} rows to {val_path}")
    
    df.iloc[val_end:].to_csv(test_path, index=False)
    print(f"Saved {n - val_end} rows to {test_path}")

    # 3. Save Metadata
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump({
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "label": LABEL,
            "time_col": TIME_COL,
        }, f)

    print("Preprocess completed successfully")

if __name__ == "__main__":
    main()
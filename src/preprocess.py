import argparse
import json
import os
from typing import List, Dict

import pandas as pandas
from google.cloud import bigquery

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL, TIME_COL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--view", default="avazu_feature")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rows", type=int, default=2000000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    client = bigquery.Client(project=args.project)

    query = f"""
    SELECT * FROM `{args.project}.{args.dataset}.{args.view}`
    ORDER BY event_ts
    LIMIT {args.sample_rows}
    """
    
    print("Reading data from BigQuery in chunks to prevent OOM...")
    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    
    train_end = int(args.sample_rows * 0.8)
    val_end = int(args.sample_rows * 0.9)
    
    from collections import Counter
    vocab_counts = {col: Counter() for col in CATEGORICAL_FEATURES}
    
    current_row = 0
    query_job = client.query(query)
    
    # Write headers for CSV files
    train_col_written = False
    val_col_written = False
    test_col_written = False
    
    import pandas as pd
    
    for df_chunk in query_job.result(page_size=100000).to_dataframe_iterable():
        if df_chunk.empty:
            continue
            
        df_chunk[TIME_COL] = pd.to_datetime(df_chunk[TIME_COL])
        
        # We don't sort here again because query has ORDER BY, but keeping it robust locally isn't strictly needed
        # df_chunk = df_chunk.sort_values(TIME_COL)
        
        chunk_size = len(df_chunk)
        start_idx = current_row
        end_idx = current_row + chunk_size
        
        # Accumulate vocab
        for col in CATEGORICAL_FEATURES:
            vocab_counts[col].update(df_chunk[col].astype(str).tolist())
            
        # Distribute into train/val/test
        if end_idx <= train_end:
            df_chunk.to_csv(train_path, index=False, mode='a', header=not train_col_written)
            train_col_written = True
        elif start_idx >= train_end and end_idx <= val_end:
            df_chunk.to_csv(val_path, index=False, mode='a', header=not val_col_written)
            val_col_written = True
        elif start_idx >= val_end:
            df_chunk.to_csv(test_path, index=False, mode='a', header=not test_col_written)
            test_col_written = True
        else:
            # Chunk overlaps splits
            train_mask = (df_chunk.index + start_idx) < train_end
            val_mask = ((df_chunk.index + start_idx) >= train_end) & ((df_chunk.index + start_idx) < val_end)
            test_mask = (df_chunk.index + start_idx) >= val_end
            
            if train_mask.any():
                df_chunk[train_mask].to_csv(train_path, index=False, mode='a', header=not train_col_written)
                train_col_written = True
            if val_mask.any():
                df_chunk[val_mask].to_csv(val_path, index=False, mode='a', header=not val_col_written)
                val_col_written = True
            if test_mask.any():
                df_chunk[test_mask].to_csv(test_path, index=False, mode='a', header=not test_col_written)
                test_col_written = True
        current_row += chunk_size
        print(f"Processed {current_row}/{args.sample_rows} rows...")

    vocab = {}
    for col in CATEGORICAL_FEATURES:
        vocab[col] = [item[0] for item in vocab_counts[col].most_common(50000)]

    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump({
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "label": LABEL,
            "time_col": TIME_COL,
        }, f)

    print("Preprocess completed")

if __name__ == "__main__":
    main()
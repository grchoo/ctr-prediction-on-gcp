import argparse
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    # GPU Memory Growth configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled GPU Memory Growth")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    y = test_df[LABEL].values

    print(f"Loading model from {args.model_dir}...")
    infer = tf.saved_model.load(os.path.join(args.model_dir, "saved_model"))
    serving_fn = infer.signatures["serving_default"]

    print(f"Starting batched inference (batch_size={args.batch_size}) for {len(test_df)} rows...")
    all_preds = []
    
    # Process in batches to avoid GPU OOM
    for i in range(0, len(test_df), args.batch_size):
        batch_df = test_df.iloc[i : i + args.batch_size]
        
        inputs = {}
        for col in CATEGORICAL_FEATURES:
            inputs[col] = tf.reshape(tf.constant(batch_df[col].astype(str).values, dtype=tf.string), [-1, 1])
        for col in NUMERIC_FEATURES:
            inputs[col] = tf.reshape(tf.cast(tf.constant(batch_df[col].values), tf.float32), [-1, 1])
        
        outputs = serving_fn(**inputs)
        # Assuming the first output contains the predictions
        preds = list(outputs.values())[0].numpy().flatten()
        all_preds.extend(preds)
        
        if (i // args.batch_size) % 5 == 0:
            print(f"Processed {min(i + args.batch_size, len(test_df))}/{len(test_df)} rows...")

    auc = roc_auc_score(y, all_preds)
    logloss = log_loss(y, all_preds)
    print(f"\n✅ Evaluation Results:")
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {logloss:.4f}")

if __name__ == "__main__":
    main()
    
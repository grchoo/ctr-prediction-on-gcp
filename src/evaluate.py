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
    args = parser.parse_args()

    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    y = test_df[LABEL].values

    infer = tf.saved_model.load(os.path.join(args.model_dir, "saved_model"))
    serving_fn = infer.signatures["serving_default"]

    inputs = {}
    for col in CATEGORICAL_FEATURES:
        inputs[col] = tf.constant(test_df[col].astype(str)).values.resahpe(-1, 1)
    for col in NUMERIC_FEATURES:
        inputs[col] = tf.constant(test_df[col].astype(float)).values.resahpe(-1, 1)
    
    preds = serving_fn(**inputs)["pctr"].numpy().flatten()

    auc = roc_auc_score(y, preds)
    logloss = log_loss(y, preds)
    print(f"AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

if __name__ == "__main__":
    main()    
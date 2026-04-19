import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL

AUTOTUNE = tf.data.AUTOTUNE

def process_features(features, labels):
    processed_features = {}
    for col in CATEGORICAL_FEATURES:
        processed_features[col] = tf.strings.as_string(features[col])
    for col in NUMERIC_FEATURES:
        processed_features[col] = tf.cast(features[col], tf.float32)
    return processed_features, tf.reshape(tf.cast(labels, tf.float32), [-1, 1])

def make_dataset(csv_path: str, batch_size: int = 8192, shuffle: bool = True) -> tf.data.Dataset:
    ds = tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=batch_size,
        label_name=LABEL,
        num_epochs=1,
        header=True,
        shuffle=shuffle,
        shuffle_buffer_size=100000 if shuffle else None,
        ignore_errors=True
    )
    ds = ds.map(process_features, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

class CrossLayerV2(tf.keras.layers.Layer):
    def __init__(self, num_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.dense_layers = [tf.keras.layers.Dense(input_shape[-1]) for _ in range(self.num_layers)]

    def call(self, x0):
        xl = x0
        for i in range(self.num_layers):
            xl = x0 * self.dense_layers[i](xl) + xl
        return xl

def get_encoded_inputs(vocab: Dict[str, List[str]], constant_emb_dim: int = None):
    inputs = {}
    encoded = []
    
    for col in CATEGORICAL_FEATURES:
        inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)
        lookup = tf.keras.layers.StringLookup(
            vocabulary=vocab[col],
            mask_token=None,
            num_oov_indices=1,
            output_mode="int"
        )
        x = lookup(inputs[col])
        emb_dim = constant_emb_dim if constant_emb_dim else min(32, max(4, int(np.ceil(np.log2(len(vocab[col])+2)))))
        x = tf.keras.layers.Embedding(
            input_dim=len(vocab[col])+2,
            output_dim=emb_dim,
            name=f"{col}_emb"
        )(x)
        x = tf.keras.layers.Reshape((emb_dim,))(x)
        encoded.append(x)
        
    for col in NUMERIC_FEATURES:
        inputs[col] = tf.keras.layers.Input(shape=(1,), name=col, dtype=tf.float32)
        # Reshape to ensure rank 2 even if input is provided as a scalar/rank 1
        x = tf.keras.layers.Reshape((1,), name=f"{col}_reshape")(inputs[col])
        if constant_emb_dim:
            x = tf.keras.layers.Dense(constant_emb_dim, name=f"{col}_proj")(x)
            encoded.append(x)
        else:
            encoded.append(x)
            
    return inputs, encoded

def build_dcn_v2_model(vocab: Dict[str, List[str]]) -> tf.keras.Model:
    inputs, encoded = get_encoded_inputs(vocab)
    x0 = tf.keras.layers.Concatenate()(encoded)
    
    cross_out = CrossLayerV2(num_layers=3)(x0)
    
    dnn_out = tf.keras.layers.Dense(256, activation="relu")(x0)
    dnn_out = tf.keras.layers.BatchNormalization()(dnn_out)
    dnn_out = tf.keras.layers.Dropout(0.2)(dnn_out)
    dnn_out = tf.keras.layers.Dense(128, activation="relu")(dnn_out)
    dnn_out = tf.keras.layers.BatchNormalization()(dnn_out)
    dnn_out = tf.keras.layers.Dropout(0.2)(dnn_out)
    dnn_out = tf.keras.layers.Dense(64, activation="relu")(dnn_out)
    
    stacked = tf.keras.layers.Concatenate()([cross_out, dnn_out])
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="pctr")(stacked)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryCrossentropy(name="logloss")]
    )
    return model

def build_autoint_model(vocab: Dict[str, List[str]], emb_dim: int = 32) -> tf.keras.Model:
    inputs, encoded = get_encoded_inputs(vocab, constant_emb_dim=emb_dim)
    
    # Use Lambda layer to wrap tf.stack for Keras 3 compatibility
    stacked_embeddings = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(encoded)
    
    attention_output = stacked_embeddings
    for _ in range(3): # 3 Attention layers
        attn_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=emb_dim)
        res = attn_layer(attention_output, attention_output)
        attention_output = tf.keras.layers.Add()([attention_output, res])
        attention_output = tf.keras.layers.LayerNormalization()(attention_output)
        attention_output = tf.keras.layers.Activation('relu')(attention_output)
        
    x = tf.keras.layers.Flatten()(attention_output)
    
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="pctr")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryCrossentropy(name="logloss")]
    )
    return model
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--model_type", type=str, choices=["dcn_v2", "autoint"], default="dcn_v2", help="Model architecture")
    # GPU Memory Growth configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled GPU Memory Growth")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)
    
    train_ds = make_dataset(os.path.join(args.data_dir, "train.csv"), args.batch_size, shuffle=True)
    val_ds = make_dataset(os.path.join(args.data_dir, "val.csv"), args.batch_size, shuffle=False)

    if args.model_type == "dcn_v2":
        model = build_dcn_v2_model(vocab)
    else:
        model = build_autoint_model(vocab)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=2,
            mode="max",
            restore_best_weights=True,
        )
    ]

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    with open(os.path.join(args.model_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

    export_dir = os.path.join(args.model_dir, "saved_model")
    model.export(export_dir)
    print(f"Saved model to {export_dir}")

if __name__ == "__main__":
    main()    
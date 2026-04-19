from kfp import dsl
from kfp.dsl import component, Output, Dataset, Model, Artifact, Metrics, ClassificationMetrics
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# Get base image from environment variable or use a placeholder for local compilation
# Do not hardcode your project ID here if uploading to GitHub.
BASE_IMAGE = os.getenv("GCP_PIPELINE_BASE_IMAGE")

@component(
    base_image=BASE_IMAGE,
)
def preprocess_op(
    project: str,
    dataset: str,
    output_dir: Output[Artifact]
):
    import os
    import json
    import pandas as pd
    from google.cloud import bigquery

    CATEGORICAL_FEATURES = [
        'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 
        'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 
        'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 
        'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]
    NUMERIC_FEATURES = [
        'hour_of_day', 'day_of_week', 'is_weekend'
    ]
    os.makedirs(output_dir.path, exist_ok=True)
    client = bigquery.Client(project=project)
    query = f"""
    SELECT * FROM `{project}.{dataset}.avazu_feature`
    ORDER BY event_ts
    LIMIT 2000000
    """
    print("Reading data from BigQuery in chunks to prevent OOM...")
    train_path = os.path.join(output_dir.path, "train.csv")
    val_path = os.path.join(output_dir.path, "val.csv")
    test_path = os.path.join(output_dir.path, "test.csv")
    
    sample_rows = 2000000
    train_end = int(sample_rows * 0.8)
    val_end = int(sample_rows * 0.9)
    
    from collections import Counter
    vocab_counts = {col: Counter() for col in CATEGORICAL_FEATURES}
    
    current_row = 0
    query_job = client.query(query)
    
    train_col_written = False
    val_col_written = False
    test_col_written = False
    
    for df_chunk in query_job.result(page_size=100000).to_dataframe_iterable():
        if df_chunk.empty:
            continue
            
        df_chunk['event_ts'] = pd.to_datetime(df_chunk['event_ts'])
        
        chunk_size = len(df_chunk)
        start_idx = current_row
        end_idx = current_row + chunk_size
        
        for col in CATEGORICAL_FEATURES:
            vocab_counts[col].update(df_chunk[col].astype(str).tolist())
            
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

    vocab = {col: [item[0] for item in vocab_counts[col].most_common(50000)] for col in CATEGORICAL_FEATURES}

    with open(os.path.join(output_dir.path, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    with open(os.path.join(output_dir.path, "feature_spec.json"), "w") as f:
        json.dump({
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "label": "click"
        }, f)

@component(
    base_image=BASE_IMAGE,
)
def train_op(
    data_dir: dsl.Input[Artifact],
    model_dir: dsl.Output[Model],
    model_type: str = "dcn_v2"
):
    import os
    import json
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    
    with open(os.path.join(data_dir.path, "vocab.json")) as f:
        vocab = json.load(f)

    with open(os.path.join(data_dir.path, "feature_spec.json")) as f:
        spec = json.load(f)

    CATEGORICAL_FEATURES = spec["categorical"]
    NUMERIC_FEATURES = spec["numeric"]
    LABEL = spec["label"]

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
        return ds.map(process_features, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

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

    def get_encoded_inputs(constant_emb_dim=None):
        inputs = {}
        encoded = []
        for col in CATEGORICAL_FEATURES:
            inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)
            lookup = tf.keras.layers.StringLookup(vocabulary=vocab[col], mask_token=None, num_oov_indices=1, output_mode="int")
            x = lookup(inputs[col])
            emb_dim = constant_emb_dim if constant_emb_dim else min(32, max(4, int(np.ceil(np.log2(len(vocab[col])+2)))))
            x = tf.keras.layers.Embedding(input_dim=len(vocab[col])+2, output_dim=emb_dim, name=f"{col}_emb")(x)
            x = tf.keras.layers.Reshape((emb_dim,))(x)
            encoded.append(x)
        for col in NUMERIC_FEATURES:
            inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)
            x = tf.keras.layers.Reshape((1,), name=f"{col}_reshape")(inputs[col])
            if constant_emb_dim:
                x = tf.keras.layers.Dense(constant_emb_dim, name=f"{col}_proj")(x)
                encoded.append(x)
            else:
                encoded.append(x)
        return inputs, encoded

    if model_type == "dcn_v2":
        inputs, encoded = get_encoded_inputs()
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
    else:
        emb_dim = 32
        inputs, encoded = get_encoded_inputs(constant_emb_dim=emb_dim)
        # Use Lambda layer to wrap tf.stack for Keras 3 compatibility
        attention_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(encoded)
        for _ in range(3):
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
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    
    train_ds = make_dataset(os.path.join(data_dir.path, "train.csv"), 8192, True)
    val_ds = make_dataset(os.path.join(data_dir.path, "val.csv"), 8192, False)
    
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.export(model_dir.path)

@component(
    base_image=BASE_IMAGE,
)
def eval_op(
    data_dir: dsl.Input[dsl.Artifact],
    model_dir: dsl.Input[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    output_artifact: dsl.Output[dsl.Artifact],
):
    import os
    import json
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, log_loss

    with open(os.path.join(data_dir.path, "feature_spec.json")) as f:
        spec = json.load(f)

    CATEGORICAL_FEATURES = spec["categorical"]
    NUMERIC_FEATURES = spec["numeric"]
    LABEL = spec["label"]

    df = pd.read_csv(os.path.join(data_dir.path, "test.csv"))
    y = df[LABEL].values

    infer = tf.saved_model.load(model_dir.path)
    serving_fn = infer.signatures["serving_default"]

    inputs = {}
    for col in CATEGORICAL_FEATURES:
        inputs[col] = tf.reshape(tf.constant(df[col].astype(str).values, dtype=tf.string), [-1, 1])
    for col in NUMERIC_FEATURES:
        # Explicitly cast to float32 to avoid 'double tensor' (float64) errors
        inputs[col] = tf.reshape(tf.cast(tf.constant(df[col].values), tf.float32), [-1, 1])
    
    # Handle potentially dynamic output names (e.g., 'pctr' vs 'output_0')
    outputs = serving_fn(**inputs)
    preds = list(outputs.values())[0].numpy().flatten()
    auc = roc_auc_score(y, preds)
    logloss = log_loss(y, preds)
    print(f"AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

    # Log metrics to Vertex AI Experiments
    metrics.log_metric("auc", float(auc))
    metrics.log_metric("logloss", float(logloss))

    # Save results to GCS as a JSON artifact
    results = {
        "auc": float(auc),
        "logloss": float(logloss),
        "sample_count": len(y)
    }
    with open(output_artifact.path, "w") as f:
        json.dump(results, f)


@component(
    base_image=BASE_IMAGE,
)
def deploy_op(
    project: str,
    region: str,
    model: dsl.Input[Model],
    deployment_metadata: dsl.Output[Artifact],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    print("Uploading model to Vertex AI Model Registry...")
    uploaded_model = aiplatform.Model.upload(
        display_name="avazu-ctr-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
    )

    print("Creating Endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name="avazu-ctr-endpoint")

    print("Deploying model to Endpoint...")
    endpoint.deploy(
        model=uploaded_model,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
    )
    print(f"Deployment successful! Endpoint resource name: {endpoint.resource_name}")

    # Save deployment metadata to GCS
    import json
    metadata = {
        "endpoint": endpoint.resource_name,
        "model": uploaded_model.resource_name,
        "artifact_uri": model.uri
    }
    with open(deployment_metadata.path, "w") as f:
        json.dump(metadata, f)


@dsl.pipeline(
    name="ctr-prediction-pipeline",
    description="CTR prediction pipeline",
)
def ctr_pipeline(
    project: str,
    dataset: str,
    model_type: str = "dcn_v2"
):
    prep = preprocess_op(project=project, dataset=dataset)
    
    train = train_op(data_dir=prep.outputs['output_dir'], model_type=model_type)
    train.set_accelerator_type('NVIDIA_TESLA_T4')
    train.set_accelerator_count(1)
    train.set_cpu_limit('4')
    train.set_memory_limit('16G')
    
    eval = eval_op(data_dir=prep.outputs['output_dir'], model_dir=train.outputs['model_dir'])
    deploy = deploy_op(project=project, region="asia-northeast3", model=train.outputs['model_dir'])
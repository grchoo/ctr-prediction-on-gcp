import argparse
import subprocess
import os
import shutil

def main():
    parser = argparse.ArgumentParser(description="Download model from GCS or copy from local to run TF Serving")
    parser.add_argument("--model_uri", required=True, 
                        help="GCS URI or Local path where SavedModel is located (e.g. gs://YOUR_BUCKET/pipeline_root/.../model_dir or /tmp/ctr_model_dcn)")
    parser.add_argument("--port", default=8501, type=int)
    args = parser.parse_args()

    local_model_base = "/tmp/tfserving_avazu_ctr"
    version_dir = os.path.join(local_model_base, "1")
    
    # 1. Clean previous model
    if os.path.exists(local_model_base):
        shutil.rmtree(local_model_base)
    os.makedirs(version_dir, exist_ok=True)

    uri = args.model_uri.rstrip('/')
    
    # 2. Download or copy model artifacts
    # TF Serving expects contents of SavedModel directly under version folder (/models/name/1/saved_model.pb)
    if uri.startswith("gs://"):
        print(f"📥 Downloading model from {uri} to local temp directory...")
        # Check if saved_model subfolder exists on GCS (simple check via gsutil)
        has_subfolder = subprocess.run(f"gsutil -q ls {uri}/saved_model/", shell=True).returncode == 0
        target_uri = f"{uri}/saved_model/*" if has_subfolder else f"{uri}/*"
        subprocess.run(f"gsutil -m cp -r {target_uri} {version_dir}/", shell=True, check=True)
    else:
        print(f"📥 Copying local model from {uri} to temp directory...")
        # Check if saved_model subfolder exists locally
        subfolder_path = os.path.join(uri, "saved_model")
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            print(f"🔍 Found 'saved_model' subfolder, copying its contents...")
            subprocess.run(f"cp -r {subfolder_path}/* {version_dir}/", shell=True, check=True)
        else:
            subprocess.run(f"cp -r {uri}/* {version_dir}/", shell=True, check=True)

    # 3. Start TF Serving via Docker
    print(f"\n🚀 Starting TF Serving on PORT {args.port}...")
    print(f"❗ Make sure Docker daemon is running on your machine.")
    print("Press Ctrl+C to stop the server.\n")
    
    docker_cmd = [
        "docker", "run", "--rm",
        "-p", f"{args.port}:8501",
        "--mount", f"type=bind,source={local_model_base},target=/models/avazu_ctr",
        "-e", "MODEL_NAME=avazu_ctr",
        "-t", "tensorflow/serving"
    ]
    
    try:
        subprocess.run(docker_cmd)
    except KeyboardInterrupt:
        print("\n⏹️ TF Serving successfully stopped.")
    except FileNotFoundError:
        print("\n❌ Error: Docker is not installed or not in PATH.")
    except Exception as e:
        print(f"\n❌ Error starting container: {e}")

if __name__ == "__main__":
    main()

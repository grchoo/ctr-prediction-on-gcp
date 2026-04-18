import argparse
import sys
import os
from dotenv import load_dotenv

try:
    from google.cloud import aiplatform
except ImportError:
    print("Error: google-cloud-aiplatform is not installed. Please run:")
    print("uv add google-cloud-aiplatform")
    sys.exit(1)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def main():
    parser = argparse.ArgumentParser(description="Test Vertex AI Endpoint Prediction")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"), help="GCP Project ID")
    parser.add_argument("--region", default=os.getenv("GCP_REGION", "asia-northeast3"), help="Region where endpoint is deployed")
    parser.add_argument("--endpoint_id", default=os.getenv("ENDPOINT_ID"), help="The numeric ID of your Vertex AI Endpoint")
    args = parser.parse_args()

    # Initialize Vertex AI SDK
    aiplatform.init(project=args.project, location=args.region)
    
    # Construct endpoint resource name
    endpoint_name = f"projects/{args.project}/locations/{args.region}/endpoints/{args.endpoint_id}"
    
    print(f"🔗 Connecting to Endpoint: {args.endpoint_id}...")
    try:
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
    except Exception as e:
        print(f"❌ Failed to connect to endpoint: {e}")
        sys.exit(1)

    # Sample test payload (mirrors schema of avazu_feature)
    instances = [
        {
            "C1": "1005",
            "banner_pos": "0",
            "site_id": "85f751fd",
            "site_domain": "c4e18dd6",
            "site_category": "50e219e0",
            "app_id": "ecad2386",
            "app_domain": "7801e8d9",
            "app_category": "07d7df22",
            "device_id": "a99f214a",
            "device_ip": "1fbe01fe",
            "device_model": "1f0bc64f",
            "device_type": "1",
            "device_conn_type": "0",
            "C14": "15706",
            "C15": "320",
            "C16": "50",
            "C17": "1722",
            "C18": "0",
            "C19": "35",
            "C20": "-1",
            "C21": "79",
            "hour_of_day": [13.0],
            "day_of_week": [2.0],
            "is_weekend": [0.0]
        }
    ]

    print(f"🚀 Sending prediction request...")
    try:
        response = endpoint.predict(instances=instances)
        # Assuming the model outputs a 'pctr' tensor
        print("\n✅ Prediction Response:")
        for idx, prediction in enumerate(response.predictions):
            print(f"Instance {idx+1} CTR Score: {prediction}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()

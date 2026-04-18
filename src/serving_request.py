import json
import requests
import sys

def main():
    # Model serving URL for TensorFlow Serving (default REST port 8501)
    url = "http://localhost:8501/v1/models/avazu_ctr:predict"

    # Sample payload representing one row of the Avazu dataset
    # Note: Handcrafted cross-features (banner_x_device, c1_x_hour) were removed.
    payload = {
        "instances": [
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
                "hour_of_day": 13.0,
                "day_of_week": 2.0,
                "is_weekend": 0.0
            }
        ]
    }

    print(f"🚀 Sending local prediction request to {url}...")
    try:
        resp = requests.post(url, json=payload, timeout=10)
        
        if resp.status_code == 200:
            print(f"✅ Prediction Response (Status {resp.status_code}):")
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"❌ Error (Status {resp.status_code}):")
            print(resp.text)
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to localhost:8501.")
        print("Please ensure TF Serving docker container is running.")
        sys.exit(1)

if __name__ == "__main__":
    main()

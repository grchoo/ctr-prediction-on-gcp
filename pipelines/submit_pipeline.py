import argparse
import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def main():
    parser = argparse.ArgumentParser(description="Submit CTR pipeline to Vertex AI")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"), help="GCP Project ID")
    parser.add_argument("--region", default=os.getenv("GCP_REGION", "asia-northeast3"), help="GCP Region")
    parser.add_argument("--pipeline_root", default=os.getenv("PIPELINE_ROOT"), help="GCS URI for Pipeline Root")
    parser.add_argument("--dataset", default=os.getenv("DATASET_ID", "ad_ml"), help="BigQuery Dataset ID")
    parser.add_argument("--model_type", default="dcn_v2", choices=["dcn_v2", "autoint"])
    parser.add_argument("--template_path", default="ctr_pipeline.yaml")
    args = parser.parse_args()

    if not args.project or not args.pipeline_root:
        parser.error("Project ID and Pipeline Root must be provided via arguments or .env file.")

    # Initialize Vertex AI SDK
    aiplatform.init(project=args.project, location=args.region)

    # Define pipeline parameters
    parameter_values = {
        "project": args.project,
        "dataset": args.dataset,
        "model_type": args.model_type,
    }

    print(f"Submitting pipeline to Vertex AI...")
    print(f"Project: {args.project}, Region: {args.region}, Root: {args.pipeline_root}")
    
    # Create PipelineJob
    job = aiplatform.PipelineJob(
        display_name=f"ctr-pipeline-{args.model_type}",
        template_path=args.template_path,
        pipeline_root=args.pipeline_root,
        parameter_values=parameter_values,
        enable_caching=True,
    )
    
    # Submit the job
    job.submit()
    print("\n✅ Pipeline submitted successfully!")
    print(f"You can monitor it via the generated URL above or the Vertex AI Console.")

if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
from kfp import compiler

# Load environment variables from .env file at the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from ctr_pipeline import ctr_pipeline

compiler.Compiler().compile(pipeline_func=ctr_pipeline, package_path="ctr_pipeline.yaml")
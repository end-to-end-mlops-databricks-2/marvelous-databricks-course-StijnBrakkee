import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from merktrouw.config import ProjectConfig, Tags
from merktrouw.models.custom_merktrouw_model import MerktrouwModel

# Set databricks uri's
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Set DAB parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path

config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model with the config path
model = MerktrouwModel(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[f"{root_path}/files/notebooks/merktrouw-0.1.4-py3-none-any.whl"],
)
logger.info("Model initialized.")


model.load_data()
model.prepare_features()
logger.info("Data loaded & features prepared.")

# Train the model
model.train()
logger.info("Model training completed.")
model.log_model()

# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

model_improved = model.model_improved(test_set=test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

if model_improved:
    # Register the model
    latest_version = model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)

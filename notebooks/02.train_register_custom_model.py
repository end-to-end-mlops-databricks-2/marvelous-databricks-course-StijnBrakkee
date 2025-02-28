# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from merktrouw.config import ProjectConfig, Tags
from merktrouw.models.custom_merktrouw_model import MerktrouwModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
model = MerktrouwModel(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[
        "/Workspace/Users/stijn.brakkee@gibbs.ac/.bundle/mlops-course/dev/files/notebooks/merktrouw-0.0.1-py3-none-any.whl"
    ],
)

# COMMAND ----------
model.load_data()
model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
model.train()
model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(experiment_names=["/Shared/experiments/merktrouw-custom"]).run_id[0]

pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-merktrouw-model")

# COMMAND ----------
# Retrieve dataset for the current run
model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = model.load_latest_model_and_predict(X_test)
# COMMAND ----------

# Databricks notebook source
# MAGIC %pip install merktrouw-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time
from typing import Dict, List

import mlflow
import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from merktrouw.config import ProjectConfig
from merktrouw.serving.model_serving import ModelServing

# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# Set databricks registry
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.merktrouw_model_custom", endpoint_name="merktrouw-model-serving"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
# Create a sample request body
required_columns = [
    "had_eerder_merk",
    "mobipas",
    "had_factuur_13_mnd",
    "had_factuur_ldg_13_mnd",
    "wpl_aantal_facturen_13_mnd",
    "wpl_aantal_facturen_ldg_13_mnd",
    "wpl_bedrag_13_mnd",
    "wpl_bedrag_ldg_13_mnd",
    "is_hybride",
    "geslacht",
    "nieuw_gebruikt_huidig",
    "auto_segment_huidig",
]

# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the test set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'LotFrontage': 78.0,
  'LotArea': 9317,
  'OverallQual': 6,
  'OverallCond': 5,
  'YearBuilt': 2006,
  'Exterior1st': 'VinylSd',
  'Exterior2nd': 'VinylSd',
  'MasVnrType': 'None',
  'Foundation': 'PConc',
  'Heating': 'GasA',
  'CentralAir': 'Y',
  'SaleType': 'WD',
  'SaleCondition': 'Normal'}]
"""


# Call the endpoint with one sample record


def call_endpoint(serving_endpoint: str, record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/merktrouw-model-serving/invocations"
status_code, response_text = call_endpoint(serving_endpoint=serving_endpoint, record=dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

# "load test"
for i in range(len(dataframe_records)):
    print(call_endpoint(serving_endpoint=serving_endpoint, record=dataframe_records[i]))
    time.sleep(0.2)

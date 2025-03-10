import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from merktrouw.config import ProjectConfig
from merktrouw.serving.model_serving import ModelServing

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

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.merktrouw_model_custom", endpoint_name="merktrouw-model-serving"
)


# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")

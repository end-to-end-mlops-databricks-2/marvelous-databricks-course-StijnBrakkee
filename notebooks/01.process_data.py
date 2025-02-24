# Databricks notebook source
# MAGIC %pip install merktrouw-0.0.1-py3-none-any.whl

#/Volumes/mlops_dev/merktrouw/package/
# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import yaml
from pyspark.sql import SparkSession

from merktrouw.config import ProjectConfig
from merktrouw.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data.csv", header=True, inferSchema=True
).toPandas()

# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------

# Split the data
df_train, df_test = data_processor.split_data()
logger.info("Training set shape: %s", df_train.shape)
logger.info("Test set shape: %s", df_test.shape)

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(df_train, df_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
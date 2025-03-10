import argparse
import logging

import yaml
from pyspark.sql import SparkSession

from merktrouw.config import ProjectConfig
from merktrouw.data_processor import DataProcessor

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

# Configure logging
config_path = f"{root_path}/files/project_config.yml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.data_2")

# # Generate synthetic data
# ### This is mimicking a new data arrival. In real world, this would be a new batch of data.
# # df is passed to infer schema
# synthetic_df = generate_synthetic_data(df, num_rows=df.shape[0])
# logger.info("Synthetic data generated.")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
df_train, df_test = data_processor.split_data()
logger.info("Training set shape: %s", df_train.shape)
logger.info("Test set shape: %s", df_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(df_train, df_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()

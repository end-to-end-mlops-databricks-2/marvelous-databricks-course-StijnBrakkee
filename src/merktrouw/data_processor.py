import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split

from merktrouw.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark


    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""

        ## ---- FILTERING ----

        # Only keep rows of finished ownership
        self.df = self.df.filter(self.df.bezit_afgelopen == 1)
 
        # Delete rows with many cars at once or before
        self.df = self.df.filter(self.df.aantal_autos_tegelijk <= 3)
        self.df = self.df.filter(self.df.aantal_eerdere_autos_van_merk <= 5)
    
        # Delete high amounts of aftersales
        self.df = self.df.filter((self.df.wpl_aantal_facturen_13_mnd <= 8) & (self.df.wpl_aantal_facturen_14_24_mnd <= 8))
        self.df = self.df.filter((self.df.wpl_bedrag_13_mnd <= 200000) & (self.df.wpl_bedrag_14_24_mnd <= 200000))

        # Delete very old cars
        self.df = self.df.filter(self.df.leeftijd_auto <= 20)

        # Delete to short ownerships
        self.df = self.df.filter(self.df.bezitsduur_mnd_huidig >= 1)


        ## ---- TRANSFORM TO PANDAS ----
        self.df = self.df.toPandas()


        ## ---- CATEGORICAL VARIABLES ----
        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")


        ## ---- SELECT VARS ----
        # Extract target and relevant features
        target = self.config.target
        num_features = self.config.num_features

        relevant_columns = cat_features + num_features + [target] + ["id_eigenaar_rdc"]
        self.df = self.df[relevant_columns]


    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        
        return train_set, test_set


    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
from typing import List

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from merktrouw.config import ProjectConfig, Tags

"""
infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
"""

"""
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""


class MekrtrouwModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame | np.ndarray):
        predictions = self.model.predict(model_input)
        return {"Prediction": predictions}


class MerktrouwModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: List[str]):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_custom
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        Splits data into:
        Features (X_train, X_test)
        Target (y_train, y_test)
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self):
        """
        Encodes categorical features with OneHotEncoder (ignores unseen categories).
        Passes numerical features as-is (remainder='passthrough').
        Defines a pipeline combining:
            Features processing
            XGB classification model
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", XGBClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self):
        """
        Train the model.
        """
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self):
        """
        Log the model.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_pred = self.pipeline.predict(self.X_test)
            y_score = self.pipeline.predict_proba(self.X_test)[:, 1]

            # Evaluate metrics
            acc = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_score)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {acc}")
            logger.info(f"📊 Roc Auc Score: {roc_auc}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "XGBoost with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1", f1)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output={"Prediction": 1})
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=MekrtrouwModelWrapper(self.pipeline),
                artifact_path="pyfunc-merktrouw-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
            )

    def register_model(self):
        """
        Register model in UC
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-merktrouw-model",
            name=f"{self.catalog_name}.{self.schema_name}.merktrouw_model_custom",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.merktrouw_model_custom",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self):
        """
        Retrieve MLflow run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()
        logger.info("✅ Dataset source loaded.")

    def retrieve_current_run_metadata(self):
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params
        logger.info("✅ Dataset metadata loaded.")

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow (alias=latest-model) and make predictions.
        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.merktrouw_model_custom@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions: None is context
        predictions = model.predict(input_data)

        # This also works
        # model.unwrap_python_model().predict(None, input_data)
        # check out this article:
        # https://medium.com/towards-data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535

        # Return predictions as a DataFrame
        return predictions

    def model_improved(self, test_set: DataFrame):
        """
        Evaluate the model performance on the test set.
        """
        test_set = test_set.toPandas()
        X_test = test_set.drop(self.config.target, axis=1)

        predictions_latest = self.load_latest_model_and_predict(X_test)["Prediction"]
        predictions_current = self.pipeline.predict(X_test)

        logger.info("Predictions are ready.")

        # Calculate the accuracy for each model
        acc_current = accuracy_score(test_set[self.config.target], predictions_current)
        acc_latest = accuracy_score(test_set[self.config.target], predictions_latest)

        # Compare models based on MAE
        logger.info(f"Accuracy for Current Model: {acc_current}")
        logger.info(f"Accuracy for Latest Model: {acc_latest}")

        if acc_current > acc_latest:
            logger.info("Current Model performs better. Registering new model.")
            return True
        else:
            logger.info("New Model performs worse. Keeping the old model.")
            return False

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType


def create_or_refresh_monitoring(config, spark, workspace):
    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`merktrouw-model-serving_payload`")

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("geslacht", StringType(), True),
                            StructField("nieuw_gebruikt_huidig", StringType(), True),
                            StructField("auto_segment_huidig", StringType(), True),
                            StructField("had_eerder_merk", IntegerType(), True),
                            StructField("mobipas", IntegerType(), True),
                            StructField("had_factuur_13_mnd", IntegerType(), True),
                            StructField("had_factuur_ldg_13_mnd", IntegerType(), True),
                            StructField("wpl_aantal_facturen_13_mnd", IntegerType(), True),
                            StructField("wpl_aantal_facturen_ldg_13_mnd", IntegerType(), True),
                            StructField("wpl_bedrag_13_mnd", DoubleType(), True),
                            StructField("wpl_bedrag_ldg_13_mnd", DoubleType(), True),
                            StructField("is_hybride", StringType(), True),
                            StructField("id_eigenaar_rdc", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.id_eigenaar_rdc").alias("id_eigenaar_rdc"),
        F.col("record.geslacht").alias("geslacht"),
        F.col("record.nieuw_gebruikt_huidig").alias("nieuw_gebruikt_huidig"),
        F.col("record.auto_segment_huidig").alias("auto_segment_huidig"),
        F.col("record.had_eerder_merk").alias("had_eerder_merk"),
        F.col("record.mobipas").alias("mobipas"),
        F.col("record.had_factuur_13_mnd").alias("had_factuur_13_mnd"),
        F.col("record.had_factuur_ldg_13_mnd").alias("had_factuur_ldg_13_mnd"),
        F.col("record.wpl_aantal_facturen_13_mnd").alias("wpl_aantal_facturen_13_mnd"),
        F.col("record.wpl_aantal_facturen_ldg_13_mnd").alias("wpl_aantal_facturen_ldg_13_mnd"),
        F.col("record.wpl_bedrag_13_mnd").alias("wpl_bedrag_13_mnd"),
        F.col("record.wpl_bedrag_ldg_13_mnd").alias("wpl_bedrag_ldg_13_mnd"),
        F.col("record.is_hybride").alias("is_hybride"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("merktrouw-model").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    df_final_with_status = (
        df_final.join(test_set.select("id_eigenaar_rdc", "merktrouw"), on="id_eigenaar_rdc", how="left")
        .withColumnRenamed("merktrouw", "merktrouw_test")
        .join(inference_set_skewed.select("id_eigenaar_rdc", "merktrouw"), on="id_eigenaar_rdc", how="left")
        .withColumnRenamed("merktrouw", "merktrouw_inference")
        .select("*", F.coalesce(F.col("merktrouw_test"), F.col("merktrouw_inference")).alias("merktrouw"))
        .drop("merktrouw_test", "merktrouw_inference")
        .withColumn("merktrouw", F.col("merktrouw").cast("int"))
        .withColumn("prediction", F.col("prediction").cast("int"))
        .dropna(subset=["merktrouw", "prediction"])
    )

    df_final_with_status.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="merktrouw",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

import sys
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, isnan, when, count
from pyspark.sql.types import IntegerType, FloatType, StringType, TimestampType
from awsglue.context import GlueContext
from awsglue.transforms import *
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
import boto3
import base64
from botocore.exceptions import ClientError
import json

# Parse arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_PATH', 'SECRET_NAME', 'SNOWFLAKE_WAREHOUSE', 'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA', 'SNOWFLAKE_TABLE'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Function to get secret
def get_secret(secret_name):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    # Decrypts secret using the associated KMS key.
    if 'SecretString' in get_secret_value_response:
        return json.loads(get_secret_value_response['SecretString'])
    else:
        return json.loads(base64.b64decode(get_secret_value_response['SecretBinary']))

# Get Snowflake credentials from Secrets Manager
secret = get_secret(args['SECRET_NAME'])
snowflake_username = secret['sfUser']
snowflake_password = secret['sfPassword']
snowflake_URL = secret['sfURL']

# Load data from S3
s3_input_path = args['S3_PATH']
df = spark.read.format("parquet").option("header", "true").load(s3_input_path)

# 1. Column Rename
df = df.withColumnRenamed("customer_ID", "customer_id") \
    .withColumnRenamed("C_previous", "c_previous")
# Replace NaN values with None (null)
df = df.select([when(isnan(c), None).otherwise(col(c)).alias(c) for c in df.columns])

# 2. Data type casting
df = df.withColumn("customer_id", col("customer_id").cast(IntegerType())) \
    .withColumn("shopping_pt", col("shopping_pt").cast(FloatType())) \
    .withColumn("record_type", col("record_type").cast(FloatType())) \
    .withColumn("day", col("day").cast(FloatType())) \
    .withColumn("time", col("time").cast(TimestampType())) \
    .withColumn("state", col("state").cast(StringType())) \
    .withColumn("location", col("location").cast(FloatType())) \
    .withColumn("group_size", col("group_size").cast(FloatType())) \
    .withColumn("homeowner", col("homeowner").cast(FloatType())) \
    .withColumn("car_age", col("car_age").cast(FloatType())) \
    .withColumn("car_value", col("car_value").cast(StringType())) \
    .withColumn("risk_factor", col("risk_factor").cast(FloatType())) \
    .withColumn("age_oldest", col("age_oldest").cast(FloatType())) \
    .withColumn("age_youngest", col("age_youngest").cast(FloatType())) \
    .withColumn("married_couple", col("married_couple").cast(FloatType())) \
    .withColumn("c_previous", col("c_previous").cast(FloatType())) \
    .withColumn("duration_previous", col("duration_previous").cast(FloatType())) \
    .withColumn("A", col("A").cast(FloatType())) \
    .withColumn("B", col("B").cast(FloatType())) \
    .withColumn("C", col("C").cast(FloatType())) \
    .withColumn("D", col("D").cast(FloatType())) \
    .withColumn("E", col("E").cast(FloatType())) \
    .withColumn("F", col("F").cast(FloatType())) \
    .withColumn("G", col("G").cast(FloatType())) \
    .withColumn("cost", col("cost").cast(FloatType()))

# 3. Drop rows having Null as the target value (assuming 'cost' is the target value)
df = df.na.drop(subset=["cost"])

# 4. Drop rows with more than 50% Null values
threshold = int(len(df.columns)*0.5)
df = df.na.drop(thresh=threshold)

# # 5. Sanity checks
# df = df.filter(
#     (col("age_youngest").isNull() | col("age_oldest").isNull() | col("age_oldest").isin([-1]) | col("age_youngest").isin([-1]) | (col("age_youngest") <= col("age_oldest"))) &
#     (col("state").isNull() | col("state").isin([-1]) | (length(col("state")) == 2)) &
#     (col("record_type").isNull() | col("record_type").isin([0, 1, -1])) &
#     (col("day").isNull() | col("day").isin([0, 1, 2, 3, 4, 5, 6, -1])) &
#     (col("homeowner").isNull() | col("homeowner").isin([0, 1, -1])) &
#     (col("car_value").isNull() | col("car_value").isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', '-1'])) &
#     (col("risk_factor").isNull() | col("risk_factor").isin([1, 2, 3, 4, -1])) &
#     (col("married_couple").isNull() | col("married_couple").isin([0, 1, -1])) &
#     (col("c_previous").isNull() | col("c_previous").isin([0, 1, 2, 3, 4, -1])) &
#     (col("A").isNull() | col("A").isin([0, 1, 2, 3, 4, -1])) &
#     (col("B").isNull() | col("B").isin([0, 1, 2, 3, 4, -1])) &
#     (col("C").isNull() | col("C").isin([0, 1, 2, 3, 4, -1])) &
#     (col("D").isNull() | col("D").isin([0, 1, 2, 3, 4, -1])) &
#     (col("E").isNull() | col("E").isin([0, 1, 2, 3, 4, -1])) &
#     (col("F").isNull() | col("F").isin([0, 1, 2, 3, 4, -1])) &
#     (col("G").isNull() | col("G").isin([0, 1, 2, 3, 4, -1]))
# )

# # Show the transformed, sanity-checked, and cleaned data (optional, for debugging purposes)
# df.show()

# Define Snowflake connection options
sfOptions = {
    "sfURL": snowflake_URL,
    "sfUser": snowflake_username,
    "sfPassword": snowflake_password,
    "sfDatabase": args['SNOWFLAKE_DATABASE'],
    "sfSchema": args['SNOWFLAKE_SCHEMA'],
    "sfWarehouse": args['SNOWFLAKE_WAREHOUSE']
}

# Write transformed, sanity-checked, and cleaned data to Snowflake
df.write \
    .format("net.snowflake.spark.snowflake") \
    .options(**sfOptions) \
    .option("dbtable", args['SNOWFLAKE_TABLE']) \
    .mode("append") \
    .save()

# Commit the job
job.commit()

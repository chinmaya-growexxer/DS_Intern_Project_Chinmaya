from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType
from datetime import datetime
import threading
import time
import io
import pyarrow as pa
import pyarrow.parquet as pq
import boto3  # Import the boto3 library for AWS

from AWS_Credentials import aws_access_key_id, aws_secret_access_key, aws_region, bucket_name, raw_data_folder

# Initialize the S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Define buffer and other parameters
buffer = []
buffer_lock = threading.Lock()
flush_interval = 1000  # Flush interval in seconds
#buffer_size = 10000  # Buffer size limit

# Function to write buffer to AWS S3 in Parquet format
def flush_buffer_to_s3():
    if not buffer:
        return
    print("Flushing buffer to S3...")
    parquet_buffer = io.BytesIO()
    table = pa.Table.from_pydict({key: [row[key] for row in buffer] for key in buffer[0].keys()})
    pq.write_table(table, parquet_buffer)
    parquet_buffer.seek(0)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{raw_data_folder}{current_time}_file.parquet"

    s3_client.upload_fileobj(parquet_buffer, bucket_name, s3_key)
    buffer.clear()

# Function to periodically flush the buffer
def periodic_flush():
    while True:
        time.sleep(flush_interval)
        with buffer_lock:
            flush_buffer_to_s3()

# Start the periodic flush in a separate thread
flush_thread = threading.Thread(target=periodic_flush, daemon=True)
flush_thread.start()

# Function to process each micro-batch
def foreach_batch_function(df, epoch_id):
    rows = df.collect()
    print(f"Processing {len(rows)} rows in micro-batch {epoch_id}")
    for row in rows:
        with buffer_lock:
            buffer.append(row.asDict())
            print(f"Buffer length after appending: {len(buffer)}")

def main():
    # Path to downloaded JARs
    jars_path = "/home/growlt240/Downloads/hadoop-aws-3.3.4.jar,/home/growlt240/Downloads/aws-java-sdk-bundle-1.11.1026.jar"
    
    spark = SparkSession.builder.appName("LiveProject")\
        .config("spark.jars", jars_path)\
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")\
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')

    # Define the schema with StringType for all fields
    schema = StructType([
        StructField("customer_ID", StringType(), True),
        StructField("shopping_pt", StringType(), True),
        StructField("record_type", StringType(), True),
        StructField("day", StringType(), True),
        StructField("time", StringType(), True),
        StructField("state", StringType(), True),
        StructField("location", StringType(), True),
        StructField("group_size", StringType(), True),
        StructField("homeowner", StringType(), True),
        StructField("car_age", StringType(), True),
        StructField("car_value", StringType(), True),
        StructField("risk_factor", StringType(), True),
        StructField("age_oldest", StringType(), True),
        StructField("age_youngest", StringType(), True),
        StructField("married_couple", StringType(), True),
        StructField("C_previous", StringType(), True),
        StructField("duration_previous", StringType(), True),
        StructField("A", StringType(), True),
        StructField("B", StringType(), True),
        StructField("C", StringType(), True),
        StructField("D", StringType(), True),
        StructField("E", StringType(), True),
        StructField("F", StringType(), True),
        StructField("G", StringType(), True),
        StructField("cost", StringType(), True)
    ])

    def read_kafka_topic(topic, schema):
        return (spark.readStream
                .format('kafka')
                .option('kafka.bootstrap.servers', 'localhost:9092')
                .option('subscribe', topic)
                .option('startingOffsets', 'latest')
                .load()
                .selectExpr('CAST(value AS STRING)')
                .select(from_json(col('value'), schema).alias('data'))
                .select('data.*')
                )

    insuranceDF = read_kafka_topic('Stream', schema).alias('data')

    # Apply the function to each micro-batch
    query = insuranceDF.writeStream.foreachBatch(foreach_batch_function).start()
    
    # Wait for the first batch to complete
    query.awaitTermination(7500)

if __name__ == "__main__":
    main()

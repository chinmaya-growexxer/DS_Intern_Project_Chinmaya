import csv
import time
from confluent_kafka import SerializingProducer
import uuid
import json

# Kafka producer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'Stream'
}

# Create Producer instance
producer = SerializingProducer(conf)

def json_serializer(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

# Function to handle delivery reports
def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

# Path to your CSV file
csv_file = 'data/Amended_Insurance Product Purchase_dataset.csv'
# Read CSV file and extract headers
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)  # Read the first row as headers
    count = 0
    # Read remaining rows and send each as a JSON message to Kafka
    for row in csv_reader:
        # Create a dictionary with headers as keys and corresponding row values
        message_dict = {header: value for header, value in zip(headers, row)}
        count +=1
        # Convert message dictionary to JSON string
        json_data = json.dumps(message_dict, default=json_serializer)

        # Log the JSON message before sending to Kafka
        print(f'Producing JSON message: {json_data}')
        print(f'{count} row send')
        # Produce to Kafka with key as the headers (converted to bytes)
        producer.produce(topic='Stream', key=str.encode(','.join(headers)), value=json_data.encode('utf-8'),
                         on_delivery=delivery_report)
        producer.poll(0)  # Introduce a small delay
        time.sleep(0.01)    # Introduce a larger delay between messages (adjust as needed)

producer.flush()

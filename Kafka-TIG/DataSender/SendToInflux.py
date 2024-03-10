import time
import csv
from kafka import KafkaProducer
import json
from datetime import datetime

# List of known anomalous timestamps
anomalous_timestamps = [
    "2014-04-15 15:44:00",
    "2014-04-16 03:34:00"
]

def convert_to_line_protocol(timestamp, value):
    # Convert timestamp to nanoseconds since epoch
    # Assuming your timestamps are in UTC and in the format "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    timestamp_ns = int(dt.timestamp() * 1e9)  # Convert to nanoseconds
    
    # Check if the timestamp is in the list of known anomalous timestamps
    is_anomaly = "true" if timestamp in anomalous_timestamps else "false"
    
    # Format the data as InfluxDB Line Protocol including the is_anomaly tag
    line = f"cpu_utilization,is_anomaly={is_anomaly} value={value} {timestamp_ns}"
    return line

def send_csv_data(producer, topic, file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            line_protocol = convert_to_line_protocol(row['timestamp'], row['value'])
            # Send the Line Protocol formatted data to Kafka
            producer.send(topic, value=line_protocol.encode('utf-8'))
            producer.flush()
            print(f"Sent data: {line_protocol}")
            time.sleep(0)  # Simulate delay

if __name__ == "__main__":
    bootstrap_servers = "localhost:9093" # Kafka broker address
    topic = "cpu_util" # Topic to send CPU utilization data
    file_path = "ec2_cpu_utilization_825cc2.csv" # Path to CSV file

    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v, # Use default serialization
        api_version=(2, 0, 2)
    )

    try:
        send_csv_data(producer, topic, file_path)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    finally:
        print("Closing Kafka producer...")
        producer.close()

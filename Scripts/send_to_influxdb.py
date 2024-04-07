import time
import csv
from kafka import KafkaProducer
from datetime import datetime, timezone


#List of known anomalous timestamps for CC2
#anomalous_timestamps_CC2 = [
#    "2014-04-15 15:44:00",
#    "2014-04-16 03:34:00"
#]


# List of known anomalous timestamps for B3B
#anomalous_timestamps_B3B = [
#    "2014-04-13 06:52:00",
#    "2014-04-18 23:27:00"
#]

dataset = "duplicated_cc2"
anomaly_name = "duplicated_anomalous_cc2"


def convert_to_line_protocol(timestamp, value):
    # Convert timestamp to nanoseconds since epoch
    # Assuming your timestamps are in UTC and in the format "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    timestamp_ns = int(dt.timestamp() * 1e9)  # Convert to nanoseconds
    
    # Format the data as InfluxDB Line Protocol
    line = f"{dataset} value={value} {timestamp_ns}"
    lines = [line]
    
    # If the timestamp is an anomaly, create a duplicate entry in a different measurement
    if timestamp in {anomaly_name}:
        anomaly_line = f"{anomaly_name} value={value} {timestamp_ns}"
        lines.append(anomaly_line)
    
    return lines

def send_csv_data(producer, topic, file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            line_protocols = convert_to_line_protocol(row['timestamp'], row['value'])
            for line_protocol in line_protocols:
                # Send the Line Protocol formatted data to Kafka
                producer.send(topic, value=line_protocol.encode('utf-8'))
                producer.flush()
                print(f"Sent data: {line_protocol}")
            time.sleep(0)  # Simulate delay

if __name__ == "__main__":
    bootstrap_servers = "localhost:9093" # Kafka broker address
    topic = "cpu_util" # Topic to send CPU utilization data
    file_path = "./datasets/enlarged_ec2_cpu_utilization_825cc2.csv" # Path to CSV file
    #file_path = "./datasets/enlarged_rds_cpu_utilization_e47b3b.csv"

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

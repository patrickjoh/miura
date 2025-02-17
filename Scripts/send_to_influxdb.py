import time
import csv
from kafka import KafkaProducer
from datetime import datetime, timezone

def read_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file]

# To send dataset to InfluxDB without anomalies
labels_EMPTY   = []

# List of labeled anomalies for rds_cpu_utilization_e47b3b.csv
labels_B3B     = read_from_file("../Datasets/Labels/labels_B3B.txt")
# List of labeled anomalies for ec2_cpu_utilization_825cc2.csv
labels_CC2     = read_from_file("../Datasets/Labels/labels_CC2.txt")
labels_CC2_seq = read_from_file("../Datasets/Labels/labels_CC2_seq.txt")
# List of labeled anomalies for C6H6_GT.csv
labels_C6H6_seq = read_from_file("../Datasets/Labels/labels_C6H6_seq.txt")
# List of labeled anomalies for PT08_S1_CO.csv
labels_PT08_S1_seq = read_from_file("../Datasets/Labels/labels_PT08_S1_seq.txt")
# List of labeled anomalies for PT08_S2_CO.csv
labels_PT08_S2_seq = read_from_file("../Datasets/Labels/labels_PT08_S2_seq.txt")
# List of labeled anomalies for Temperature.csv
labels_TEMP_seq = read_from_file("../Datasets/Labels/labels_TEMP_seq.txt")


# Choose the dataset and labels to use
dataset = "CC2_org" # Name of the measurement in InfluxDB
anomaly_name = "labels_CC2" # Name of the measurement for anomalies in InfluxDB
labels = labels_CC2         # Which labels to use


def convert_to_line_protocol(timestamp, value):
    # Convert timestamp to nanoseconds since epoch
    # Assuming your timestamps are in UTC and in the format "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    timestamp_ns = int(dt.timestamp() * 1e9)  # Convert to nanoseconds
    
    # Format the data as InfluxDB Line Protocol
    line = f"{dataset} value={value} {timestamp_ns}"
    lines = [line]
    
    # If the timestamp is an anomaly, create a duplicate entry in a different measurement
    if timestamp in labels:
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
    #file_path = "../Datasets/rds_cpu_utilization_e47b3b.csv"  # B3B 
    file_path = "../Datasets/ec2_cpu_utilization_825cc2.csv"  # CC2
    #file_path = "../Datasets/C6H6_GT.csv"                     # C6H6
    #file_path = "../Datasets/C6H6_GT_org.csv"                 # C6H6 org
    #file_path = "../Datasets/PT08_S1_CO.csv"                  # PT08_S1
    #file_path = "../Datasets/PT08_S1_CO_org.csv"              # PT08_S1 org
    #file_path = "../Datasets/PT08_S2_NMHC.csv"                # PT08_S2
    #file_path = "../Datasets/PT08_S2_NMHC_org.csv"            # PT08_S2 org
    #file_path = "../Datasets/Temperature.csv"                 # Temperature
    #file_path = "../Datasets/Temperature_org.csv"             # Temperature org



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

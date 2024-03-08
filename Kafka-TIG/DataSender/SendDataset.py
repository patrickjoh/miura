import time
import csv
from kafka import KafkaProducer
import json

def send_csv_data(producer, topic, file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            # Assuming `row` is a dictionary with 'timestamp' and 'value' keys
            temperature_payload = {"cpu_util": float(row['value'])}

            # Serialize the payload to a JSON string
            data = json.dumps(temperature_payload)

            # Send this JSON string as the message value to Kafka
            producer.send(topic, value=data.encode("utf-8"))
            producer.flush()

            print(f"Sent data: {row}")

            # Wait for 5 seconds to simulate delay
            time.sleep(5)

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
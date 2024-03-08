import time
from kafka import KafkaProducer
import json


def send_sensor_data(producer, temperature_topic, pressure_topic):
    temperature = 25.0  # Initial temperature
    pressure = 100.0  # Initial pressure

    while True:
        # Generate temperature data
        temperature += 0.5

        # Create JSON payload for temperature
        temperature_payload = {"temperature": temperature}

        # Serialize temperature payload to JSON
        temperature_data = json.dumps(temperature_payload)

        # Send temperature data to temperature topic
        producer.send(temperature_topic,
                      value=temperature_data.encode("utf-8"))
        producer.flush()

        # Generate pressure data
        pressure += 1.0

        # Create JSON payload for pressure
        pressure_payload = {"pressure": pressure}

        # Serialize pressure payload to JSON
        pressure_data = json.dumps(pressure_payload)

        # Send pressure data to pressure topic
        producer.send(pressure_topic, value=pressure_data.encode("utf-8"))
        producer.flush()

        print(f"Sent temperature data: {temperature}")
        print(f"Sent pressure data: {pressure}")

        time.sleep(1)  # Wait for 5 seconds


if __name__ == "__main__":
    bootstrap_servers = "localhost:9093"  # Kafka broker address
    temperature_topic = "states1"  # Topic to send temperature data
    pressure_topic = "states2"  # Topic to send pressure data

    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        # Use default serialization
        value_serializer=lambda v: v,
        api_version=(2, 0, 2)
    )

    try:
        send_sensor_data(producer, temperature_topic, pressure_topic)
    except KeyboardInterrupt:
        pass
    finally:
        producer.close()

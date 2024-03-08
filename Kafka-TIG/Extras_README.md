Notes:

Repository that was used as a starting point for our implementation:
[/eternalmit5/Kafka-TIG](https://github.com/eternalamit5/Kafka-TIG)

To connect Grafa to InfluxDB, you need to add a data source in Grafana.
Query language: Flux

Find the IP-address of the InfluxDB container by inspecting the network
of the container


Initialized in the `variables.env` file

```bash
InfluxDB Username
InfluxDB Password
InfluxDB ORG_name
InfluxDB Bucket_name
InfluxDB Token
```

SendDataset.py is a Python script that sends the datapoints specified in a csv files 
of format 'timestamp', 'value' (Check the file ec2_cpu_utilization_825cc2.csv for reference)

The script may need to be changed to work with other datasets.




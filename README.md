# Introduction

This repository was produced as part of the Bachelor's Thesis of:
- Patrick Johannessen
- Magnus Johannessen

The repostitory contains all code used in the production of results related to MIURA, including:
- Models
- Datasets; with labels
- Docker-infrastructure
- Scripts for:
    - Sending of data to InfluxDB
    - Various method for evalution of the models

The last commit performed during the writing of said report was on *21st of May 2024*

Link to report: [MIURA: Memory-efficient and Incrementally learning Unsupervised Real-time Anomaly detection for time series data](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3141337)

# Setup

## Python environment

- Setup Python-environment with version Python=3.11.7

## Docker-stack

See Kafka-TIG-README.md and Extras_README.md in '/Kafka-TIG'

## /Scripts

- Install required libraries: `pip install -r requirements.txt`

### Script: 'send_to_influxdb.py'

Uses both the timestamp and value from the dataset, ensuring that timing of data is not dependant on when it arrived at the database.

The following must/should be changed based on use:
- 'dataset'
- 'anomaly_name'
- 'labels'
- `time.sleep(X)` at the end of function 'send_csv_data()' - Adds delay to the sending of data; practical for testing purposes.
- 'file_path' - Easiest to comment out any datasets other than the one you wish to send.

## /Models

- Install required libraries: `pip install -r requirements.txt`


# LICENSE

Copyright (c) 2024 Patrick Johannessen \
Copyright (c) 2024 Magnus Johannessen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
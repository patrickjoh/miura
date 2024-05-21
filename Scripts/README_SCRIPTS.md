# Description for each script

## "calculate_f1.ipynb"

Used to automatically calculate the F1-score of the model on a set dataset.
This does not take sequential anomalies into consideration and will consider any labels not directly detected as a false negative.
If a sequential anomaly is labeled, every single label will be counted as a false negative if not detected.

## "calculate_f1_seq.ipynb"

This script does the same as the "calculate_f1.ipynb", but does take sequential anomalies into consideration.
The model needs only detected a single datapoint within the whole sequential anomaly in order for the whole period to be marked as detected.
Seqential anomalies are also considered a single anomaly, no matter how many datapoints it is made up of.


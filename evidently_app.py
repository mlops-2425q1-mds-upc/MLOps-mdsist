"""
Evidently script for computing metrics and exposing them so prometheus can collect them
"""

import time
import os
import pandas as pd
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from prometheus_client import Gauge, start_http_server

from config import MONITORING_DIR

# Define metrics to export
tl_drift_gauge = Gauge("true_label_drift_score", "True Label Drift Metric")
pl_drift_gauge = Gauge("predicted_label_drift_score", "Predicted Label Drift Metric")
cur_accuracy_gauge = Gauge("current_model_accuracy", "Current Model Accuracy")
ref_accuracy_gauge = Gauge("reference_model_accuracy", "Reference Model Accuracy")

# Export metrics to Prometheus
start_http_server(8100)  # Prometheus scraper will pull from this port
while True:

    reference_data_path = f"{MONITORING_DIR}/reference_data.csv"
    current_data_path = f"{MONITORING_DIR}/current_data.csv"

    # Check that there is data to report
    if not os.path.exists(reference_data_path) or not os.path.exists(current_data_path):
        time.sleep(60)
        continue

    # Load your data
    reference_data = pd.read_csv(reference_data_path)[
        ["true_label", "predicted_label"]
    ]
    current_data = pd.read_csv(current_data_path)[["true_label", "predicted_label"]]

    # Map columns (adjust for your dataset)
    column_mapping = ColumnMapping(target="true_label", prediction="predicted_label")

    # Set up Evidently report with specific metrics
    data_drift_report = Report(metrics=[DataDriftPreset()])
    classification_report = Report(metrics=[ClassificationPreset()])

    # Generate reports
    data_drift_report.run(
        reference_data=reference_data, current_data=current_data, column_mapping=column_mapping
    )
    classification_report.run(
        reference_data=reference_data, current_data=current_data, column_mapping=column_mapping
    )

    # Metrics extraction
    tl_drift_score = data_drift_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][
        "true_label"
    ]["drift_score"]
    pl_drift_score = data_drift_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][
        "predicted_label"
    ]["drift_score"]
    cur_accuracy = classification_report.as_dict()["metrics"][0]["result"]["current"]["accuracy"]
    ref_accuracy = classification_report.as_dict()["metrics"][0]["result"]["reference"]["accuracy"]

    print("updating metrics...")
    tl_drift_gauge.set(tl_drift_score)
    pl_drift_gauge.set(pl_drift_score)
    cur_accuracy_gauge.set(str(cur_accuracy))
    ref_accuracy_gauge.set(str(ref_accuracy))
    print("Metrics updated!")
    time.sleep(10)  # Update metrics every 10 seconds

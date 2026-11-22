
import os
import csv
import time
import threading
from pathlib import Path

import psutil
from ultralytics import YOLO

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "dataset" / "data.yaml"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

CSV_FILE = REPORT_DIR / "resource_usage.csv"

# Monitoring controls
stop_monitoring = threading.Event()


def monitor_resources():
    """Collect CPU and RAM measurements for MCOA analysis."""

    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "CPU_Usage_Percent",
            "RAM_Usage_Percent",
            "RAM_Used_MB"
        ])

        psutil.cpu_percent(interval=None)

        while not stop_monitoring.wait(1):
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                cpu,
                memory.percent,
                round(memory.used / (1024 ** 2), 2)
            ])

            file.flush()


def train_model():

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration missing: {DATA_YAML}")

    print("\nLoading YOLOv8n model...")

    model = YOLO("yolov8n.pt")

    print("Starting CPU and RAM monitoring...")

    monitor_thread = threading.Thread(
        target=monitor_resources,
        daemon=True
    )

    monitor_thread.start()

    try:
        print("\nStarting road damage detection training...\n")

        results = model.train(
            data=str(DATA_YAML),
            epochs=30,
            imgsz=416,
            batch=4,
            device="cpu",
            workers=0,
            project=str(PROJECT_DIR / "runs" / "road_damage"),
            name="yolov8n_cpu",
            exist_ok=True,
            plots=True,
            save=True,
            cache=False,
            verbose=True
        )

        print("\nTRAINING COMPLETED SUCCESSFULLY!")

        print("Best model location:")
        print(
            PROJECT_DIR /
            "runs" / "road_damage" /
            "yolov8n_cpu" / "weights" / "best.pt"
        )

    finally:
        stop_monitoring.set()
        monitor_thread.join()

        print("\nCPU and RAM monitoring stopped.")
        print(f"Resource report saved: {CSV_FILE}")


if __name__ == "__main__":
    train_model()

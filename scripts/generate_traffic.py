"""Generate realistic traffic against the API so Prometheus/Grafana dashboards have data.

    python scripts/generate_traffic.py --url http://127.0.0.1:8000 --requests 300
Samples rows from the cleaned dataset; ~5% of requests are deliberately invalid (to show 422s).
"""

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "heart_disease_clean.csv"
INT_FIELDS = {"age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def load_rows():
    with DATA.open() as handle:
        rows = []
        for row in csv.DictReader(handle):
            row.pop("target", None)
            rows.append({k: (None if v == "" else (int(float(v)) if k in INT_FIELDS else float(v)))
                         for k, v in row.items()})
        return rows


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as err:
        return err.code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    rows, statuses = load_rows(), {}
    for i in range(args.requests):
        payload = dict(random.choice(rows))
        if random.random() < 0.05:
            payload["age"] = 999
        status = post(f"{args.url.rstrip('/')}/predict", payload)
        statuses[status] = statuses.get(status, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"sent {i + 1}/{args.requests} -> {statuses}")
        time.sleep(args.delay)
    print(f"done: {statuses}")


if __name__ == "__main__":
    main()

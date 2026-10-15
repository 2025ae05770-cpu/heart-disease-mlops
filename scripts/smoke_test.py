"""End-to-end smoke test of a running API (local, Docker, or Kubernetes). Stdlib only.

    python scripts/smoke_test.py --url http://127.0.0.1:8000
Exits non-zero on any failure so it can gate CI.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLE = json.loads((Path(__file__).parent / "sample_request.json").read_text())
HEALTHY_PATIENT = {**SAMPLE, "age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
                   "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3}
HIGH_RISK_PATIENT = {**SAMPLE, "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
                     "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3}


def call(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode()
            return response.status, body
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def wait_until_healthy(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _ = call("GET", f"{base}/health")
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=int, default=60, help="seconds to wait for /health")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    if not wait_until_healthy(base, args.wait):
        print(f"FAIL: {base}/health not healthy within {args.wait}s")
        return 1

    checks = []

    status, body = call("GET", f"{base}/health")
    checks.append(("GET /health -> 200", status == 200, body))

    for name, patient in [("sample", SAMPLE), ("healthy", HEALTHY_PATIENT), ("high-risk", HIGH_RISK_PATIENT)]:
        status, body = call("POST", f"{base}/predict", patient)
        ok = status == 200 and {"prediction", "confidence"} <= set(json.loads(body))
        checks.append((f"POST /predict ({name}) -> 200 + prediction/confidence", ok, body))

    status, body = call("POST", f"{base}/predict", {**SAMPLE, "age": 500})
    checks.append(("POST /predict invalid -> 422", status == 422, body[:120]))

    status, body = call("POST", f"{base}/predict/batch", {"records": [SAMPLE, HIGH_RISK_PATIENT]})
    checks.append(("POST /predict/batch -> 200", status == 200 and json.loads(body)["count"] == 2, body[:200]))

    status, body = call("GET", f"{base}/metrics")
    checks.append(("GET /metrics exposes counters", status == 200 and "heart_api_requests_total" in body, ""))

    failed = 0
    for name, ok, body in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if body:
            print(f"        {body}")
        failed += not ok
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

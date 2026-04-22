"""
Sends out-of-distribution text to force drift score up → triggers Grafana alert.
Usage: python scripts/simulate_drift.py
"""
import random
import string
import time

import requests

URL = "http://localhost:8000/predict"


def random_gibberish(n=200):
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(n // 6)
    )


def main():
    print("Sending 100 OOD gibberish payloads to inflate drift score...")
    for i in range(100):
        try:
            r = requests.post(URL, json={"text": random_gibberish()}, timeout=10)
            if i % 10 == 0:
                print(f"  {i}: drift={r.json().get('drift_score', 0):.3f}")
        except Exception as e:
            print(f"  {i}: ERROR {e}")
        time.sleep(0.1)
    print("Done. Check Grafana + AlertManager.")


if __name__ == "__main__":
    main()

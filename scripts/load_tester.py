"""
Load tester: sends 100 concurrent POSTs and tallies which container_id handled each.
Usage: python scripts/load_tester.py
"""
import concurrent.futures
from collections import Counter

import requests

URL = "http://localhost:8000/predict"
PAYLOAD = {"text": "I've been feeling really anxious lately and can't sleep at night."}


def call():
    try:
        r = requests.post(URL, json=PAYLOAD, timeout=10)
        return r.json().get("container_id", "unknown")
    except Exception as e:
        return f"FAILED:{type(e).__name__}"


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(lambda _: call(), range(100)))
    counts = Counter(results)
    print("\nDistribution across containers:")
    for cid, n in counts.most_common():
        bar = "█" * n
        print(f"  {cid:<20} {n:>3}  {bar}")
    print(f"\nTotal: {sum(counts.values())} requests, "
          f"{len([k for k in counts if not k.startswith('FAILED')])} distinct containers")


if __name__ == "__main__":
    main()

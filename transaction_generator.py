"""
FraudBOT AI — Layer 1, Step 1
Generates realistic financial transaction logs
with normal and fraudulent patterns
"""

import random
import json
from datetime import datetime, timedelta

MERCHANTS = [
    "Amazon", "Walmart", "Target", "Starbucks",
    "Shell Gas", "Netflix", "Apple Store", "Uber"
]

FRAUD_MERCHANTS = [
    "Unknown Vendor 4821", "Overseas Transfer XY",
    "Crypto Exchange 99", "Wire Transfer INTL"
]

LOCATIONS = ["San Francisco CA", "Oakland CA", "San Jose CA", "Fremont CA"]
FRAUD_LOCATIONS = ["L1 NG", "BaT1 RO", "Unknown", "M1 RU"]

def generate_transactions(n=50):
    transactions = []
    base_time = datetime.now() - timedelta(hours=6)
    account_id = "ACC-78234"

    for i in range(n):
        # 20% chance of fraudulent transaction
        is_fraud = random.random() < 0.2

        if is_fraud:
            fraud_type = random.choice([
                "unusual_location",
                "high_amount",
                "rapid_succession",
                "suspicious_merchant"
            ])

            txn = {
                "txn_id":      f"TXN-{1000 + i}",
                "timestamp":   (base_time + timedelta(minutes=i*7)).isoformat(),
                "account_id":  account_id,
                "amount":      round(random.uniform(500, 5000), 2),
                "merchant":    random.choice(FRAUD_MERCHANTS),
                "location":    random.choice(FRAUD_LOCATIONS),
                "category":    "suspicious",
                "fraud_type":  fraud_type,
                "label":       "FRAUD"
            }
        else:
            txn = {
                "txn_id":      f"TXN-{1000 + i}",
                "timestamp":   (base_time + timedelta(minutes=i*7)).isoformat(),
                "account_id":  account_id,
                "amount":      round(random.uniform(5, 200), 2),
                "merchant":    random.choice(MERCHANTS),
                "location":    random.choice(LOCATIONS),
                "category":    "normal",
                "fraud_type":  None,
                "label":       "NORMAL"
            }

        transactions.append(txn)

    return transactions

if __name__ == "__main__":
    transactions = generate_transactions(50)

    with open("transactions.json", "w") as f:
        json.dump(transactions, f, indent=2)

    fraud = [t for t in transactions if t["label"] == "FRAUD"]
    normal = [t for t in transactions if t["label"] == "NORMAL"]

    print(f"Generated {len(transactions)} transactions")
    print(f"  FRAUD:  {len(fraud)}")
    print(f"  NORMAL: {len(normal)}")
    print(f"\nSample FRAUD transaction:")
    print(json.dumps(fraud[0], indent=2))
    print(f"\nSample NORMAL transaction:")
    print(json.dumps(normal[0], indent=2))
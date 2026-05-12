"""
FraudBot AI — Layer 1, Step 1
Transaction log generator with behavior-based fraud signals.
No location-based or demographic bias. 
Fraud is detected by WHAT the user does, not WHERE they are from.
"""

import random
import json
from datetime import datetime, timedelta

# --- Merchants by category ---
MERCHANTS = {
    "groceries":     ["Whole Foods", "Safeway", "Trader Joes", "Costco"],
    "dining":        ["Starbucks", "Chipotle", "Panera Bread", "McDonalds"],
    "retail":        ["Amazon", "Target", "Walmart", "Best Buy"],
    "transport":     ["Uber", "Lyft", "Shell Gas", "Chevron"],
    "subscriptions": ["Netflix", "Spotify", "Apple One", "YouTube Premium"],
}

# Flatten merchant list for normal use
ALL_NORMAL_MERCHANTS = [m for group in MERCHANTS.values() for m in group]

# Suspicious merchant types (no real names, no country bias)
SUSPICIOUS_MERCHANTS = [
    "Unverified Vendor #4821",
    "Unnamed Digital Services",
    "Unregistered Transfer Co.",
    "Unknown Marketplace #77",
]

# Behavioral fraud signals — based on actions, not demographics
FRAUD_SIGNALS = [
    "velocity_spike",      # multiple transactions in very short time
    "new_device",          # transaction from a device never seen before
    "amount_anomaly",      # amount far above normal spending pattern
    "odd_hours",           # transaction at unusual hours (1am - 4am)
    "merchant_mismatch",   # merchant category never used before by this account
]

DEVICE_TYPES = ["iPhone 14", "MacBook Pro", "iPad", "Chrome Browser"]
UNKNOWN_DEVICES = ["Unknown Device", "New Android", "Unrecognized Browser"]


def make_normal_transaction(txn_id, timestamp, account_id):
    """Create a normal, low-risk transaction."""
    return {
        "txn_id":         txn_id,
        "timestamp":      timestamp.isoformat(),
        "account_id":     account_id,
        "amount":         round(random.uniform(5.0, 180.0), 2),
        "merchant":       random.choice(ALL_NORMAL_MERCHANTS),
        "merchant_cat":   random.choice(list(MERCHANTS.keys())),
        "device":         random.choice(DEVICE_TYPES),
        "hour_of_day":    timestamp.hour,
        "velocity":       "normal",       # spacing between transactions
        "is_new_device":  False,
        "fraud_signal":   None,
        "label":          "NORMAL"
    }


def make_fraud_transaction(txn_id, timestamp, account_id):
    """Create a fraud transaction based on behavioral anomaly — no location bias."""
    signal = random.choice(FRAUD_SIGNALS)

    # Adjust transaction properties based on the fraud signal type
    if signal == "velocity_spike":
        # Rapid back-to-back transactions
        amount = round(random.uniform(50, 300), 2)
        merchant = random.choice(ALL_NORMAL_MERCHANTS)
        device = random.choice(DEVICE_TYPES)
        velocity = "rapid"
        is_new_device = False

    elif signal == "new_device":
        # Transaction from an unrecognized device
        amount = round(random.uniform(100, 800), 2)
        merchant = random.choice(ALL_NORMAL_MERCHANTS)
        device = random.choice(UNKNOWN_DEVICES)
        velocity = "normal"
        is_new_device = True

    elif signal == "amount_anomaly":
        # Amount far above normal (10x typical spend)
        amount = round(random.uniform(1500, 5000), 2)
        merchant = random.choice(SUSPICIOUS_MERCHANTS)
        device = random.choice(DEVICE_TYPES)
        velocity = "normal"
        is_new_device = False

    elif signal == "odd_hours":
        # Force timestamp to 1am–4am window
        odd_hour = random.randint(1, 4)
        timestamp = timestamp.replace(hour=odd_hour, minute=random.randint(0, 59))
        amount = round(random.uniform(200, 1200), 2)
        merchant = random.choice(SUSPICIOUS_MERCHANTS)
        device = random.choice(UNKNOWN_DEVICES)
        velocity = "normal"
        is_new_device = True

    else:  # merchant_mismatch
        # Merchant category never used before
        amount = round(random.uniform(300, 2000), 2)
        merchant = random.choice(SUSPICIOUS_MERCHANTS)
        device = random.choice(DEVICE_TYPES)
        velocity = "normal"
        is_new_device = False

    return {
        "txn_id":         txn_id,
        "timestamp":      timestamp.isoformat(),
        "account_id":     account_id,
        "amount":         amount,
        "merchant":       merchant,
        "merchant_cat":   "suspicious",
        "device":         device,
        "hour_of_day":    timestamp.hour,
        "velocity":       velocity,
        "is_new_device":  is_new_device,
        "fraud_signal":   signal,
        "label":          "FRAUD"
    }


def generate_transactions(n=50):
    transactions = []
    base_time = datetime.now() - timedelta(hours=6)
    account_id = "ACC-78234"

    for i in range(n):
        timestamp = base_time + timedelta(minutes=i * 7)
        txn_id = f"TXN-{1000 + i}"

        # 20% fraud rate
        if random.random() < 0.2:
            txn = make_fraud_transaction(txn_id, timestamp, account_id)
        else:
            txn = make_normal_transaction(txn_id, timestamp, account_id)

        transactions.append(txn)

    return transactions


if __name__ == "__main__":
    transactions = generate_transactions(50)

    # Save to file
    with open("transactions.json", "w") as f:
        json.dump(transactions, f, indent=2)

    fraud  = [t for t in transactions if t["label"] == "FRAUD"]
    normal = [t for t in transactions if t["label"] == "NORMAL"]

    print("=" * 55)
    print("FraudBot AI — Transaction Generator")
    print("=" * 55)
    print(f"Total transactions : {len(transactions)}")
    print(f"  NORMAL           : {len(normal)}")
    print(f"  FRAUD            : {len(fraud)}")

    print("\nFraud signals detected:")
    from collections import Counter
    signal_counts = Counter(t["fraud_signal"] for t in fraud)
    for signal, count in signal_counts.items():
        print(f"  {signal:<25} : {count}")

    print(f"\nSample FRAUD transaction:")
    print(json.dumps(fraud[0], indent=2))

    print(f"\nSample NORMAL transaction:")
    print(json.dumps(normal[0], indent=2))

    print("\nTransactions saved to transactions.json")
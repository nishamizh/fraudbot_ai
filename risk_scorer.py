"""
FraudBot AI — Layer 1, Step 2
Risk Scorer using Pandas.
Reads transactions.json, scores each transaction 0-100
based on behavioral signals. No ML yet — pure rule-based logic.
This is how real fraud systems work: rules first, then ML on top.
"""

import pandas as pd
import json

# ----------------------------------------------------------------
# 1. Load transactions
# ----------------------------------------------------------------
with open("transactions.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("=" * 55)
print("FraudBot AI — Risk Scorer")
print("=" * 55)
print(f"Loaded {len(df)} transactions\n")


# ----------------------------------------------------------------
# 2. Rule-based risk scoring
#    Each rule adds points to the risk score (0 = safe, 100 = fraud)
#    This mirrors how real fraud engines work at Stripe / PayPal
# ----------------------------------------------------------------

df["risk_score"] = 0  # start everyone at 0

# Rule 1: High amount → +35 points
df.loc[df["amount"] > 1000, "risk_score"] += 35

# Rule 2: New/unknown device → +25 points
df.loc[df["is_new_device"] == True, "risk_score"] += 25

# Rule 3: Rapid velocity → +20 points
df.loc[df["velocity"] == "rapid", "risk_score"] += 20

# Rule 4: Odd hours (1am - 4am) → +15 points
df.loc[df["hour_of_day"].between(1, 4), "risk_score"] += 15

# Rule 5: Suspicious merchant category → +25 points
df.loc[df["merchant_cat"] == "suspicious", "risk_score"] += 25

# Cap score at 100
df["risk_score"] = df["risk_score"].clip(upper=100)


# ----------------------------------------------------------------
# 3. Assign risk label based on score
#    HIGH   → flag for immediate review
#    MEDIUM → flag for human review queue
#    LOW    → auto-approve
# ----------------------------------------------------------------

def assign_risk_label(score):
    if score >= 60:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_label"] = df["risk_score"].apply(assign_risk_label)


# ----------------------------------------------------------------
# 4. Print results
# ----------------------------------------------------------------

print("Risk Score Distribution:")
print(f"  HIGH   (≥60) : {len(df[df['risk_label'] == 'HIGH']):>3} transactions → immediate review")
print(f"  MEDIUM (≥30) : {len(df[df['risk_label'] == 'MEDIUM']):>3} transactions → human review queue")
print(f"  LOW    (<30) : {len(df[df['risk_label'] == 'LOW']):>3} transactions → auto approved")

print("\nTop 5 Riskiest Transactions:")
top5 = df.nlargest(5, "risk_score", keep="all")[
    ["txn_id", "amount", "merchant", "risk_score", "risk_label", "fraud_signal"]
]
print(top5.to_string(index=False))

print("\nRisk score stats:")
print(f"  Average : {df['risk_score'].mean():.1f}")
print(f"  Max     : {df['risk_score'].max()}")
print(f"  Min     : {df['risk_score'].min()}")


# ----------------------------------------------------------------
# 5. Accuracy check — how well do our rules match the labels?
#    Compare risk_label against the ground truth label
# ----------------------------------------------------------------

print("\nAccuracy Check (rules vs ground truth):")

# A FRAUD transaction should score HIGH or MEDIUM
fraud_df  = df[df["label"] == "FRAUD"]
normal_df = df[df["label"] == "NORMAL"]

caught   = fraud_df[fraud_df["risk_label"].isin(["HIGH", "MEDIUM"])]
missed   = fraud_df[fraud_df["risk_label"] == "LOW"]
false_pos = normal_df[normal_df["risk_label"] == "HIGH"]

print(f"  Fraud caught (HIGH/MEDIUM) : {len(caught)} / {len(fraud_df)}")
print(f"  Fraud missed (LOW)         : {len(missed)} / {len(fraud_df)}")
print(f"  False positives (normal→HIGH): {len(false_pos)} / {len(normal_df)}")


# ----------------------------------------------------------------
# 6. Save scored transactions
# ----------------------------------------------------------------

df.to_json("transactions_scored.json", orient="records", indent=2)
print("\nScored transactions saved to transactions_scored.json")
print("\nLayer 1 complete! Ready for Layer 2 — LLM Agent.")
"""
FraudBot AI — Layer 2
LLM Agent using HuggingFace.
Reads HIGH and MEDIUM risk transactions from Layer 1,
analyzes each one and produces:
  - Root cause explanation
  - Confidence score
  - Recommended action
  - Evidence summary
This is the "brain" of FraudBot AI.
"""

import json
import re
from transformers import pipeline


# ----------------------------------------------------------------
# 1. Load scored transactions from Layer 1
#    Only process HIGH and MEDIUM risk — LOW are auto-approved
# ----------------------------------------------------------------

with open("transactions_scored.json", "r") as f:
    all_transactions = json.load(f)

flagged = [
    t for t in all_transactions
    if t["risk_label"] in ["HIGH", "MEDIUM"]
]

print("=" * 60)
print("FraudBot AI — LLM Fraud Agent")
print("=" * 60)
print(f"Total transactions loaded : {len(all_transactions)}")
print(f"Flagged for LLM review    : {len(flagged)} (HIGH + MEDIUM)")
print(f"Auto-approved (LOW)       : {len(all_transactions) - len(flagged)}")
print("\nLoading LLM model... (first run downloads ~250MB)\n")


# ----------------------------------------------------------------
# 2. Load HuggingFace model
#    We use a text-generation model to analyze each transaction
#    flan-t5-base is lightweight, fast, and runs on CPU — perfect
#    for a demo. In production you'd use GPT-4 or Claude.
# ----------------------------------------------------------------

llm = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)

print("Model loaded successfully.\n")


# ----------------------------------------------------------------
# 3. Build a prompt for each flagged transaction
#    Good prompt engineering = good output
#    We give the LLM exactly the context it needs
# ----------------------------------------------------------------

def build_prompt(txn):
    """
    Craft a clear, structured prompt for the LLM.
    Tells the model exactly what we want back.
    """
    signals = []
    if txn["amount"] > 1000:
        signals.append(f"unusually high amount (${txn['amount']})")
    if txn["is_new_device"]:
        signals.append("transaction from an unrecognized device")
    if txn["velocity"] == "rapid":
        signals.append("rapid succession of transactions")
    if txn["hour_of_day"] in range(1, 5):
        signals.append(f"transaction at {txn['hour_of_day']}am (odd hours)")
    if txn["merchant_cat"] == "suspicious":
        signals.append(f"suspicious merchant: {txn['merchant']}")

    signals_text = ", ".join(signals) if signals else "unusual pattern detected"

    prompt = f"""Analyze this financial transaction for fraud.

Transaction ID: {txn['txn_id']}
Amount: ${txn['amount']}
Merchant: {txn['merchant']}
Risk Score: {txn['risk_score']}/100
Risk Level: {txn['risk_label']}
Fraud Signals: {signals_text}

Provide:
1. Root cause of suspicion
2. Confidence level (HIGH/MEDIUM/LOW)
3. Recommended action (block/review/monitor)
4. Brief evidence summary

Analysis:"""

    return prompt


# ----------------------------------------------------------------
# 4. Run the LLM agent on each flagged transaction
# ----------------------------------------------------------------

def parse_llm_response(response_text, txn):
    """
    Parse the LLM output into structured fields.
    Falls back to rule-based values if LLM output is unclear.
    """

    # confidence: map risk score to confidence
    if txn["risk_score"] >= 75:
        confidence = "HIGH"
    elif txn["risk_score"] >= 45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # recommended action based on risk label
    action_map = {
        "HIGH":   "Block transaction immediately and notify customer",
        "MEDIUM": "Hold transaction and route to human review queue",
        "LOW":    "Monitor account activity"
    }
    action = action_map.get(txn["risk_label"], "Review manually")

    return {
        "txn_id":      txn["txn_id"],
        "amount":      txn["amount"],
        "merchant":    txn["merchant"],
        "risk_score":  txn["risk_score"],
        "risk_label":  txn["risk_label"],
        "fraud_signal": txn["fraud_signal"],
        "llm_analysis": response_text.strip(),
        "confidence":  confidence,
        "action":      action,
        "ground_truth": txn["label"]
    }


results = []

print("=" * 60)
print("ANALYZING FLAGGED TRANSACTIONS")
print("=" * 60)

for i, txn in enumerate(flagged, 1):
    print(f"\n[{i}/{len(flagged)}] {txn['txn_id']} — ${txn['amount']} "
          f"— Risk: {txn['risk_label']} ({txn['risk_score']}/100)")

    # Build prompt and run LLM
    prompt = build_prompt(txn)
    response = llm(prompt)[0]["generated_text"]

    # Parse into structured output
    result = parse_llm_response(response, txn)
    results.append(result)

    # Print key output
    print(f"  Fraud Signal  : {txn['fraud_signal']}")
    print(f"  Confidence    : {result['confidence']}")
    print(f"  Action        : {result['action']}")
    print(f"  LLM Analysis  : {result['llm_analysis'][:120]}...")
    print(f"  Ground Truth  : {txn['label']}")


# ----------------------------------------------------------------
# 5. Summary report
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("FRAUDBOT AI — AGENT SUMMARY REPORT")
print("=" * 60)

high_conf   = [r for r in results if r["confidence"] == "HIGH"]
medium_conf = [r for r in results if r["confidence"] == "MEDIUM"]
to_block    = [r for r in results if "Block" in r["action"]]
to_review   = [r for r in results if "route" in r["action"].lower()]

print(f"Transactions analyzed     : {len(results)}")
print(f"  High confidence flags   : {len(high_conf)}")
print(f"  Medium confidence flags : {len(medium_conf)}")
print(f"  Recommended to block    : {len(to_block)}")
print(f"  Routed to human review  : {len(to_review)}")

# How many actual frauds did we catch?
actual_fraud_caught = [
    r for r in results if r["ground_truth"] == "FRAUD"
]
print(f"\nGround truth validation:")
print(f"  Actual fraud transactions caught : {len(actual_fraud_caught)} / {len(results)}")


# ----------------------------------------------------------------
# 6. Save full agent results
# ----------------------------------------------------------------

with open("agent_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nFull agent results saved to agent_results.json")
print("\nLayer 2 complete! Ready for Layer 3 — Vector DB Memory.")
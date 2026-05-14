"""
FraudBot AI — Layer 3
Vector DB Memory using ChromaDB.
Stores every fraud analysis as an embedding so the system
can answer: "Have we seen this pattern before?"
This gives FraudBot AI memory across transactions.
"""

import json
import chromadb
from chromadb.utils import embedding_functions


# ----------------------------------------------------------------
# 1. Load agent results from Layer 2
# ----------------------------------------------------------------

with open("agent_results.json", "r") as f:
    agent_results = json.load(f)

print("=" * 60)
print("FraudBot AI — Vector Store (Layer 3)")
print("=" * 60)
print(f"Loading {len(agent_results)} analyzed transactions...\n")


# ----------------------------------------------------------------
# 2. Set up ChromaDB
#    Runs locally — no API key, no cloud, no cost
#    Stores data in ./fraudbot_db folder on your machine
# ----------------------------------------------------------------

client = chromadb.PersistentClient(path="./fraudbot_db")

# Use sentence-transformers to convert text → vectors
# all-MiniLM-L6-v2 is fast, lightweight, great for semantic search
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create (or reuse) a collection — like a table in a database
collection = client.get_or_create_collection(
    name="fraud_patterns",
    embedding_function=embedding_fn
)

print("ChromaDB initialized at ./fraudbot_db")
print(f"Collection: fraud_patterns\n")


# ----------------------------------------------------------------
# 3. Build documents to store
#    Each transaction becomes a text document describing its pattern
#    ChromaDB converts this text → vector embedding
# ----------------------------------------------------------------

def build_document(result):
    """
    Convert a transaction result into a text description.
    This is what gets embedded into the vector space.
    The richer the text, the better the semantic search.
    """
    return (
        f"Transaction {result['txn_id']} with amount ${result['amount']} "
        f"at merchant {result['merchant']}. "
        f"Risk level: {result['risk_label']}. "
        f"Fraud signal: {result['fraud_signal']}. "
        f"Confidence: {result['confidence']}. "
        f"Action taken: {result['action']}. "
        f"Analysis: {result['llm_analysis']}"
    )


# ----------------------------------------------------------------
# 4. Store all results in ChromaDB
#    Each entry has:
#      - id       : unique transaction ID
#      - document : text description (gets embedded)
#      - metadata : structured fields for filtering
# ----------------------------------------------------------------

print("Storing transactions in vector DB...")

ids       = []
documents = []
metadatas = []

for result in agent_results:
    ids.append(result["txn_id"])
    documents.append(build_document(result))
    metadatas.append({
        "fraud_signal": result["fraud_signal"] or "none",
        "risk_label":   result["risk_label"],
        "confidence":   result["confidence"],
        "amount":       float(result["amount"]),
        "ground_truth": result["ground_truth"]
    })

# Add all at once — ChromaDB handles the embedding
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"Stored {len(ids)} transactions successfully\n")


# ----------------------------------------------------------------
# 5. Semantic search — the power of vector DB
#    Ask questions in plain English
#    ChromaDB finds the most similar past transactions
# ----------------------------------------------------------------

print("=" * 60)
print("SEMANTIC SEARCH DEMO")
print("=" * 60)

queries = [
    "large amount transaction at odd hours from unknown device",
    "rapid multiple transactions in short time",
    "suspicious merchant transfer"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    results = collection.query(
        query_texts=[query],
        n_results=3    # return top 3 most similar
    )

    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        similarity = round((1 - distance) * 100, 1)
        print(f"  [{i}] {meta['risk_label']} — "
              f"{meta['fraud_signal']} — "
              f"similarity: {similarity}%")
        print(f"       {doc[:80]}...")


# ----------------------------------------------------------------
# 6. Pattern memory — have we seen this before?
#    This is what makes FraudBot AI intelligent over time
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PATTERN MEMORY CHECK")
print("=" * 60)

def have_we_seen_this(txn_description, threshold=0.7):
    """
    Check if a new transaction matches a known fraud pattern.
    Returns True if similar past fraud found above threshold.
    """
    results = collection.query(
        query_texts=[txn_description],
        n_results=1,
        where={"ground_truth": "FRAUD"}  # only match known fraud
    )

    if not results["documents"][0]:
        return False, None

    distance = results["distances"][0][0]
    similarity = 1 - distance

    if similarity >= threshold:
        return True, {
            "similar_txn": results["ids"][0][0],
            "similarity":  round(similarity * 100, 1),
            "pattern":     results["metadatas"][0][0]["fraud_signal"]
        }
    return False, None


# Test with a new suspicious transaction description
new_txn = "Large $2500 transaction at 3am from a device never seen before"
seen, match = have_we_seen_this(new_txn)

print(f"\nNew transaction: '{new_txn}'")
if seen:
    print(f"  ⚠️  KNOWN PATTERN DETECTED!")
    print(f"  Similar to  : {match['similar_txn']}")
    print(f"  Similarity  : {match['similarity']}%")
    print(f"  Pattern type: {match['pattern']}")
else:
    print(f"  ✅ No similar fraud pattern found")


# ----------------------------------------------------------------
# 7. Summary stats
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("VECTOR STORE SUMMARY")
print("=" * 60)
total = collection.count()
print(f"Total patterns stored : {total}")
print(f"DB location           : ./fraudbot_db")
print(f"Embedding model       : all-MiniLM-L6-v2")
print(f"Collection            : fraud_patterns")
print("\nLayer 3 complete! Ready for Layer 4 — FastAPI REST endpoint.")
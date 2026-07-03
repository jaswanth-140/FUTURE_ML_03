import numpy as np
import joblib

def calculate_ndcg(relevance_scores, k=5):
    # NumPy 2.0 Compatible Patch
    actual = np.asarray(relevance_scores, dtype=float)[:k]
    if actual.size == 0: return 0.0
    
    dcg = np.sum((2**actual - 1) / np.log2(np.arange(2, actual.size + 2)))
    
    ideal = np.sort(actual)[::-1]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, ideal.size + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_precision_at_k(relevance_scores, k=5):
    top_k = relevance_scores[:k]
    return sum(top_k) / len(top_k) if len(top_k) > 0 else 0.0

def save_vector_database(embeddings, resume_ids, filepath="resume_vector_db.joblib"):
    print(f"\n💾 Saving Vector Database to {filepath}...")
    db = {"embeddings": embeddings, "resume_ids": resume_ids}
    joblib.dump(db, filepath)
    print("✅ Vector Database Saved in Colab!")

print("✅ Evaluator loaded into memory.")

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

def initialize_models():
    # Dynamically detect hardware
    hardware = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Hardware Check: System locked onto -> [{hardware.upper()}]")
    
    print("⏳ Initializing Stage 1 Bi-Encoder (BAAI/bge-m3)...")
    bi_encoder = SentenceTransformer('BAAI/bge-m3', device=hardware)
    
    print("⏳ Initializing Stage 2 Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', device=hardware)
    
    return bi_encoder, cross_encoder

def stage1_bi_encoder_retrieval(bi_encoder, query_text, corpus_texts, top_k=50):
    query_emb = bi_encoder.encode(query_text, convert_to_tensor=True, show_progress_bar=False)
    
    print(f"    [Status] Embedding {len(corpus_texts)} resumes...")
    # Progress bar enabled, batch size tuned for stability
    corpus_embs = bi_encoder.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=8)
    
    cos_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), corpus_embs)
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus_texts)))
    
    return top_results.indices.cpu().numpy(), top_results.values.cpu().numpy(), corpus_embs.cpu().numpy()

def stage2_cross_encoder_rerank(cross_encoder, query_text, candidate_texts):
    sentence_pairs = [[query_text, cand] for cand in candidate_texts]
    cross_scores = cross_encoder.predict(sentence_pairs, show_progress_bar=False)
    return cross_scores

print("✅ Neural Engine loaded into memory.")

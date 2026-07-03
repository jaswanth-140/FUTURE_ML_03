import numpy as np

print("\n" + "="*50)
print("🚀 INITIATING SOTA RESUME SCREENING PIPELINE")
print("="*50)

# 1. Load Data 
data_file = "ats_resume_dataset_elite_v3.csv"
corpus_df, queries_df = load_and_process_data(data_file)

# --- FAST TEST MODE ---
# Comment out the line below if you want to wait 4 hours to process all 6,000 resumes on a CPU
corpus_df = corpus_df.head(200) 
print(f"⚠️ TEST MODE ENGAGED: Slicing database to {len(corpus_df)} candidates.")

# 2. Boot up System 
bi_encoder, cross_encoder = initialize_models()

all_candidate_texts = corpus_df['candidate_text'].tolist()
all_candidate_ids = corpus_df['resume_id'].tolist()
all_labels = corpus_df['shortlisted'].tolist()

metrics = {"ndcg@5": [], "precision@5": []}
final_embeddings = None

# 3. The Search Engine Loop
print("\n🔍 Running Query Evaluations...\n")
test_jobs = queries_df.head(3)

for idx, job in test_jobs.iterrows():
    job_title = job['job_role']
    job_query = job['job_text']
    print(f"--- Querying for: {job_title} ---")
    
    top_50_indices, _, corpus_embeddings = stage1_bi_encoder_retrieval(
        bi_encoder, job_query, all_candidate_texts, top_k=50
    )
    final_embeddings = corpus_embeddings 
    
    stage1_candidates = [all_candidate_texts[i] for i in top_50_indices]
    stage1_labels = [all_labels[i] for i in top_50_indices]
    
    cross_scores = stage2_cross_encoder_rerank(cross_encoder, job_query, stage1_candidates)
    
    reranked_indices = np.argsort(cross_scores)[::-1] 
    final_top_5_labels = [stage1_labels[i] for i in reranked_indices[:5]]
    
    ndcg = calculate_ndcg(final_top_5_labels, k=5)
    precision = calculate_precision_at_k(final_top_5_labels, k=5)
    
    metrics["ndcg@5"].append(ndcg)
    metrics["precision@5"].append(precision)
    
    print(f"  🏆 nDCG@5:      {ndcg:.4f}")
    print(f"  🎯 Precision@5: {precision:.4f}\n")

print("="*50)
print("📊 FINAL ARCHITECTURE REPORT")
print(f"Mean nDCG@5:      {np.mean(metrics['ndcg@5']):.4f}")
print(f"Mean Precision@5: {np.mean(metrics['precision@5']):.4f}")
print("="*50)

# Save the vector database
save_vector_database(final_embeddings, all_candidate_ids)

# Save variables for the next cells
top_candidate_id_for_report = all_candidate_ids[top_50_indices[reranked_indices[0]]]
job_title_for_report = test_jobs.iloc[-1]['job_role']

import pandas as pd

def load_and_process_data(filepath):
    print("⏳ Loading dataset and executing anti-leakage protocols...")
    df = pd.read_csv(filepath)

    # 1. CRITICAL: Drop Target Leakage Columns
    leakage_cols = [
        "skill_match_score", "experience_match", 
        "education_match", "final_score", "similarity_score"
    ]
    df = df.drop(columns=leakage_cols, errors="ignore")

    # 2. Build the Comprehensive Candidate Profile
    df["candidate_text"] = (
        "Resume: " + df["resume_text"].fillna("") + " | " +
        "Skills: " + df["resume_skills"].fillna("") + " | " +
        "Experience: " + df["experience_years"].astype(str) + " years | " +
        "Education: " + df["education_level"].fillna("") + " | " +
        "Projects: " + df["projects"].fillna("")
    )

    # 3. Build the Target Job Profile
    df["job_text"] = (
        "Job Role: " + df["job_role"].fillna("") + " | " +
        "Required Experience: " + df["job_experience_required"].astype(str) + " years | " +
        "Required Skills: " + df["required_skills"].fillna("") + " | " +
        "Description: " + df["job_description"].fillna("")
    )

    # 4. Isolate Unique Candidates & Jobs
    corpus_df = df[['resume_id', 'candidate_text', 'job_role', 'shortlisted', 'resume_skills']].drop_duplicates(subset=['resume_id'])
    queries_df = df[['job_role', 'job_text', 'required_skills']].drop_duplicates(subset=['job_role'])

    print(f"✅ Processed {len(corpus_df)} unique candidates and {len(queries_df)} unique job queries.")
    return corpus_df, queries_df

print("✅ Data Engine loaded into memory.")

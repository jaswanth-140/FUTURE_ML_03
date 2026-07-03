Markdown
# 🚀 Two-Stage Neural Resume Screening Pipeline

An advanced, AI-driven Applicant Tracking System (ATS) that utilizes a State-of-the-Art (SOTA) Information Retrieval architecture to score, rank, and analyze candidate resumes with semantic precision. 

Unlike traditional ATS systems that rely on rigid keyword matching, this pipeline uses deep learning transformer models to understand the contextual alignment between a candidate's background and a job description.

## ✨ Key Features

*   **Two-Stage Retrieval Architecture:** Implements a highly scalable pipeline used by modern search engines (Dense Retrieval + Cross-Encoder Reranking).
*   **Semantic Understanding:** Captures the nuanced meaning of experience and project descriptions rather than just counting keywords.
*   **Zero-Shot Generalization:** Capable of evaluating resumes for entirely new job categories it has never explicitly been trained on.
*   **Explainable AI (Skill Gap Analysis):** Includes a logic layer that clearly outputs exactly which required skills a top candidate possesses and which they lack.
*   **Hardware Agnostic:** Auto-detects system capabilities to run on either a dedicated GPU (CUDA) or a standard CPU.

---

## 🧠 System Architecture

This system avoids the "Tabular Trap" of standard classification models by treating resume screening as an Information Retrieval (IR) problem, executed in three distinct stages:

### Stage 1: Fast Retrieval (The Net)
*   **Model:** `BAAI/bge-m3` (Bi-Encoder)
*   **Process:** Instantly scans the entire database of candidates. It converts the job description and all resumes into mathematical vectors and uses Cosine Similarity to retrieve the Top 50 most relevant candidates in milliseconds.

### Stage 2: Deep Reranking (The Scalpel)
*   **Model:** `BAAI/bge-reranker-v2-m3` (Cross-Encoder)
*   **Process:** Performs a deep, word-by-word attention calculation on the Top 50 candidates retrieved in Stage 1. It meticulously reranks them to output the definitive Top 5 best fits.

### Stage 3: Explainability (Skill Verification)
*   **Process:** Evaluates the #1 ranked candidate against the job description using strict Set Theory, generating a recruiter-friendly report highlighting explicit skill matches and missing requirements.

---

## 📊 Performance Metrics

The architecture was evaluated using standard search engine ranking metrics. The system demonstrates exceptional ability to push the absolute best candidates to the #1 and #2 spots.

*   **Mean nDCG@5:** `0.9823` *(Ranking Quality)*
*   **Mean Precision@5:** `0.3333` *(Accuracy of Top 5)*

*(Note: See `pipeline_performance_chart.png` in the repository for a visual breakdown of performance across different job roles).*

---

## 📂 Repository Structure

The project is modularized for production engineering standards:
├── data_processor.py      # Data ingestion, cleaning, and semantic text building
├── neural_engine.py       # PyTorch logic, GPU allocation, and Transformer models
├── metrics.py             # Evaluator (nDCG/Precision math) and Skill Gap Analyzer
├── main.py                # Main orchestrator loop
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
💻 Installation & Usage
1. Clone the repository
Bash
git clone [https://github.com/YourUsername/Neural-Resume-Screener.git](https://github.com/YourUsername/Neural-Resume-Screener.git)
cd Neural-Resume-Screener
2. Install dependencies
Bash
pip install -r requirements.txt
3. Run the Pipeline
Ensure your candidate dataset is in the root directory, then execute the main orchestrator:

Bash
python main.py
Note: The system includes a FAST TEST MODE that slices the database to 200 candidates for rapid CPU evaluation. To run the full database, remove or comment out the slice command in main.py.

💾 Outputs
Upon successful execution, the pipeline generates:

Console Report: Real-time nDCG and Precision scores per job query.

Skill Gap Analysis: A printed breakdown of the #1 candidate's skill intersections.

Vector Database (resume_vector_db.joblib): The fully embedded candidate pool, serialized for sub-second production queries without requiring model retraining.

Developed by Chellamlla Jaswanth Reddy

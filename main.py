from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load Data ===
freelancers_df = pd.read_csv("freelancers.csv")

# === Fit TF-IDF on freelancer skills ===
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(freelancers_df["skills"])

# === FastAPI App ===
app = FastAPI(title="Freelancer Recommendation API")

# === Input Schema ===
class JobInput(BaseModel):
    required_skills: List[str]
    budget: int
    timeline_days: int

# === Endpoint ===
@app.post("/recommend")
def recommend(job: JobInput):
    job_skills_str = ", ".join(job.required_skills)
    job_tfidf = vectorizer.transform([job_skills_str])
    cosine_similarities = cosine_similarity(job_tfidf, tfidf_matrix).flatten()

    freelancers_df["similarity"] = cosine_similarities
    weekly_budget = job.budget / (job.timeline_days / 7)

    filtered = freelancers_df[freelancers_df["rate"] <= weekly_budget]
    top_matches = filtered.sort_values(by="similarity", ascending=False).head(5)

    return top_matches[["name", "skills", "experience", "rate", "past_projects", "similarity"]].to_dict(orient="records")
